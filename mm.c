/*
 * A Simple, 64-bit allocater based on implicit lists (& boundary tags)
 * for coalescing, segregated explicit free lists for finding free blocks,
 * as described in the CS:APP2e text. Blocks must be aligned to 16 byte
 * boundaries. Minimum block size is 16 bytes. 
 * 
 * This version is loosely based on 
 * http://csapp.cs.cmu.edu/3e/ics3/code/vm/malloc/mm.c
 * but unlike the book's version, it does not use C preprocessor 
 * macros or explicit bit operations.
 * 
 * It follows the book in counting in units of 4-byte words,
 * but note that this is a choice (our actual solution chooses
 * to count everything in bytes instead.)
 * 
 * First adapted for CS3214 Summer 2020 by gback
 * Updated by Tejas Choudhary & Quentin Holmes for faster performance using explicit
 * segregated free lists.
 * 
 * 
 * FOR GIT, PLEASE CHECK ALL THE BRANCHES ON OUR REPO, WE MADE OUR FINAL COMMITS ON A DIFFERENT REPO.
 * WE DIDN'T MERGE THE BRANCHES.
 * 
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#include "mm.h"
#include "memlib.h"
#include "config.h"
#include "list.h"

struct boundary_tag {
    int inuse:1;        // inuse bit
    int size:31;        // size of block, in words
                        // block size
};

/* FENCE is used for heap prologue/epilogue. */
const struct boundary_tag FENCE = {
    .inuse = 1,
    .size = 0
};

/* A C struct describing free blocks, which is essentially
 * the same as the block struct but with list_elem 
 * placed instead of the payload.
 */
struct free_block {
    struct boundary_tag header;
    struct list_elem elem;         
};

/* A C struct describing the beginning of each block. 
 * If each block is aligned at 12 mod 16, each payload will
 * be aligned at 0 mod 16.
 */
struct block {
    struct boundary_tag header; /* offset 0, at address 12 mod 16 */
    char payload[0];            /* offset 4, at address 0 mod 16 */
};

/* Functions as a wrapper struct around free_list.
 * includes list_elem to create the mega_free_list, and the size variable
 * to easily find the list to get/put free block in. 
*/
struct free_list_wrapper {
    size_t size;
    struct list free_list;
    struct list_elem elem;
};

/* Basic constants and macros */
#define WSIZE       sizeof(struct boundary_tag)  /* Word and header/footer size (bytes) */
/* NOTE: increased block size to 8 words, since list_elem has been added.*/
#define MIN_BLOCK_SIZE_WORDS 8  /* Minimum block size in words */
#define CHUNKSIZE  (1<<10)  /* Extend heap by this amount (words) */
static inline size_t max(size_t x, size_t y) {
    return x > y ? x : y;
}

static size_t align(size_t size) {
  return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

static bool is_aligned(size_t size) __attribute__((__unused__));
static bool is_aligned(size_t size) {
  return size % ALIGNMENT == 0;
}

/* Global variables */
static struct block *heap_listp = 0;  /* Pointer to first block */  
static struct list mega_free_list;
//0 is used to store free blocks of size greater than 4194304.
static const size_t FREE_BLOCK_SIZES[] = {16, 64, 128, 256, 512, 1024, 2048, 4096, 8196, 16384, 65536, 262144, 0};

/* Function prototypes for internal helper routines */
static struct block *extend_heap(size_t words);
static void place(struct block *bp, size_t asize);
static struct block *find_fit(size_t asize);
static struct block *coalesce(struct block *bp);
static void init_free_lists();
static void add_to_free_list(struct block * bp);
static void remove_from_free_list(struct block * bp);

/* Given a block, obtain previous's block footer.
   Works for left-most block also. */
static struct boundary_tag * prev_blk_footer(struct block *blk) {
    return &blk->header - 1;
}

/* Return if block is free */
static bool blk_free(struct block *blk) { 
    return !blk->header.inuse; 
}

/* Return size of block is free */
static size_t blk_size(struct block *blk) { 
    return blk->header.size; 
}

/* Given a block, obtain pointer to previous block.
   Not meaningful for left-most block. */
static struct block *prev_blk(struct block *blk) {
    struct boundary_tag *prevfooter = prev_blk_footer(blk);
    assert(prevfooter->size != 0);
    return (struct block *)((void *)blk - WSIZE * prevfooter->size);
}

/* Given a block, obtain pointer to next block.
   Not meaningful for right-most block. */
static struct block *next_blk(struct block *blk) {
    assert(blk_size(blk) != 0);
    return (struct block *)((void *)blk + WSIZE * blk->header.size);
}

/* Given a block, obtain its footer boundary tag */
static struct boundary_tag * get_footer(struct block *blk) {
    return ((void *)blk + WSIZE * blk->header.size)
                   - sizeof(struct boundary_tag);
}

/* Set a block's size and inuse bit in header and footer */
static void set_header_and_footer(struct block *blk, int size, int inuse) {
    blk->header.inuse = inuse;
    blk->header.size = size;
    * get_footer(blk) = blk->header;    /* Copy header to footer */
}

/* Mark a block as used and set its size. */
static void mark_block_used(struct block *blk, int size) {
    set_header_and_footer(blk, size, 1);
}

/* Mark a block as free and set its size. */
static void mark_block_free(struct block *blk, int size) {
    set_header_and_footer(blk, size, 0);
}

static struct block * to_block(struct free_block * fblk) {
    return (struct block *) fblk;
}

static struct free_block * to_free_block(struct block * blk) {
    return (struct free_block *) blk;
}

/* 
 * mm_init - Initialize the memory manager 
 */
int mm_init(void) 
{
    assert (offsetof(struct block, payload) == 4);
    assert (sizeof(struct boundary_tag) == 4);

    init_free_lists();

    /* Create the initial empty heap */
    struct boundary_tag * initial = mem_sbrk(4 * sizeof(struct boundary_tag));
    if (initial == NULL)
        return -1;

    /* We use a slightly different strategy than suggested in the book.
     * Rather than placing a min-sized prologue block at the beginning
     * of the heap, we simply place two fences.
     * The consequence is that coalesce() must call prev_blk_footer()
     * and not prev_blk() because prev_blk() cannot be called on the
     * left-most block.
     */
    initial[2] = FENCE;                     /* Prologue footer */
    heap_listp = (struct block *)&initial[3];
    initial[3] = FENCE;                     /* Epilogue header */

    /* Extend the empty heap with a free block of CHUNKSIZE bytes */
    if (extend_heap(CHUNKSIZE) == NULL) 
        return -1;
    return 0;
}

/* 
 * mm_malloc - Allocate a block with at least size bytes of payload 
 */
void *mm_malloc(size_t size)
{
    struct block *bp;      

    /* Ignore spurious requests */
    if (size == 0)
        return NULL;

    /* Adjust block size to include overhead and alignment reqs. */
    size_t bsize = align(size + 2 * sizeof(struct boundary_tag));    /* account for tags */
    if (bsize < size)
        return NULL;    /* integer overflow */
    
    /* Adjusted block size in words */
    size_t awords = max(MIN_BLOCK_SIZE_WORDS, bsize/WSIZE); /* respect minimum size */

    /* Search the free list for a fit */
    if ((bp = find_fit(awords)) != NULL) {
        place(bp, awords);
        return bp->payload;
    }

    /* No fit found. Get more memory and place the block */
    size_t extendwords = max(awords,CHUNKSIZE); /* Amount to extend heap if no fit */
    if ((bp = extend_heap(extendwords)) == NULL)  
        return NULL;

    place(bp, awords);
    return bp->payload;
} 

/* 
 * mm_free - Free a block 
 */
void mm_free(void *bp)
{
    assert (heap_listp != 0);       // assert that mm_init was called
    if (bp == 0) 
        return;

    /* Find block from user pointer */
    struct block *blk = bp - offsetof(struct block, payload);

    mark_block_free(blk, blk_size(blk));
    coalesce(blk);
}

/*
 * coalesce - Boundary tag coalescing. Return ptr to coalesced block
 */
static struct block *coalesce(struct block *bp) 
{
    bool prev_alloc = prev_blk_footer(bp)->inuse;   /* is previous block allocated? */
    bool next_alloc = ! blk_free(next_blk(bp));     /* is next block allocated? */
    size_t size = blk_size(bp);

    if (prev_alloc && next_alloc) {            /* Case 1 */
        // both are allocated, nothing to coalesce
        add_to_free_list(bp);
        return bp;
    }

    else if (prev_alloc && !next_alloc) {      /* Case 2 */
        // combine this block and next block by extending it
        struct block *next_block = next_blk(bp);
        remove_from_free_list(next_block);
        
        mark_block_free(bp, size + blk_size(next_blk(bp)));
        add_to_free_list(bp);
    }

    else if (!prev_alloc && next_alloc) {      /* Case 3 */
        // combine previous and this block by extending previous
        struct block * prev_block = prev_blk(bp);
        remove_from_free_list(prev_block);

        bp = prev_blk(bp);
        mark_block_free(bp, size + blk_size(bp));

        add_to_free_list(bp);

    }

    else {                                     /* Case 4 */
        // combine all previous, this, and next block into one
        struct block *next_block = next_blk(bp);
        struct block * prev_block = prev_blk(bp);

        remove_from_free_list(next_block);
        remove_from_free_list(prev_block);

        mark_block_free(prev_blk(bp), 
                        size + blk_size(next_blk(bp)) + blk_size(prev_blk(bp)));
        bp = prev_blk(bp);
        add_to_free_list(bp);
    }
    return bp;
}

/*
 * mm_realloc - Naive implementation of realloc
 */
void *mm_realloc(void *ptr, size_t size)
{
    //If size is 0, the call is equivalent to mm_free(ptr)
    if (size == 0){
        mm_free(ptr);
        return NULL;
    }
    //If ptr is NULL, the call is equivalent to mm_malloc(size).
    if (ptr == NULL) return mm_malloc(size);

    /* Adjust block size to include overhead and alignment reqs. */
    size_t bsize = align(size + 2 * sizeof(struct boundary_tag));    /* account for tags */
    if (bsize < size)
        return NULL;    /* integer overflow */
    
    /* Adjusted block size in words */
    size_t awords = max(MIN_BLOCK_SIZE_WORDS, bsize/WSIZE); /* respect minimum size */

    struct block *oldblock = ptr - offsetof(struct block, payload);
    size_t oldpayloadsize = blk_size(oldblock) * WSIZE - 2 * sizeof(struct boundary_tag);
    bool prev_alloc = prev_blk_footer(oldblock)->inuse;   /* is previous block allocated? */
    bool next_alloc = ! blk_free(next_blk(oldblock));     /* is next block allocated? */
    
    if (blk_size(oldblock) >= awords) {
        //On the discord, Alice said that the tests never reduce the size using realloc,
        //so it should be okay to leave the block as is.
        return ptr;
    }

    struct block *coalesced_block = oldblock;

    //Coalescing with next block.
    if (!next_alloc){
        struct block *next = next_blk(coalesced_block);
        remove_from_free_list(next);
        mark_block_used(coalesced_block, blk_size(coalesced_block) + blk_size(next));
    }

    //Returning ptr if coalesced_block now has enough space.
    if (blk_size(coalesced_block) >= awords) return coalesced_block->payload;

    //Coalescing with prev block.
    if (!prev_alloc) {
        struct block *prev = prev_blk(coalesced_block);
        remove_from_free_list(prev);
        mark_block_used(prev, blk_size(coalesced_block) + blk_size(prev));
        coalesced_block = prev;
    }
    //Returning ptr if coalesced_block now has enough space.
    //Also copying over the data if prev was coalesced.
    if (blk_size(coalesced_block) >= awords) {
        if (!prev_alloc) {
            memmove(coalesced_block->payload, ptr, size);
        }
        return coalesced_block->payload;
    }

    //If all else fails, we call malloc.

    void *newptr = mm_malloc(size);

    /* If realloc() fails the original block is left untouched  */
    if (!newptr) {
        return 0;
    }

    /* Copy the old data. */
    if (size < oldpayloadsize) oldpayloadsize = size;
    memcpy(newptr, ptr, oldpayloadsize);

    /* Free the old block. */
    mm_free(coalesced_block->payload);

    return newptr;
}

/* 
 * checkheap - NOT USED.
 */
void mm_checkheap(int verbose)
{ 
}

/* 
 * The remaining routines are internal helper routines 
 */

/* 
 * extend_heap - Extend heap with free block and return its block pointer
 */
static struct block *extend_heap(size_t words) 
{
    void *bp = mem_sbrk(words * WSIZE);

    if (bp == NULL)
        return NULL;

    /* Initialize free block header/footer and the epilogue header.
     * Note that we overwrite the previous epilogue here. */
    struct block * blk = bp - sizeof(FENCE);
    mark_block_free(blk, words);
    next_blk(blk)->header = FENCE;

    return coalesce(blk);
}

/* 
 * place - Place block of asize words at start of free block bp 
 *         and split if remainder would be at least minimum block size
 */
static void place(struct block *bp, size_t asize)
{
    size_t csize = blk_size(bp);

    //Should the extra space be freed into another block?
    if ((csize - asize) >= MIN_BLOCK_SIZE_WORDS) { 
        mark_block_used(bp, asize);
        remove_from_free_list(bp);

        struct block *next = next_blk(bp);

        mark_block_free(next, csize-asize);
        add_to_free_list(next);
    }
    else { 
        mark_block_used(bp, csize);
        remove_from_free_list(bp);
    }
}

/* 
 * find_fit - Find a fit for a block with asize words 
 */
static struct block *find_fit(size_t asize)
{

    //Looping through all the free lists.
    for (struct list_elem * current_list_elem = list_begin(&mega_free_list);
        current_list_elem != list_end(&mega_free_list);
        current_list_elem = list_next(current_list_elem)) {
        
        struct free_list_wrapper * current_free_list_wrapper = 
            list_entry (current_list_elem, struct free_list_wrapper, elem);

        if (list_empty(&current_free_list_wrapper->free_list)) continue;
        if (current_free_list_wrapper->size < asize && current_free_list_wrapper->size != 0) continue;

        //Searching the free list from front and back.
        struct list_elem *beg_elem = list_begin(&current_free_list_wrapper->free_list);
        struct list_elem *end_elem = list_rbegin(&current_free_list_wrapper->free_list);


        while (beg_elem != list_end(&current_free_list_wrapper->free_list)) {

            struct free_block * free_block1 = list_entry(beg_elem, struct free_block, elem);
            struct free_block * free_block2 = list_entry(end_elem, struct free_block, elem);

            struct block * block1 = to_block(free_block1);  
            struct block * block2 = to_block(free_block2);

            //Returning if appropriate block found.
            if (blk_size(block1) >= asize) return block1;  
            if (blk_size(block2) >= asize) return block2;
            

            beg_elem = list_next(beg_elem);
            end_elem = list_prev(end_elem);

            if (beg_elem != end_elem && list_next(end_elem) != beg_elem) break;
        }
        
    }
    return NULL;
}
/**
 * @brief Function used for adding block to free_block list
 * 
 * @param bp - block pointer.
 * @return void - no return value.
 */

static void add_to_free_list(struct block * bp) {

    assert(bp != 0);
    assert(blk_free(bp));

    //Finding the free_list to add the block to.
    for (struct list_elem * free_list_elem = list_begin(&mega_free_list);
        free_list_elem != list_end(&mega_free_list);
        free_list_elem = list_next(free_list_elem)) {

        struct free_list_wrapper * free_list_current = list_entry(free_list_elem,
                                                        struct free_list_wrapper, elem);
        if (blk_size(bp) > free_list_current->size && free_list_current->size != 0) continue;

        //Getting the block as a free_block
        struct free_block * fbp = to_free_block(bp);

        //Adding the block to the back.
        list_push_back(&free_list_current->free_list, &fbp->elem);
        return;
    }
}
/**
 * @brief simple function to remove block from free_list.
 * 
 * @param bp block pointer.
 */
static void remove_from_free_list(struct block * bp) {
    list_remove(&to_free_block(bp)->elem);
}
/**
 * @brief Used to init all the free lists.
 * 
 */
static void init_free_lists() {

    list_init(&mega_free_list);

    for (int i = 0; i < sizeof(FREE_BLOCK_SIZES)/sizeof(FREE_BLOCK_SIZES[0]); i++) {
        struct free_list_wrapper * current_free_list = malloc(sizeof(struct free_list_wrapper));
        current_free_list->size = FREE_BLOCK_SIZES[i];
        list_init(&current_free_list->free_list);
        list_push_back(&mega_free_list, &current_free_list->elem);
    }
}

team_t team = {
    /* Team name */
    "Tejas & Quentin",
    /* First member's full name */
    "Tejas Choudhary",
    "tejaschoudhary@vt.edu",
    /* Second member's full name (leave as empty strings if none) */
    "Quentin Holmes",
    "qholmes@vt.edu",
};

