/**
 * \file Needleman-Wunsch-recmemo.c
 * \brief recursive implementation with memoization of Needleman-Wunsch global alignment algorithm that computes the distance between two genetic sequences 
 * \version 0.1
 * \date 03/10/2022 
 * \author Jean-Louis Roch (Ensimag, Grenoble-INP - University Grenoble-Alpes) jean-louis.roch@grenoble-inp.fr
 *
 * Documentation: see Needleman-Wunsch-recmemo.h
 * Costs of basic base opertaions (SUBSTITUTION_COST, SUBSTITUTION_UNKNOWN_COST, INSERTION_COST) are
 * defined in Needleman-Wunsch-recmemo.h
 */


#include "Needleman-Wunsch-recmemo.h"
#include <stdio.h>  
#include <stdlib.h> 
#include <stdbool.h>
#include <string.h> /* for strchr */
// #include <math.h>
// #include <ctype.h> /* for toupper */
#include "utils.h"

#include "characters_to_base.h" /* mapping from char to base */

#define THREADS_PER_BLOCK 8

/*****************************************************************************/
   
/* Context of the memoization : passed to all recursive calls */
/** \def NOT_YET_COMPUTED
 * \brief default value for memoization of minimal distance (defined as an impossible value for a distance, -1).
 */
#define NOT_YET_COMPUTED -1L 

/** \struct NW_MemoContext
 * \brief data for memoization of recursive Needleman-Wunsch algorithm 
*/
struct NW_MemoContext 
{
    char *X ; /*!< the longest genetic sequences */
    char *Y ; /*!< the shortest genetic sequences */
    size_t M; /*!< length of X */
    size_t N; /*!< length of Y,  N <= M */
    long **memo; /*!< memoization table to store memo[0..M][0..N] (including stopping conditions phi(M,j) and phi(i,N) */
} ;

struct NW_NoMemoContext {
   char *X;
   char *Y;
   size_t M;
   size_t N;
};

#define min(a, b) ( (a < b) ? (a) : (b))

__global__ void pre_compute_min(long* tab_in, long* tab_out, long N, long i, struct NW_NoMemoContext *c){

   int j =   blockDim.x * blockIdx.x + threadIdx.x;
   
   if(j < N + 1){
      if(j == N ) tab_out[j] = 2 * isBase(c->X[i]) + tab_in[j];
      
      else if (!isBase(c->X[i])) tab_out[j] = tab_in[j];

      else if (!isBase(c->Y[j])) tab_out[j] = tab_out[j + 1]; // problem
      
      else{
         long min_tmp = (isUnknownBase(c->X[i]) ? SUBSTITUTION_UNKNOWN_COST :
                     (isSameBase(c->X[i], c->Y[j]) ? 0 : SUBSTITUTION_COST)) + tab_in[j+1];

         tab_out[j] = min(INSERTION_COST + tab_in[j], min_tmp );
      } 
         
   }


}

long EditDistance_NW_It(char *A, size_t lengthA, char *B, size_t lengthB) {
   _init_base_match();
   
   struct NW_NoMemoContext ctx;
   if (lengthA >= lengthB) {
      ctx.X = A;
      ctx.M = lengthA;
      ctx.Y = B;
      ctx.N = lengthB;
   } else {
      ctx.X = B;
      ctx.M = lengthB;
      ctx.Y = A;
      ctx.N = lengthA;
   }
   struct NW_NoMemoContext *c = &ctx;
   
   const long M = c->M;
   const long N = c->N;
   const long size = sizeof(long) * (N+1);
   const long size_masked = sizeof(long) * (N+1+1);
   long *h_tab_in = (long*)malloc(size_masked);
   long *h_tab_out= (long*)malloc(size_masked);
   
   long *d_tab_in ;
   long *d_tab_out;

   // alloc mem on GPU
   cudaMalloc( (void**)&d_tab_in, size  );
   cudaMalloc( (void**)&d_tab_out, size );

   cudaMemcpy(d_tab_in, h_tab_in, size, cudaMemcpyHostToDevice);
   
   
   h_tab_in[N] = 0;
   for (long j = N - 1; j >= 0; j--) {
      h_tab_in[j] = 2 * isBase(c->Y[j]) + h_tab_in[j + 1];
   }
   
   for (long i = M - 1; i >= 0; i--) {

      pre_compute_min<<< ceil(N/THREADS_PER_BLOCK) , THREADS_PER_BLOCK>>>(d_tab_in, d_tab_out, N, i,c);

      h_tab_out[N+1] = h_tab_out[N];
      cudaMemcpy(d_tab_out, h_tab_out, size, cudaMemcpyHostToDevice);
      
      //val = INSERTION_COST + tab[j + 1];    // necessary for the scan 

      prescan<<<ceil(N/THREADS_PER_BLOCK) , THREADS_PER_BLOCK>>>(h_tab_out,h_tab_out, N+1);

      //permute tab_in and tab_out

      long *temp = h_tab_in;
      h_tab_in   = h_tab_out;
      h_tab_out  = temp;


   }

   long res = h_tab_in[0]; // Normally it has to be h_tab_out[0] but there is a permutation just before

   return res;
}



//long EditDistance_NW_It(char *A, size_t lengthA, char *B, size_t lengthB) {
//   _init_base_match();
//   
//   struct NW_NoMemoContext ctx;
//   if (lengthA >= lengthB) {
//      ctx.X = A;
//      ctx.M = lengthA;
//      ctx.Y = B;
//      ctx.N = lengthB;
//   } else {
//      ctx.X = B;
//      ctx.M = lengthB;
//      ctx.Y = A;
//      ctx.N = lengthA;
//   }
//   struct NW_NoMemoContext *c = &ctx;
//   
//   const long M = c->M;
//   const long N = c->N;
//   long *tab = malloc(sizeof(long) * (N+1));
//   long last = 0;
//   
//   tab[N] = 0;
//   for (long j = N - 1; j >= 0; j--) {
//      tab[j] = 2 * isBase(c->Y[j]) + tab[j + 1];
//   }
//
//   for (long i = M - 1; i >= 0; i--) {
//      for (long j = N; j >= 0; j--) {
//         long curr = tab[j];
//
//         if (j == N) tab[j] = 2 * isBase(c->X[i]) + tab[j];
//
//         else if (!isBase(c->X[i])) tab[j] = tab[j];
//
//         else if (!isBase(c->Y[j])) tab[j] = tab[j + 1];
//            
//         else {
//            long min = (isUnknownBase(c->X[i]) ? SUBSTITUTION_UNKNOWN_COST :
//                          (isSameBase(c->X[i], c->Y[j]) ? 0 : SUBSTITUTION_COST)) + last; 
//            
//            long val = INSERTION_COST + tab[j];   //   
//            if (val < min) min = val;
//
//            val = INSERTION_COST + tab[j + 1];    //  
//            if (val < min) min = val;
//
//            tab[j] = min;
//         }
//
//         last = curr;
//      }
//   }
//
//   long res = tab[0];
//
//   free(tab);
//
//   return res;
//}
//