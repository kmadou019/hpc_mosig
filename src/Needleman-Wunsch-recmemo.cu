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

#include "characters_to_base.h" /* mapping from char to base */

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

long min(long a, long b) {
   return (a < b) ? a : b;
}


void computeBlock(long begin_1, long end_1, long begin_2, long end_2, 
            long *tab, long *tab_dessus, long *tab_droite, long *last_i1j1,
            struct NW_NoMemoContext *c) {
   long last = 0;
   for (long i = begin_1; i > end_1; i--) {
      for (long j = begin_2; j > end_2; j--) {
         long val_i1 = tab[j]; // valeur (i+1, j)
         if (i == begin_1) val_i1 = tab_droite[j];

         long val_j1 = tab[j + 1]; // valeur (i, j+1)
         if (j == begin_2) val_j1 = tab_dessus[i];

         long curr = tab[j];

         if (!isBase(c->X[i])) tab[j] = val_i1;

         else if (!isBase(c->Y[j])) tab[j] = val_j1;
            
         else {
            long val_i1j1 = last; // valeur (i+1, j+1)
            if (i == begin_1 && j == begin_2) val_i1j1 = *last_i1j1;
            else if (i == begin_1) val_i1j1 = tab_droite[j + 1];

            long min = (isUnknownBase(c->X[i]) ? SUBSTITUTION_UNKNOWN_COST :
                          (isSameBase(c->X[i], c->Y[j]) ? 0 : SUBSTITUTION_COST)) + val_i1j1; 
            
            long val = INSERTION_COST + val_i1;      
            if (val < min) min = val;

            val = INSERTION_COST + val_j1;      
            if (val < min) min = val;

            tab[j] = min;
         }

         if (i == end_1 + 1 && j == end_2 + 1) *last_i1j1 = tab_droite[j];

         last = curr;
         if (j == end_2 + 1) last = tab_dessus[i];

         if (j == end_2 + 1) tab_dessus[i] = tab[j];

         if (i == end_1 + 1) tab_droite[j] = tab[j];

      }
   }

}


long EditDistance_NW_CacheAware(char *A, size_t lengthA, char *B, size_t lengthB,
      int K1, int K2) {
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
   long *tab = malloc(sizeof(long) * (N+1));
   long *tab_droite = malloc(sizeof(long) * (N+1));
   long *tab_dessus = malloc(sizeof(long) * (M+1));
   long last_i1j1 = 0;

   tab_droite[N] = 0;
   for (long j = N - 1; j >= 0; j--) {
      tab_droite[j] = 2 * isBase(c->Y[j]) + tab_droite[j + 1];
   }
   tab_dessus[M] = 0;
   for (long i = M - 1; i >= 0; i--) {
      tab_dessus[i] = 2 * isBase(c->X[i]) + tab_dessus[i + 1];
   }

   for (long I = M - 1; I >= 0; I -= K1) {
      long i_end = (I - K1 < 0) ? -1 : I - K1;
      for (long J = N - 1; J >= 0; J -= K2) {
         long j_end = (J - K2 < 0) ? -1 : J - K2;

         computeBlock(I, i_end, J, j_end, 
            tab, tab_dessus, tab_droite, &last_i1j1, c);

      }
   }

   long res = tab[0];

   free(tab);
   free(tab_droite);
   free(tab_dessus);

   return res;
}
