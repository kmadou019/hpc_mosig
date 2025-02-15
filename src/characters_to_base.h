/**
 * \file characters_to_base.h 
 * \brief mapping of characters to canonic ADN/ARN bases 
 * \version 0.1
 * \date 03/10/2022
 * \author Jean-Louis Roch (Ensimag, Grenoble-INP - University Grenoble-Alpes) jean-louis.roch@grenoble-inp.fr
 * 
 * Mapping of characters (eg 'a' and 'A' to ADENINE is computed through a pre_computed table, 
 * initialized by 
 */

#ifndef __CHARACTERS_TO_BASE_h__
#define __CHARACTERS_TO_BASE_h__

/**********************************************/
/* Matching between chars and bases 
*/
/**
 * \enum  Base
 * \brief canonical bases
 *
 * Bases is the set of canonical bases {ADENINE, CYTOSINE, GUANINE, THYMINE, URACILE} plus
 * two additional constants :  
 *    UNKOWN_BASE : corresponding to an existing but unknown or erroneous base (represented by 'N' in FASTA file)
 *    SKIP_BASE : corresponding to a character in the file that has to be skip (like '\n') /
 */
enum Base 
{ 
   SKIP_BASE=0,	/*!< unexpected char, different from the ones below: skip in the computation. */ 
   ADENINE,	/*!< Adenine: matchs char 'a' and 'A'. */
   CYTOSINE, 	/*!< Cytosine: matchs char 'c' and 'C'. */
   GUANINE, 	/*!< Guanine: matchs char 'g' and 'G'. */
   THYMINE, 	/*!< Thymine: matchs char 't' and 'T'. */
   URACILE, 	/*!< Uracile: matchs char 'u' and 'U'. */
   UNKOWN_BASE 	/*!< Unknown or erroneous base: matchs char 'n' and 'N' in FASTA files */
} ;

/** \var static enum Base  _base_match[256]
 * 
 * \brief _base_match maps directly a char to its corresponding base 
 */ 
__constant__ enum Base  _base_match[256]; /* initially all chars are ignored */
/**
 * \fn static void _init_base_match()
 * \brief definition of the  mapping from char to base
 *
 * Has to be called once before any computation for identification of a char as a base
 */
static void  _init_base_match() /* initialisation of _base_match array for correspondence from char to base */
{ 
   enum Base h_base_match[256];

   for (int c = 0; c < 256; ++c) h_base_match[c] = SKIP_BASE ;
   h_base_match['a'] = ADENINE ;
   h_base_match['A'] = ADENINE ;
   h_base_match['c'] = CYTOSINE ;
   h_base_match['C'] = CYTOSINE ;
   h_base_match['g'] = GUANINE ;
   h_base_match['G'] = GUANINE ;
   h_base_match['t'] = THYMINE ;
   h_base_match['T'] = THYMINE ;
   h_base_match['u'] = URACILE ;
   h_base_match['U'] = URACILE ;
   h_base_match['n'] = UNKOWN_BASE ;
   h_base_match['N'] = UNKOWN_BASE ;

   cudaMemcpyToSymbol(_base_match, h_base_match, sizeof(h_base_match)); 
}


/**
 * \def CharToBase(c)
 * \brief retuns the Base (among enum Base} that matches character c 
 */
#define CharToBase(c)	( _base_match[c] )

/**
 * \def isBase(c)
 * \brief retuns 0 iff char c match does not match a base 
 * i.e. c matches a base which is either known (A,C,G,T,U) or unknown (N)
 */
#define isBase(c)	( _base_match[c] != SKIP_BASE )

/**
 * \def isUnknownBase(c)
 * \brief retuns 0 iff char c match does not match an unknown  base ('n' or 'N')
 * i.e. c matches a base which is A,C,G,T,U
 */
#define isUnknownBase(c)	( _base_match[c] == UNKOWN_BASE )

/**
 * \def isSameBase(a,b)
 * \brief retuns 0 iff chars a and b are mapped to two different bases 
 */
#define isSameBase(a,b)	( _base_match[a] == _base_match[b] )

/** \enum BASE_ERROR_TREATMENT_MODE
 * \brief  BASE_ERROR_TREATMENT defines way a char not in AaCcGgTtUuNn is processed; either IGNORED (default), or WARNING (prints a message on stderr), or EROOR (stops execution). 
 */
enum BASE_ERROR_TREATMENT_MODE { IGNORED = 0, WARNING = 1, ERROR=2  } ;

/** 
 * \fn void ManageBaseError(char c)
 * \brief according to BASE_ERROR_TREATMENT prints on stderr either nothing, or a warning or an error if the char passed as argument is not a base (known or unknown) nor a space char
 * \param c the character 
 *
 * If BASE_ERROR_TREATMENT is undefined: ManageBaseError(c) does nothing (just return)
 * else, switch ( BASE_ERROR_TREATMENT ) :
 *   BASE_WARNING : if c is neither a base nor a space, then prints a warning with c on stderr 
 *   BASE_ERROR   : if c is neither a base nor a space, then prints an error with c on stderr and exit
 *   default : does nothing (just return)
*/
void ManageBaseError(char c)
{ 
   #ifdef BASE_ERROR_TREATMENT
   {  if (isBase(c)) return ; // no error
      if (isspace(c)) return ; // \n or \t or ' ' etc are  ignored 
      switch ( BASE_ERROR_TREATMENT )
      {  case BASE_WARNING : 
            fprintf(stderr, "Warning: character %c=0x%xc not matching a base is skipped.\n", c, c ) ;
            break;
         case BASE_ERROR :
            fprintf(stderr, 
               "Error: character %c=0x%xc  not matching a base is skipped.\n"
               "   Expected bases:A=0x%xc, a=0x%xc, C=0x%xc, c=0x%xc, G=0x%xc, g=0x%xc, T=0x%xc, t=0x%xc, U0x%xc, u=0x%xc\n",
               c, c, 'A', 'a', 'C', 'c', 'G', 'g', 'T', 't', 'U', 'u' 
            ) ;
            exit(EXIT_FAILURE) ;
         default: break ;
      }
   }
   #endif
   /* else no errors are managed */
   return;
} 

#endif /* __CHARACTERS_TO_BASE_h__ */
