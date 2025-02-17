#ifndef __UTILS_h
#define __UTILS_h

#define INSERTION_COST		2

__global__ void prescan(long *d_out, long *d_in, long n);

#endif