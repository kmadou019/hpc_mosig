#include <iostream>

#define N 8
#define THREADS_PER_BLOCK 8

__global__ void prescan(long *g_idata, long n)
{
  extern __shared__ long temp[];
  // allocated on invocation
  int thid = threadIdx.x;
  long offset = 1;

  temp[thid] = g_idata[thid]; // load input into shared memory

  for (long d = n >> 1; d > 0; d = d >> 1)
  // build sum in place up the tree
  {
    __syncthreads();
    if (thid < d)
    {

      long ai = offset * (2 * thid + 1) - 1;
      long bi = offset * (2 * thid + 2) - 1;

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  if (thid == 0)
  {
    temp[n - 1] = 0;
  } // clear the last element

  for (long d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset = offset >> 1;
    __syncthreads();
    if (thid < d)
    {

      long ai = offset * (2 * thid + 1) - 1;
      long bi = offset * (2 * thid + 2) - 1;
      long t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();

  g_idata[thid] = temp[thid]; // write results to device memory
}

//long main(){
//    //long N = 10;
//    long tab_host[N] = {0,1,2,3,4,5,6,7};
//    long *tab_device;
//
//    const long size = N*sizeof(long);
//  
//    // alloc mem on GPU
//    cudaMalloc( (void**)&tab_device, size );
//
//    cudaMemcpy(tab_device, tab_host, size, cudaMemcpyHostToDevice);
//
//    //const long blocksize = 8; 
//
//    //dim3 dimBlock(blocksize , 1);
//    //dim3 dimGrid(2,1);
//
//    prescan<<<2,8,N * sizeof(long)>>>(tab_device, N);
//
//    cudaDeviceSynchronize();
//
//    cudaMemcpy(tab_host, tab_device, size, cudaMemcpyDeviceToHost);
//      
//    for (size_t i = 0; i < N; i++)
//    {
//        std::cout<<tab_host[i]<<" ";
//    }
//
//	std::cout<<std::endl;
//
//    cudaFree(tab_device);
//    
//}