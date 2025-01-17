#include <iostream>

__global__ void prescan(int *g_idata, int n)
{
  extern __shared__ int temp[];
  // allocated on invocation
  int thid = threadIdx.x;
  int offset = 1;

  temp[thid] = g_idata[thid]; // load input into shared memory

  for (int d = n >> 1; d > 0; d = d >> 1)
  // build sum in place up the tree
  {
    __syncthreads();
    if (thid < d)
    {

      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }
  if (thid == 0)
  {
    temp[n - 1] = 0;
  } // clear the last element

  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset = offset >> 1;
    __syncthreads();
    if (thid < d)
    {

      int ai = offset * (2 * thid + 1) - 1;
      int bi = offset * (2 * thid + 2) - 1;
      float t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();



  g_idata[thid] = temp[thid]; // write results to device memory
}

int main(){
    int N = 10;
    int tab_host[N] = {0,1,2,3,4,5,6,7,8,9};
    int *tab_device;

    const int size = N*sizeof(int);

    // alloc mem on GPU
    cudaMalloc( (void**)&tab_device, size );

	cudaMemcpy(tab_device, tab_host, size, cudaMemcpyHostToDevice);

    const int blocksize = 8; 

    dim3 dimBlock(blocksize , 1);
    dim3 dimGrid(2,1);

    prescan<<<dimGrid,dimBlock>>>(tab_device, 10);

	cudaDeviceSynchronize();

	cudaMemcpy(tab_host, tab_device, size, cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < 10; i++)
    {
        std::cout<<tab_host[i]<<" ";
    }

	std::cout<<std::endl;

    
}