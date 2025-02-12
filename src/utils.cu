#include <cuda_runtime.h>
#include <iostream>
#include "utils.h"

#define THREADS_PER_BLOCK 8

__global__ void prescan(long *d_out, long *d_in, long n) {
    extern __shared__ long temp[];  // Allocation dynamique de la mémoire partagée
    int tid = threadIdx.x;

    // Chargement des données dans la mémoire partagée
    temp[2 * tid] = (2 * tid < n) ? d_in[2 * tid] : 0;
    temp[2 * tid + 1] = (2 * tid + 1 < n) ? d_in[2 * tid + 1] : 0;
    __syncthreads();

    // Phase d'upsweep (réduction)
    for (long stride = 1; stride <= n / 2; stride *= 2) {
        long index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            temp[index] = (temp[index] + temp[index - stride]);
        }
        __syncthreads();
    }

    // Mise à zéro du dernier élément
    if (tid == 0) {
        temp[n - 1] = 0;
    }
    __syncthreads();

    // Phase de downsweep (propagation)
    for (long stride = n / 2; stride > 0; stride /= 2) {
        long index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            long t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] = (temp[index]+t);
        }
        __syncthreads();
    }

    // Stockage du résultat dans la mémoire globale
    if (2 * tid + 1 < n) d_out[2 * tid] = temp[2 * tid + 1];
    if (2 * tid + 1 + 1< n) d_out[2 * tid + 1] = temp[2 * tid + 1 + 1];
    //if (2 * tid < n) d_out[2 * tid] = temp[2 * tid ];
    //if (2 * tid + 1< n) d_out[2 * tid + 1] = temp[2 * tid + 1];
}

int main() {
    long N = 15;
    long *h_in  = (long*)malloc((N+1) * sizeof(long));
    long *h_out = (long*)malloc((N+1) * sizeof(long));

    long values[N] = {8, 3, 1, 7, 14, 4, 6, 3, 9, 2, 8, 1, 7, 4, 30};
    for (int i = 0; i < N; i++) {
        h_in[i] = values[i];
    }

    std::cout << "Input: ";
    for (long i = 0; i < N; i++) std::cout << h_in[i] << " ";
    std::cout << std::endl;

    long *d_in, *d_out;
    cudaMalloc((void**)&d_in, (N+1)* sizeof(long));
    cudaMalloc((void**)&d_out, (N+1) * sizeof(long));

    h_in[N] = h_in[N-1];
    cudaMemcpy(d_in, h_in, (N) * sizeof(long), cudaMemcpyHostToDevice);

    prescan<<<2, THREADS_PER_BLOCK, (N+1) * sizeof(long)>>>(d_out, d_in, N+1);

    cudaMemcpy(h_out, d_out, (N) * sizeof(long), cudaMemcpyDeviceToHost);

    std::cout << "Prescan Output: ";
    for (long i = 0; i < N; i++) std::cout <<h_out[i] << " ";
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
