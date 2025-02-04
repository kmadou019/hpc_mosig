#include <cuda_runtime.h>
#include <iostream>

#define N 16
#define THREADS_PER_BLOCK 8

#define add(a , b) (a + b)

__global__ void prescan(float *d_out, float *d_in, int n) {
    

    extern __shared__ float temp[];  // Allocation dynamique de la mémoire partagée
    int tid = threadIdx.x;

    // Chargement des données dans la mémoire partagée
    temp[2 * tid] = (2 * tid < n) ? d_in[2 * tid] : 0.0f;
    temp[2 * tid + 1] = (2 * tid + 1 < n) ? d_in[2 * tid + 1] : 0.0f;
    __syncthreads();

    // Phase d'upsweep (réduction)
    for (int stride = 1; stride <= n / 2; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            temp[index] = add(temp[index] , temp[index - stride]);
        }
        __syncthreads();
    }

    // Mise à zéro du dernier élément
    if (tid == 0) {
        temp[n - 1] = INFINITY;
    }
    __syncthreads();

    // Phase de downsweep (propagation)
    for (int stride = n / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < n) {
            float t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] = add(temp[index], t);
        }
        __syncthreads();
    }

    // Stockage du résultat dans la mémoire globale
    if (2 * tid < n) d_out[2 * tid] = temp[2 * tid];
    if (2 * tid + 1 < n) d_out[2 * tid + 1] = temp[2 * tid + 1];
}

int main() {
    float h_in[N] = {8, 3, 1, 7, 14, 4, 6, 3, 9, 2, 5, 8, 1, 7, 4, 4};
    float h_out[N];

    float *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(float));
    cudaMalloc((void**)&d_out, N * sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    prescan<<<2, THREADS_PER_BLOCK, N * sizeof(float)>>>(d_out, d_in, N);

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Prescan Output: ";
    for (int i = 0; i < N; i++) std::cout << h_out[i] << " ";
    std::cout << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
