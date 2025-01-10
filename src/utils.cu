#include <cmath>

__global__ void coreReduce(int tab[], int len, int d){
    int idx = threadIdx.x;
    if(idx < len)
        tab[idx + pow(2, d+1)] = tab[idx + pow(2, d) - 1] + tab[idx + pow(2, d) + 1 - 1] 

}

void reduce(int tab[], int len){

    for (size_t d = 0; d < log(len) ; d++)
    {
        coreReduce(tab, len,d)
    }
}

void main(){
    
}