#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<time.h>
#include<cuda.h>

__global__
void sum(float* A, float* B, float* C, int size){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id < size){
        C[id] = A[id] + B[id];
    }
}

__host__
void print(float *M, int size){
    printf("-----------Vector------------\n");
    for(int i=0; i<size; i++){
        printf("%f", M[i]);
        printf("\n");
    }
}

__host__
void receive(float *M, FILE *stream, int size){
    for(int i=0; i<size; i++){
        fscanf(stream, "%f", &M[i]);
    }
    fclose(stream);
}

int main(int argc, char** argv){
    if(argc != 3){
        printf("Must be called with the names of the files \n");
        return 1;
    }

    int sizeA, sizeB;

    cudaError_t error = cudaSuccess;
    float *h_A, *h_B, *h_C;
    FILE *f1, *f2;
    f1 = fopen(argv[1], "r");
    f2 = fopen(argv[2], "r");

    fscanf(f1, "%d", &sizeA);
    fscanf(f2, "%d", &sizeB);

    if(sizeA != sizeB){
        printf("The vectors should have same dimensions \b");
        return 1;
    }

    //CPU
    h_A = (float*)malloc(sizeA*sizeof(float));
    h_B = (float*)malloc(sizeA*sizeof(float));
    h_C = (float*)malloc(sizeA*sizeof(float));

    receive(h_A, f1, sizeA);
    receive(h_B, f2, sizeA);
    //print(h_A, sizeA);
    //print(h_B, sizeB);
    
    //GPU
    float *d_A, *d_B, *d_C;
    int blockSize = 32;
    int gridSize = ceil(sizeA / float(blockSize));

    //dim3 dimBlock(blockSize,1,1);
    //dim3 dimGrid(ceil(sizeA / float(blockSize)),1,1);

    error = cudaMalloc((void**)&d_A, sizeA*sizeof(float));
    if (error != cudaSuccess){
        printf("Error allocating memory d_A");
        return 1;
    }

    error = cudaMalloc((void**)&d_B, sizeA*sizeof(float));
    if (error != cudaSuccess){
        printf("Error allocating memory d_B");
        return 1;
    }

    error = cudaMalloc((void**)&d_C, sizeA*sizeof(float));
    if (error != cudaSuccess){
        printf("Error allocating memory d_C");
        return 1;
    }

    cudaMemcpy(d_A, h_A, sizeA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeA*sizeof(float), cudaMemcpyHostToDevice);

    sum<<<gridSize, blockSize>>>(d_A, d_B, d_C, sizeA);
    //cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, sizeA*sizeof(float), cudaMemcpyDeviceToHost);
    print(h_C, sizeA);

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}