#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<time.h>
#include<cuda.h>

__global__
void PictureKernell(float* d_Pin, float* d_Pout, int n, int m){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < m) && (Col < n)){
        d_Pout[Row*n+Col] = 2*d_Pin[Row*n+Col]; 
    }
}

__host__
void print(float* M, int rows, int cols){
    printf("-----------MATRIX ------------- \n");
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            printf("%f ", M[i * cols + j]);
        }
        printf("\n");
    }
}

__host__
void receive(float* M, FILE* stream, int rows, int cols){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++){
            fscanf(stream, "%f", &M[i * cols +j]);
        }
    }
    fclose(stream);
}

int main(int argc, char** argv){
    if (argc != 2) {
        printf("Must be called with the names of the files \n");
        return 1;
    }

    float *A_in, *A_out;
    int rowsA, colsA;

    FILE *f1;
    f1 = fopen(argv[1], "r");

    fscanf(f1, "%d", &rowsA);
    fscanf(f1, "%d", &colsA);
    
    
    //CPU
    A_in = (float*)malloc(rowsA * colsA * sizeof(float));
    A_out = (float*)malloc(rowsA * colsA * sizeof(float));

    receive(A_in, f1, rowsA, colsA);    
    //print(A_in, rowsA, colsA);
    
    //GPU
    cudaError_t error = cudaSuccess;
    float *d_Ain, *d_Aout;
    int blockSize = 32;
    //int gridSize = ceil((colsA*rowsA) / float(blockSize));
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(colsA / float(blockSize)), ceil(rowsA / float(blockSize)), 1);

    error = cudaMalloc((void**)&d_Ain, rowsA * colsA * sizeof(float));
    if(error != cudaSuccess){
        printf("Error allocating memory d_Ain");
        return 1;
    }

    error = cudaMalloc((void**)&d_Aout, rowsA * colsA * sizeof(float));
    if(error != cudaSuccess){
        printf("Error allocating memory d_Aout");
        return 1;
    }

    cudaMemcpy(d_Ain, A_in, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_Aout, A_out, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);

    PictureKernell<<<dimGrid, dimBlock>>>(d_Ain, d_Aout, rowsA, colsA);
    //cudaDeviceSynchronize();

    cudaMemcpy(A_out, d_Aout, rowsA * colsA * sizeof(float), cudaMemcpyDeviceToHost);
    print(A_out, rowsA, colsA);
    
    free(A_in);
    free(A_out);
    cudaFree(d_Ain);
    cudaFree(d_Aout);
    return 0;
}