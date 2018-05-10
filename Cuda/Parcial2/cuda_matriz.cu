#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<time.h>
#include<cuda.h>
#include "archivos_csv.c"

__global__
void matrizKernell(float* A, float* B, float* C, int rowsA, int colsA, int rowsB, int colsB){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if((Row < rowsA) && (Col < colsB)){
        float sum = 0;
        for(int i=0; i < colsB; i++){
            sum += A[Row*colsA+i] * B[i*colsB+Col];
        }
        C[Row*colsB+Col] = sum;
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
    if (argc != 3) {
        printf("Must be called with the names of the files \n");
        return 1;
    }
	
    FILE *fp;
    fp = fopen(argv[1], "r");

    if(fp==NULL){
	fputs("File error",stderr);
	return 1;
    }
    fclose(fp);
	
    fp = fopen(argv[2], "r");

    if(fp == NULL){
	fputs("file error", stderr);
	return 1;
    }
    fclose(fp);

    clock_t inicial = clock();

    float *A_in, *B_in, *C_out;
    int rowsA, colsA, rowsB, colsB;

    /*
    FILE *f1, *f2;
    f1 = fopen(argv[1], "r");
    f2 = fopen(argv[2], "r");

    fscanf(f1, "%d %d", &rowsA, &colsA);
    fscanf(f2, "%d %d", &rowsB, &colsB);
    */

    rowsA = Detected_rows(fp,argv[1]);
    colsA = Detected_columns(fp,argv[1]);

    rowsB = Detected_rows(fp,argv[2]);
    colsB = Detected_columns(fp,argv[2]);

    if(colsA != rowsB){
        printf("las matrizes no cumplen los requisitos para la multiplicacion entre estas");
        return 1;
    }

    //CPU
    A_in = (float*)malloc(rowsA * colsA * sizeof(float));
    B_in = (float*)malloc(rowsB * colsB * sizeof(float));
    C_out = (float*)malloc(rowsA * colsB * sizeof(float));
    
    //receive(A_in, f1, rowsA, colsA);    
    //receive(B_in, f2, rowsB, colsB);
    //print(A_in, rowsA, colsA);
    
    ExtracData(fp,A_in,argv[1],rowsA,colsA);
    ExtracData(fp,B_in,argv[2],rowsB,colsB);
    
    clock_t GPU_i = clock();

    //GPU
    cudaError_t error = cudaSuccess;
    float *d_Ain, *d_Bin, *d_Cout;
    int blockSize = 32;
    //int gridSize = ceil((colsA*rowsA) / float(blockSize));
    dim3 dimBlock(blockSize, blockSize, 1);
    dim3 dimGrid(ceil(colsA / float(blockSize)), ceil(rowsA / float(blockSize)), 1);

    error = cudaMalloc((void**)&d_Ain, rowsA * colsA * sizeof(float));
    if(error != cudaSuccess){
        printf("Error allocating memory d_Ain");
        return 1;
    }

    error = cudaMalloc((void**)&d_Bin, rowsB * colsB * sizeof(float));
    if(error != cudaSuccess){
        printf("Error allocating memory d_Bin");
        return 1;
    }

    error = cudaMalloc((void**)&d_Cout, rowsA * colsB * sizeof(float));
    if(error != cudaSuccess){
        printf("Error allocating memory d_Cout");
        return 1;
    }

    cudaMemcpy(d_Ain, A_in, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bin, B_in, rowsB * colsB * sizeof(float), cudaMemcpyHostToDevice);

    matrizKernell<<<dimGrid, dimBlock>>>(d_Ain, d_Bin, d_Cout, rowsA, colsA, rowsB, colsB);
    //cudaDeviceSynchronize();

    cudaMemcpy(C_out, d_Cout, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);
    print(C_out, rowsA, colsB);

    clock_t GPU_f = clock();

    clock_t final = clock();
    printf("tiempo de GPU: %.16g segundos\n", (double)(GPU_f-GPU_i/CLOCKS_PER_SEC));
    printf("tiempo de CPU: %.16g segundos\n", (double)(final-inicial/CLOCKS_PER_SEC));
    
    free(A_in); free(B_in); free(C_out);
    cudaFree(d_Ain); cudaFree(d_Bin); cudaFree(d_Cout);
    return 0;
}
