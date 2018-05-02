#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<time.h>
#include<cuda.h>

__global__
void matrixMultKernel(float* d_M, float* d_N, float* d_P, int width){
    const int TILE_WIDTH = 2;
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for(int i = 0; i < width/TILE_WIDTH; ++i){
      Mds[ty][tx] = d_M[Row*width + i*TILE_WIDTH + tx];
      Nds[ty][tx] = d_N[(i*TILE_WIDTH + ty)*width + Col];
      __syncthreads();

      for(int k=0; k<TILE_WIDTH; ++k){
        Pvalue += Mds[ty][k] + Nds[k][tx];
      }
      __syncthreads();
    }
    d_P[Row*width + Col] = Pvalue;
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

    float *A_in, *B_in, *C_out;
    int rowsA, colsA, rowsB, colsB;

    FILE *f1, *f2;
    f1 = fopen(argv[1], "r");
    f2 = fopen(argv[2], "r");

    fscanf(f1, "%d %d", &rowsA, &colsA);
    fscanf(f2, "%d %d", &rowsB, &colsB);

    //CPU
    A_in = (float*)malloc(rowsA * colsA * sizeof(float));
    B_in = (float*)malloc(rowsB * colsB * sizeof(float));
    C_out = (float*)malloc(rowsA * colsB * sizeof(float));

    receive(A_in, f1, rowsA, colsA);
    receive(B_in, f2, rowsB, colsB);
    //print(A_in, rowsA, colsA);

    if(colsA != rowsB){
        printf("Debe ser igual el numero de las columnas de A, a las filas de B");
        return 1;
    }

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

    matrixMultKernel<<<dimGrid, dimBlock>>>(d_Ain, d_Bin, d_Cout, rowsA);
    //cudaDeviceSynchronize();

    cudaMemcpy(C_out, d_Cout, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);
    print(C_out, rowsA, colsB);

    free(A_in); free(B_in); free(C_out);
    cudaFree(d_Ain); cudaFree(d_Bin); cudaFree(d_Cout);
    return 0;
}
