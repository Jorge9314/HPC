#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<time.h>
#include<cuda.h>

__global__
void matrizKernell(float* A, float* B, float* C, int Width){
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if((Row < rowsA) && (Col < colsB)){
        float sum = 0;
        for(int i=0; i < Width; ++i){
            sum += A[Row*Width+i] * B[k*Width+Col];
        }
        C[Row*Width+Col] = sum;
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

    float *A_in, *B_in, *C_out;
    int rowsA, colsA, rowsB, colsB;

    FILE *f1 *f2;
    f1 = fopen(argv[1], "r");
    f2 = fopen(argv[2], "r");

    fscanf(f1, "%d %d", &rowsA, &colsA);
    fscanf(f2, "%d %d", &rowsB, &colsB);
    
    //CPU
    A_in = (float*)malloc(rowsA * colsA * sizeof(float));
    B_in = (float*)malloc(rowsA * colsA * sizeof(float));
    C_out = (float*)malloc(rowsA * colsA * sizeof(float));
    
    receive(A_in, f1, rowsA, colsA);    
    receive(B_in, f2, rowsB, colsB);
    //print(A_in, rowsA, colsA);
    
    if(colsA != rowsB){
        print("Debe ser igual el numero de las columnas de A, a las filas de B");
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

    PictureKernell<<<dimGrid, dimBlock>>>(d_Ain, d_Bin, d_Cout, colsA);
    //cudaDeviceSynchronize();

    cudaMemcpy(C_out, d_Cout, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);
    print(C_out, rowsA, colsB);
    
    free(A_in); free(B_in); free(C_out);
    cudaFree(d_Ain); cudaFree(d_Bin); cudaFree(d_Cout);
    return 0;
}
