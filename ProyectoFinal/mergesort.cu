#include <stdio.h>
#include <math.h>
#include <cuda.h>

__host__ void checkCudaState(cudaError_t& cudaState,const char *message){
  /* it will print an error message if there is */
  if(cudaState != cudaSuccess) printf("%s",message);
}

__device__ void swap(int *points,uint lowIndex,uint upIndex){
  /* it will swap two points */
  int aux = points[lowIndex];
  points[lowIndex] = points[upIndex];
  points[upIndex] = aux;
}

__global__ void sort(int *points,uint phase,uint n){
  /* it will sort with points array with respect to phase*/
  uint ti = blockIdx.x*blockDim.x+threadIdx.x;
  if(ti >= n || ti == 0) return;

  if(ti%phase == 0){ // multiplier phase
    uint top = ti, lower = (top - phase) + 1;
    uint middle = lower + phase/2;
    uint lowG1 = lower, lowG2 = middle, topG1 = middle-1, topG2 = top;
    while(true){
      if(lowG1 > topG1 && lowG2 > topG2) break;

      // --------------------- case 1 ---------------------
      if(lowG1 <= topG1 && lowG2 <= topG2){

        if(points[lowG1] > points[lowG2]){
          swap(points,lowG1,lowG2);
          lowG2++;
        }
        else lowG1++;

      }

      // --------------------- case 2 ---------------------
      else if(lowG1 < topG1 && lowG2 > topG2){
        uint next = lowG1 + 1;
        if(points[lowG1] > points[next])
          swap(points,lowG1,next);
        lowG1++;
      }

      // --------------------- case 3 ---------------------
      else if(lowG2 < topG2 && lowG1 > topG1){
        uint next = lowG2 + 1;
        if(points[lowG2] > points[next])
          swap(points,lowG2,next);
        lowG2++;
      }

      else if(lowG1 == topG1)
        lowG1++;
      else if(lowG2 == topG2)
        lowG2++;
    }
  }
}

__host__ void fill(int *points,size_t n){
  /* it will fill points array */
  for(size_t i=0; i<n; i++)
    points[i] = n-i;
}

__host__ void show(int* points,size_t n){
  /* it will show points array */
  for(size_t i=0; i<n; i++)
    printf("%d ",points[i]);
  printf("\n\n");
}

int main(int argc, char const *argv[]) {
  size_t items = 2049;
  size_t size = items*sizeof(int);
  cudaError_t cudaState = cudaSuccess;
  int *h_points = NULL, *d_points = NULL, *h_result = NULL;

  h_points = (int*)malloc(size);
  h_result = (int*)malloc(size);
  fill(h_points,items);
  cudaState = cudaMalloc((void**)&d_points,size);
  checkCudaState(cudaState,"Impossible allocate data\n");
  if(d_points != NULL){
    cudaState = cudaMemcpy(d_points,h_points,size,cudaMemcpyHostToDevice);
    checkCudaState(cudaState,"Impossible copy data from host to device\n");
    show(h_points,items);

    dim3 blockSize(1024,1,1);
    dim3 gridSize((int)(ceil(items/1024.0)),1,1);
    uint i = 1;
    while(pow(2,i) <= items){
      sort<<<gridSize,blockSize>>>(d_points,pow(2,i),items);
      cudaDeviceSynchronize();
      i++;
    }
    cudaState = cudaMemcpy(h_result,d_points,size,cudaMemcpyDeviceToHost);
    checkCudaState(cudaState,"Impossible copy data from device to host\n");
    show(h_result,items);
  }

  if(h_points != NULL) free(h_points);
  if(h_result != NULL) free(h_result);
  if(d_points != NULL) cudaFree(d_points);
  return 0;
}
