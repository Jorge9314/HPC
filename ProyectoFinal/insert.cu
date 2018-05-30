#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

__global__ void insert(int *a, int t){

	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if(i < t-1){
		if(a[i] > a[i+1]){
			int aux = a[i];
			a[i] = a[i+1];
			a[i+1] = aux;
		}
	}

}

void cuda(int *a, int n){

	int *array;
	int blockSize = 32;

	array = (int*)malloc(n * sizeof(int));
	cudaMemcpy(array,a,n*sizeof(int),cudaMemcpyHostToDevice);

    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(ceil(n / float(blockSize)), 1, 1);
    insert<<<dimGrid,dimBlock>>>(array,n);

    cudaMemcpy(a,array,n*sizeof(int),cudaMemcpyDeviceToHost);

}

int main(){

	int n;
	cin >> n;

	int a[n];

	for(int i = 0; i < n; i++){
		cin >> a[i];
	}

	for(int i = 0; i < n; i++){
		printf("%d\n", a[i]);
	}
	printf("\n\n", );

	cuda(a,n);

	for(int i = 0; i < n; i++){
		printf("%d\n", a[i]);
	}
	printf("\n\n");

	return 0;
}