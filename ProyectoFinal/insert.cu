#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

__global__ void insert(int *a, int t){

	int i = blockIdx.x*blockDim.x+threadIdx.x;

	if(i < t-1){
		for(int k = 0; k < t-1; k++){
			if(a[k] > a[k+1]){
				int aux = a[k];
				a[k] = a[k+1];
				a[k+1] = aux;
			}
		}
	}

}

void cuda(int *a, int n){

	int *array;

	array = (int*)malloc(n * sizeof(int));
	cudaMemcpy(array,a,n * sizeof(int),cudaMemcpyHostToDevice);

	insert<<<1024,1>>>(array,n);

	cudaMemcpy(a,array,n * sizeof(int),cudaMemcpyDeviceToHost);

	free(array);

}

int main(){

	int n;
	cin >> n;

	int a[n];

	for(int i = 0; i < n; i++){
		cin >> a[i];
	}

	cuda(a,n);

	for(int i = 0; i < n; i++){
		cout << a[i] << " ";
	}
	cout<<endl<<endl;

	return 0;
}
