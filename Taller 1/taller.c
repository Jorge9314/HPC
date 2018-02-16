#include <stdio.h>
#include <stdlib.h>

float* func_vect(int t);
float** func_mat(int f, int c);
void sum(int sz);
void mult();


int main() {
	int sz, fila1, columna1, fila2, columna2;
	/*
	printf("Ingrese el numero de elementos del vector");
	scanf("%d", &sz);
	sum(sz);
	*/
	printf("Ingrese las filas y luego las columnas de la primer matriz");
	scanf("%d %d", &fila1, &columna1);
	printf("Ingrese las filas y luego las columnas de la segunda matriz");
	scanf("%d %d", &fila2, &columna2);
	mult(fila1, columna1, fila2, columna2);
	return 0;
}

float * func_vect(int t)
{
	int i, x = t;
	float* ptr;
	ptr = (float *)malloc(x * sizeof(float));
	if(ptr == NULL){
		puts("ALLOCATION FAILED \n");
	}else{
		// puts("CORRECT \n");
		for(i=0; i<x; i++){
			ptr[i] = rand() % 10;
		}
		return ptr; 
	}
}

float ** func_mat(int f, int c){
	int i,j;
	float** matriz;
	matriz = (float **)malloc(f * sizeof(float *));
	for(i = 0; i < f; ++i){
		matriz[i] = (float *)malloc(c * sizeof(float));
	}
	for(i = 0; i < f; ++i){
		for(j = 0; j < c; ++j){
			matriz[i][j] = rand() % 10;
		}
	}
	return matriz;
} 

void sum(int sz)
{
	int i;
	float* x;
	float* y;
	x = func_vect(sz);
	y = func_vect(sz);
	float z[sz];
	for(i=0; i<sz; i++){
		z[i] = x[i] + y[i];
		printf("%f \n", z[i]);
	}
}

void print_mat(int r, int c, float** x){
	int i, j;
	for(i=0; i<r; i++){
		for(j=0; j<c; j++){
			printf(" %f ", x[i][j]);
		}
		printf("\n \n");
	}
}

void mult(int f1, int c1, int f2, int c2){
	if(c1 != f2){
		puts("ERROR NO SE PUEDEN MULTIPLICAR");
		exit(1);
	}
	int i, j, k;
	float** x;
	float** y;
	float** res;
	x = func_mat(f1, c1);
	y = func_mat(f2, c2);
	res = func_mat(f1, c2);
	for(i=0; i<f1; i++){
		for(j=0; j<c2; j++){
			res[i][j] = 0;
		}
	}
	for(i=0; i<f1; i++){
		for(j=0; j<c2; j++){
			for(k=0; k<c1; ++k){
				res[j][i] += x[k][i] * y[j][k]; 
			}
		}
	}
	print_mat(f2, c2, y);
	print_mat(f1, c1, x);
	print_mat(f1, f1, res);
}