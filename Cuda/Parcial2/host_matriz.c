#include <stdio.h>
#include <stdlib.h>
#include "archivos_csv.c"
#include <time.h>


void mulM(float *d_a, float *d_b, float *d_c,int f1, int M, int c2){

	for(int i = 0; i < f1; i++){
		for(int j = 0; j < c2; j++){
			float Pvalue = 0.0;
			for(int k = 0; k < M; k++){
				Pvalue += d_a[i * M + k] * d_b[k * c2 + j];
			}
			d_c[i*c2+j] = Pvalue;
		}
	}


}

int main(int argc, char *argv[]){

	if(argc != 3){
		printf("no se han ingresado los archivos necesarios\n");
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

	clock_t time_inicial = clock();

	int Rows_1 = Detected_rows(fp,argv[1]);
	int Columns_1 = Detected_columns(fp,argv[1]);

	int Rows_2 = Detected_rows(fp,argv[2]);
	int Columns_2 = Detected_columns(fp,argv[2]);

	if(Columns_1 != Rows_2){
		printf("las matrizes no cumplen los requisitos para la multiplicacion entre estas");
		return 1;
	}

	int inteM = Columns_1;	
	
	float *Matriz1 = (float*)malloc(Rows_1 * inteM * sizeof(float));
	float *Matriz2 = (float*)malloc(inteM * Columns_2 * sizeof(float));
	float *MatrizR = (float*)malloc(Rows_1 * Columns_2 * sizeof(float));

	ExtracData(fp, Matriz1,argv[1],Rows_1,inteM);
	ExtracData(fp, Matriz2,argv[2],inteM,Columns_2);
	
	printf("Matriz 1: \n");
	printMatriz(Matriz1,Rows_1, inteM);
	printf("\n");
	printf("matriz 2: \n");
	printMatriz(Matriz2,inteM,Columns_2);
	printf("\n");

	
	mulM(Matriz1,Matriz2,MatrizR,Rows_1,inteM,Columns_2);
	clock_t time_final = clock();

	double ejecucion = (double)(time_final-time_inicial)/CLOCKS_PER_SEC;
	printf("tiempo de ejecucion en segundos: %.16g\n\n", ejecucion);

	printf("matriz Respuesta: \n");
	printMatriz(MatrizR, Rows_1, Columns_2);
	printf("\n");

	free(Matriz1);
	free(Matriz2);
	free(MatrizR);	

	return 0;

}
