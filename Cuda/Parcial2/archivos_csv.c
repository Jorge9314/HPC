#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 10000

int Detected_rows(FILE *fp, char nombre[]){

	fp = fopen(nombre,"r");

	int i = 0;
	char c[N];
	while(fscanf(fp,"%s",c) != EOF){
		i++;
	}
	fclose(fp);
	return i;
}

int Detected_columns(FILE *fp, char nombre[100]){

	fp = fopen(nombre, "r");
	char cadena[N];
	fscanf(fp,"%s",cadena);

	char *tok = strtok(cadena, ";");
	int i = 0;
	do{
		i++;
		tok = strtok(NULL, ";");
	}while(tok != NULL);
	fclose(fp);

	return i;
}

void ExtracData(FILE *fp,float *M ,char nombre[], int Rows, int Columns){

	fp = fopen(nombre,"r");
	char cadena[N];
	char *tok;

	for(int i = 0; i < Rows; i++){
		fscanf(fp,"%s",cadena);
		tok = strtok(cadena, ";");
		for(int j = 0; j < Columns; j++){
			M[i * Columns + j] = strtod(tok,NULL);
			tok = strtok(NULL, ";");
		}
	}
	fclose(fp);

}

void ExtracData(FILE *fp,int *M ,char nombre[], int Rows, int Columns){

        fp = fopen(nombre,"r");
        char cadena[N];
        char *tok;

        for(int i = 0; i < Rows; i++){
                fscanf(fp,"%s",cadena);
                tok = strtok(cadena, ";");
                for(int j = 0; j < Columns; j++){
                        M[i * Columns + j] = atoi(tok);
                        tok = strtok(NULL, ";");
                }
        }
        fclose(fp);

}

void printMatriz(float *Matriz, int f, int c){
	for(int i = 0; i < f; i++){
		for(int j = 0; j < c; j++){
			printf("[%f]", Matriz[i * c + j]);
		}
		printf("\n");
	}
}


/*
int main(int argc, char *argv[]){

	char nombre[N] = "entrada1omp.csv";
	
	FILE *fp;
	fp = fopen(nombre, "r");
	
	if(fp == NULL){
		printf("archivo no encontrado");
		return 1;
	}
	fclose(fp);

	int rows_1 = Detected_rows(fp,nombre);
	printf("El numero de filas es: %d\n", rows_1);

	int columns_1 = Detected_columns(fp,nombre);
	printf("El numero de columnas es: %d\n", columns_1);
	
	metodo de transformacion string a float
		
	float a;
	
	a = strtod(tok , NULL);
	printf("%f", a);
	
	
	return 0;
	
}
*/
