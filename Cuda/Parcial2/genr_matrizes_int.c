#include<stdio.h>
#include<stdlib.h>

void EscribirArchivo(float **Matriz,char nombre[], int N, int M){

	FILE *fp;
	fp = fopen (nombre,"w");

	for(int i = 0; i < N; i++){
		for(int j = 0; j < M; j++){
			char array[50];
			//sprintf(array, "%f", Matriz[i][j]);
			sprintf(array, "%d", Matriz[i][j]);
			fputs(array, fp);
			fputs(";", fp);
		}
		fputs("\n", fp);
	}

	fclose(fp);


}

int main(int argc, char *argv[]){

	if(argc != 5){
		printf("el numero de datos de entrada no concuerda con lo pedido");
		return 1;
	}

	int N = atoi(argv[2]);
	int M = atoi(argv[3]);
	int Num = atoi(argv[4]);

	float  **matriz;

	matriz = (float**)malloc(N * sizeof(float*));

	if(matriz == NULL){
		printf("Error reservando memoria!");
		return 1;
	}

	for(int i = 0; i < N; i++){
		matriz[i] = (float*)malloc(M*sizeof(float*));
		if(matriz[i] == NULL){
			printf("Error de reserva de memoria");
			return 1;
		}
	}

	printf("llenando matrizes... \n");

	float a = Num / 1.0;

	for(int i = 0; i < N; i++){
		for(int j = 0; j < M; j++){
			//float num = ((float)rand()/(float)(RAND_MAX)*a);
			int num = rand() % Num;
			matriz[i][j] = num;
		}
	}

	EscribirArchivo(matriz, argv[1], N, M);

	printf("la matriz ha sido llenada con exito en el archivo...\n");

	return 0;
}
