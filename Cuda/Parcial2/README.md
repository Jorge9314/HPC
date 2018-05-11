# Proyecto de multiplicación de matrices

Este proyecto de multiplicación de matrices fue realizado en lenguaje C y C++, con el objetivo de entender y aplicar CUDA para la materia de High Performance Computing (HPC)

## Multiplicación de matrices en (Lenguaje C)

Esta es una implementación básica de la multiplicación de matrices, con complejidad O(n³). [MultMatrices](https://github.com/Jorge9314/HPC/blob/master/Cuda/Parcial2/host_matriz.c) Recive como entradas las matrices.

Para compilar el código:
> gcc host_matriz.c -o test

> time ./test m1 m2 > salida.out 

## Multiplicación de matrices con CUDA (Lenguaje C)

Esta es una implementación ingenua de la multiplicación de matrices, utilizando los hilos de una GPU GeForce gtx980 con CUDA. [MultCuda](https://github.com/Jorge9314/HPC/blob/master/Cuda/Parcial2/cuda_matriz.cu)
Para compilar el código:
> nvcc cuda_matriz.cu -o test

> time ./test m1 m2 > salida1.out

## Multiplicación de matrices Memoria Compartida (Lenguaje C++)

Esta es una implementación utilizando la memoria compartida de la GeForce gtx980, para agilizar el acceso a los datos de mayor frecuencia. Permitiendo que el tiempo de ejecución sea mucho menor. [MultShared](https://github.com/Jorge9314/HPC/blob/master/Cuda/Parcial2/matrizSh.cu)
Adicionalmente se utiliza SLURM para la ejecución del codigo en el cluster. [Slurm](https://github.com/Jorge9314/HPC/blob/master/Cuda/Parcial2/batch.sh)

Para compilar el código en el cluster:
> bash ./batch.sh

Para ejecutar el código sin Slurm:
> nvcc matrizSh.cu -o test

> time ./test m1 m2 > salida2.out
