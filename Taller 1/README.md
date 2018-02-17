Taller:
	1. Crear una función en C usando malloc que permita realizar la suma de 2 vectores de números en punto flotante.
	2. Crear una función en C usando malloc que permita realizar la multiplicación de dos matrices de números en punto flotante.

NOTA:
	- El tamaño de los vectores del punto 1 y de las matrices del punto 2 se definiran por línea de comandos.
	- Las matrices deberán inicializarse de manera aleatoria al igual que los vectores.
	- Deberán guardarse tanto las matrices de entrada y el resultado en un archivo csv. Lo mismo para los vectores.

PARA EJECUTAR EL PROGRAMA Y QUE SE ENTREGUEN EN .CSV

> gcc taller.c -o salida
LINUX
> ./salida < entrada.in > salida.out
WINDOWS
> salida < entrada.in > salida.out
---
El archivo de entrada esta así:
> <b>primer linea<b>: Tamaño de los vectores.
> <b>segunda linea<b>: Filas y columnas de primer matriz.
> <b>segunda linea<b>: Filas y columnas de segunda matriz.
