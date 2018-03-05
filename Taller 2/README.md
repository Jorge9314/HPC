# Taller 2

El programa [Generador](generate.cpp) recibe las filas y las columnas de las matrices.

> g++ generate.cpp -o generador

> ./generador < entrada.in > matrices.in

El programa sin paralelizar [TallerO](tallerO.c) hace la multiplicación de las matrices.

> gcc tallerO.c -o taller

> time ./taller < matrices.in > salida1.out

El programa paralelizado con openMP [TallerF](tallerF.c) hace la multiplicacion de las matrices.

> gcc tallerF.c -o tallerP

> time ./tallerP < matrices.in > salida2.out

Para verificar que se esta realizando en ambos programas la multiplicación de manera correcta.

> diff -s salida1.out salida2.out 
