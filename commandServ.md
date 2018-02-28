#SLURM Example

+ sinfo : Muestra el estaddo de los nodos 
+ squeue : Muestra las tareas encoladas en los ndoos
+ srun -N<numeroNodos> <funtion> : srun -N6 hostname
    ** <numeroNodos> : no puede ser mayor al numero de nodos disponibles
    ** <funcion> : 
        **hostname** manda una programa que retorna el nombre del host
        **bash** crea una consola que se ejecuta en diferentes nodos.
+ srun --pty --mem 500 -t 0-1:00 /bin/bash 