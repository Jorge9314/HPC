## MPI

+ Se necesita combinar CUDA + OpenMP + MPI, o sus contrapartes
+ Antes del 2009 no habia supercomputadores que usaran GPU's
+ La interfaz de programación dominante para cluster hoy en dia es MPI
+ MPI permite comunicación entre procesos corriendo en diferentes nodos
+ MPI asume un modelo de memoria distribuida
+ Nodo "maestro"
    - int MPI_Init (int*argc, char***argv)
        inicializar   
+  ***Copiar y ejecutar el MPI basics (3/3)***
+  MPI_Sent(...), MPI_Recv(...) ***MPI Point to Point Communication (1/4) ***
+ Tutorial para practicar ***Bibliography (1/1)***
+ Repositorio @kala855, ejemplos de Slurm