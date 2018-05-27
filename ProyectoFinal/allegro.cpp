#include <stdio.h> 
#include <stdlib.h>
#include <allegro5/allegro.h> // Librería inicial de Allegro 
#include <allegro5/allegro_primitives.h> // Addon de primitivas (figuras)
#include <unistd.h>
#include <iostream>

using namespace std;

#define VENTANA_X 800 
#define VENTANA_Y 600
#define NEGRO al_map_rgb(0,0,0)
#define BLANCO al_map_rgb(255,255,255)
#define AZUL al_map_rgb(0,0,255) 
#define VERDE al_map_rgb(0,255,0)
#define ROJO al_map_rgb(255,0,0)

// Redefinición de tipos 
typedef ALLEGRO_DISPLAY aDisplay; 

// Variables globales 
aDisplay *dis = NULL;

struct Point
{
    int x, y;
};

// Funciones 
void inicializar(void){ 
    // Inicializar los addons 
    if(!al_init()) 
    { 
        fprintf(stderr, "Error al inicializar Allegro."); // Imprimir errores en stream STDERR
        exit(-1); 
    } 
    if(!al_init_primitives_addon()) 
    { 
        fprintf(stderr, "Error al inicializar el addon de primitivas."); // Imprimir errores en stream STDERR
        exit(-2); 
    } 

    dis = al_create_display(VENTANA_X, VENTANA_Y); // Crear el display de tamaño 500x300 píxeles

    al_set_window_title(dis, "Tabla Grafica"); // Establecer el título de la ventana
}

void finalizar(){ 
    al_shutdown_primitives_addon(); // Finalizar el addon de primitivas
    al_destroy_display(dis); // Destruir la ventana 
}

void colorearPantalla(){ 
    al_clear_to_color(BLANCO); // Limpiar la ventana y establecer un color de fondo RGB (color rojo: R=0=0.0, G=255=1.0, B=0=0.0)
}

void Puntos(Point v[], int Columnas, int tope){
    for(int i = 0; i < tope; i++){
        al_draw_filled_circle(v[i].x+10, VENTANA_Y-v[i].y-10, 2.0, ROJO); // Crear un círculo: x = 200px, y = 160px, radio = 130px, color
        //printf("filas: %d, columnas: %d\n", v[i]+10, VENTANA_Y-v[i+Columnas]-10);
    }
}

void Lines(Point v[], int Columnas, int tope){

    al_draw_line(v[Columnas-1].x+10,VENTANA_Y-v[Columnas-1].y-10, v[0].x+10,VENTANA_Y-v[0].y-10, NEGRO, 1.0);

    for(int i = 0; i < tope; i++){
        al_draw_line(v[i].x+10, VENTANA_Y-v[i].y-10, v[i+1].x+10, VENTANA_Y-v[i+1].y-10, NEGRO, 1.0);
    }
}

void tabla(){
    // linea y
    float x1 = 10.0;
    float x2 = 10.0;
    float y1 = 0;
    float y2 = VENTANA_Y;
    al_draw_line(x1,y1,x2,y2,NEGRO,2);
    // linea x
    float X1 = 0;
    float Y1 = VENTANA_Y - 10.0;
    float X2 = VENTANA_X;
    float Y2 = Y1;
    al_draw_line(X1,Y1,X2,Y2,NEGRO,2);

    //subdivisiones de la grafica
    int division = 10;
    int Tam_Y = VENTANA_Y - 20;
    int Tam_X = VENTANA_X - 20;

    int incre = Tam_Y / division;

    for(int i = 10; i <= VENTANA_Y; i += incre){
        al_draw_line(x1-3, i,x2+3, i,NEGRO,1);
    }

    incre = Tam_X / division;

    for(int i = 10; i <= VENTANA_X; i += incre){
        al_draw_line(i ,Y1-3, i, Y1+3,NEGRO,1);
    }

}

void Graficar(Point vp[], int cp, Point vl[], int cl){
    inicializar();

    int i = 1;
    while(i <= cp){
        colorearPantalla();
        tabla();
        Puntos(vp,cp,i);
        //Lines(vl,cl);
        al_flip_display(); // Dibujar en pantalla todo lo almacenado en el buffer
        i++;
    }

    printf("finish points\n");
    i = 0;

    while(i < cl){
        colorearPantalla();
        tabla();
        Puntos(vp,cp,cp);
        Lines(vl,cl,i);
        al_flip_display(); // Dibujar en pantalla todo lo almacenado en el buffer
        sleep(1); // retardo de 2 segundos
        i++;
    }

    printf("finsh lines...\n");
    sleep(4);
    finalizar(); 
} 

int main(int argc, char *argv[]){

    int colP;
    cin >> colP;
    Point vectorP[colP];

    for(int i = 0; i < colP; i++){
        cin >> vectorP[i].x;
        cin >> vectorP[i].y;
    }


    int colL;
    cin >> colL;
    Point vectorL[colL];

    for(int i = 0; i < colL; i++){
        cin >> vectorL[i].x;
        cin >> vectorL[i].y;
    }

    Graficar(vectorP, colP, vectorL, colL);

    return 0; 
} 