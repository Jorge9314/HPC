import pygame
from libpolares import *
import math

ALTO=600
ANCHO=1000

def rotar(A,B,C):
  return (B[0]-A[0])*(C[1]-B[1])-(B[1]-A[1])*(C[0]-B[0])

def jarvisborde(A):
    n = len(A)
    P = range(n)
    #start point
    for i in range(1,n):
        if A[P[i]][0]<A[P[0]][0]:
            P[i], P[0] = P[0], P[i]
    H = [P[0]]
    del P[0]
    P.append(H[0])
    while True:
        right = 0
        for i in range(1,len(P)):
            if rotar(A[H[-1]],A[P[right]],A[P[i]])<0:
                right = i
        if P[right]==H[0]:
            break
        else:
            H.append(P[right])
            del P[right]
    return H


if __name__ == '__main__':
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pantalla.fill(BLANCO)

    #ejes(xc,yc)
    n_puntos= [[359,0], [320,0], [550,80], [326,66], [393,41], [413,96], [197,04], [259,82], [322,17], [344,62], [417,98], [528,73],
    [556,49], [285,79]]
    centro=[10,490]
    pl=Cartesiano(centro, ANCHO, ALTO,pantalla)
    for p in n_puntos:
        pl.Punto(p)
    ls_ind=jarvisborde(n_puntos)
    print ls_ind
    ls_env=[]
    for i in ls_ind:
        print n_puntos[i]
        vp=pl.Cart(n_puntos[i])
        ls_env.append(vp)
    pygame.draw.polygon(pantalla,ROJO,ls_env,1)

    pygame.display.flip()
    fin=False
    while not fin:
       for event in pygame.event.get():
         if event.type == pygame.QUIT:
            fin=True
