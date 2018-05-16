
import pygame
from libpolares import *
import math

ALTO=600
ANCHO=1000

def rotar(A,B,C):
  return (B[0]-A[0])*(C[1]-B[1])-(B[1]-A[1])*(C[0]-B[0])

def grahamborde(A):
  n = len(A) # number of points
  P = range(n) # the list of point numbers
  for i in range(1,n):
    if A[P[i]][0]<A[P[0]][0]:
      P[i], P[0] = P[0], P[i] # swap the numbers of these points
  for i in range(2,n): # insertion sort
    j = i
    while j>1 and (rotar(A[P[0]],A[P[j-1]],A[P[j]])<0):
      P[j], P[j-1] = P[j-1], P[j]
      j -= 1
  S = [P[0],P[1]] # create the stack
  for i in range(2,n):
    while rotar(A[S[-2]],A[S[-1]],A[P[i]])<0:
      del S[-1] # pop(S)
    S.append(P[i]) # push(S,P[i])
  return S

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
    ls_ind=grahamborde(n_puntos)
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
