'''
quickhull.py

Algoritmo envolvente convexo rapido

Autores:
Carlos Andres Lopez
Sandra Milena Caicedo
'''
import pygame
from libpolares import *

ALTO=600
ANCHO=1000


def Extremos(ls):
    '''Recibe lista de puntos y retorna maximo y minimo
    ls: lista de puntos
    '''
    n=len(ls)
    if (n<2):
        return ls[0], ls[0]
    else:
        min_x=ls[0][0]
        max_x=ls[0][0]
        max_p=ls[0]
        min_p=ls[0]
        for e in ls:
            #print 'punto: ', e
            px=e[0]
            if px<min_x:
                min_x=px
                min_p=e
            if px>max_x:
                max_x=px
                max_p=e
        #print 'minimo: ', min_p, 'maximo: ', max_p
        return min_p, max_p


def Derecha(min_p,max_p,ls):
    ls_angulos=[]
    aux=[]
    for pto in ls:
        if pto == min_p or pto==max_p:
            pass
        else:
            an=lado(min_p,max_p,pto)
            aux.append([pto, an])
            if an>0:
                ls_angulos.append(pto)
    return ls_angulos

def Izquierda(min_p,max_p,ls):
    ls_angulos=[]
    aux=[]
    for pto in ls:
        if pto == min_p or pto==max_p:
            pass
        else:
            an=lado(min_p,max_p,pto)
            aux.append([pto, an])
            if an<0:
                ls_angulos.append(pto)
    return ls_angulos

def Envolvente(ls, p, q, lh, pl):
    '''Encontrar puntos a la derecha de la linea pq
    S:lista de puntos
    p:pto inicial
    q:pto final
    '''
    if len(ls)==0:
        return lh
    else:
        aux=[]

        for pto in ls:
            d=distancia(p,q,pto)
            aux.append([d,pto])
        aux.sort(reverse=True)
        maxd=aux[0]
        lh.append(maxd[1])
        ns=Derecha(maxd[1],p,ls)
        #seguir=input('seguir: ')
        #pygame.draw.line(pantalla,AZUL,pl.Cart(maxd[1]), pl.Cart(p))
        lh=Envolvente(ns,p, maxd[1],lh,pl)
        return lh


def quickHull(S,pl):
    a,b = Extremos(S)
    #pl.Linea(a,b)
    #print 'extremos: ', a,b
    s1= Derecha(a,b, S)
    s2= Derecha(b,a, S)

    #print 'derecha: ',s1
    ld=[]
    ld.append(b)
    ld=Envolvente(s1,b,a, ld,pl)
    #print 'definitiva: ', ld

    #print 'izquierda: ',s2

    lh=[]
    lh.append(a)
    lh=Envolvente(s2,a,b, lh,pl)
    #print 'definitiva: ',ld, lh
    #print lh
    lh.sort(reverse=True)
    ld.sort()
    lfin=lh+ld
    return lfin



if __name__ == '__main__':
    pygame.init()
    pantalla = pygame.display.set_mode((ANCHO, ALTO))
    pantalla.fill(BLANCO)
    #ejes(xc,yc)
    n_puntos= [[359,0], [320,0], [550,80], [326,66], [393,41], [413,96], [197,04], [259,82], [322,17], [344,62], [417,98], [528,73],
    [556,49], [285,79]]
    #n_puntos=[[-50,40],[10,40], [-20,20], [50,20], [30,10], [-10,-20], [-30,-20]]
    #print n_puntos
    poligono=[]
    centro=[10,490]
    pl=Cartesiano(centro, ANCHO, ALTO,pantalla)
    if len(n_puntos)>=2:
        for p in n_puntos:
            pl.Punto(p)
            #Punto(pantalla,p)
        lh=quickHull(n_puntos, pl)

        print 'final ', lh
        lhc=[]
        for p in lh:
            vp=pl.Cart(p)
            lhc.append(vp)
        #lhc.sort(reverse=True)
        pygame.draw.polygon(pantalla,ROJO,lhc,1)

    pygame.display.flip()
    fin=False
    while not fin:
       for event in pygame.event.get():
         if event.type == pygame.QUIT:
            fin=True
