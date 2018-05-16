'''
libpolares.py

Libreria de clases para gestion de coordenas polares

'''

import pygame
import math

ROJO=(255,0,0)
VERDE=(0,255,0)
AZUL=(0,0,255)
BLANCO=(255,255,255)

def Punto(p, pto):
    pygame.draw.circle(p, ROJO, pto, 2)

class Cartesiano:
    centro=[]
    ancho=100
    alto=100

    def __init__(self, centro, ancho, alto, pantalla):
        '''Constructor
        centro: origen del plano
        ancho: ancho de la pantalla
        alto: alto de la pantalla
        pantalla: variable de gestion grafica
        '''
        self.centro=centro
        self.ancho=ancho
        self.alto=alto
        self.p=pantalla
        self.ejes()

    def Limpiar(self):
        self.p.fill(BLANCO)
        self.ejes()

    def ejes(self):
        cx=self.centro[0]
        cy=self.centro[1]
        pygame.draw.line(self.p, ROJO, [cx,0], [cx, self.alto])
        pygame.draw.line(self.p, ROJO, [0,cy], [self.ancho, cy])

    def Cart(self, pto):
        cx=self.centro[0]
        cy=self.centro[1]
        px=pto[0]
        py=pto[1]
        x=cx+px
        y=cy-py
        return [x,y]

    def Punto(self, pto, cl=ROJO):
        pygame.draw.circle(self.p, cl, self.Cart(pto), 2)

    def Linea(self, pini, pfin, cl=ROJO):
        cpini=self.Cart(pini)
        cpfin=self.Cart(pfin)
        pygame.draw.line(self.p, cl, cpini, cpfin, 1)


class Polar (Cartesiano):


    def __init__(self, centro, ancho, alto, pantalla ):
        Cartesiano.__init__(self, centro, ancho, alto, pantalla)

    def Polares(self,r, an):
        ang=math.radians(an)
        x=r*math.cos(ang)
        y=r*math.sin(ang)
        pp=[int(x), int(y)]
        self.Punto(pp)
        self.Linea([0,0], pp)

def norma(p):
    '''Retorna la norma del vector
    p: punto
    '''
    r=len(p)
    suma=0
    for i in range(r):
        suma+=p[i]**2
    res=math.sqrt(suma)
    return res

def prod(u,v):
    '''Producto punto entre vectores
    u: vector
    v: vector
    '''
    res= (u[0]*v[0]) + (u[1]*v[1])
    return res

def vecCanonico(p1,p2):
    '''Retorna el vector canonico dados dos puntos
    p1: punto inicial
    p2:punto final
    '''
    xc=p2[0]-p1[0]
    yc=p2[1]-p1[1]
    return [xc,yc]

def Angulo(u,v):
    '''Angulo enrte dos vectores
    '''
    num=float(prod(u,v))
    den=float(norma(u))*float(norma(v))
    val=num/den
    an=math.acos(val)
    return an

def AnguloPtos(p1,p2,p3):
    '''Encuentra el angulo enrte tres puntos
    p1,p2,p3: puntos
    '''
    u=vecCanonico(p1,p3)
    v=vecCanonico(p1,p2)
    an1=math.atan2(u[1],u[0])
    an2=math.atan2(v[1],v[0])
    coran1=an1+math.fabs(an2)
    return coran1

def lado(p1,p2,p):
    val = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])
    if (val > 0):
        return 1
    if (val < 0):
        return -1


def Proyec(u,v):
    '''Proyeccion del vector u sobre vector v
    u: vector
    v: vector
    '''
    #print u,v
    res= float(prod(u,v)) / float(norma(v))
    return res

def distancia(p1,p2,p3):
    ''' Distancia entre rtes puntos
    p1: punto 1
    p2: punto 2
    p3: punto 3
    '''
    u=vecCanonico(p1,p3)
    v=vecCanonico(p1,p2)
    c1=Proyec(u,v)
    h=float(norma(u))
    val= (h**2) - (c1**2)
    #print val
    d=math.sqrt(val)
    return d
