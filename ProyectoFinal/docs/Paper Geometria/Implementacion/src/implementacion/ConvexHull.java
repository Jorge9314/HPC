package implementacion;

import java.io.File;
import java.util.Arrays;
import java.math.MathContext;
import java.util.LinkedList;
import java.util.Scanner;
import javafx.geometry.Point2D;

public class ConvexHull {
     
        private static final String RUTA_ARCHIVO = "C:\\Users\\sandra\\Documents\\NetBeansProjects\\PlantillaGraficos\\src\\main\\resources\\dat";
        
    
    public static long cross(Point O, Point A, Point B) {
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x); // se debe hallar la distancia ojo
    }
    
    public static Point[] convex_hull(Point[] P) {

       if (P.length > 1) {
			int n = P.length, k = 0;
			Point[] H = new Point[2 * n];

			Arrays.sort(P);

			// Build lower hull
			for (int i = 0; i < n; ++i) {
				while (k >= 2 && cross(H[k - 2], H[k - 1], P[i]) <= 0)
					k--;
				H[k++] = P[i];
			}

			// Build upper hull
			for (int i = n - 2, t = k + 1; i >= 0; i--) {
				while (k >= t && cross(H[k - 2], H[k - 1], P[i]) <= 0)
					k--;
				H[k++] = P[i];
			}
			if (k > 1) {
				H = Arrays.copyOfRange(H, 0, k - 1); // remove non-hull vertices after k; remove k - 1 which is a duplicate
			}
			return H;
		} else if (P.length <= 1) {
			return P;
		} else{
			return null;
		}
    }
    
    public static void main(String[] args) {
        
        long TInicio, TFin, tiempo; //Variables para determinar el tiempo de ejecución
        
        LinkedList<Point2D> vertices;
        LinkedList<Point2D> convertido; // son los convex para dibujarlos
    
        vertices = new LinkedList<>();
        vertices = cargarPuntos("100000.dat");

        System.out.println("Convex Hull Algorithm");
    
        for(int j=0 ;j < 10 ; j++){
        TInicio = System.currentTimeMillis(); //Tomamos la hora en que inicio el algoritmo y la almacenamos en la variable inicio
        //Convex Hull ( Quick Hull )
  

        int tamano = 0;
        tamano = vertices.size();

        Point[] solucion = new Point[tamano];
        Point[] puntos = new Point[tamano];

        for (int i = 0; i < vertices.size(); i++) {
            puntos[i] = new Point();
            puntos[i].x = (int) vertices.get(i).getX();
            puntos[i].y = (int) vertices.get(i).getY();
        }

        solucion = ConvexHull.convex_hull(puntos);

        convertido = new LinkedList<>();

        for (int i = 0; i < solucion.length; i++) {
            Point2D nuevoPunto = new Point2D(solucion[i].x, solucion[i].y);
            convertido.add(nuevoPunto);
        }

        tamano = convertido.size();
        Point[] puntosarea = new Point[tamano];

        for (int i = 0; i < convertido.size(); i++) {
            puntosarea[i] = new Point();
            puntosarea[i].x = (int) convertido.get(i).getX();
            puntosarea[i].y = (int) convertido.get(i).getY();
        }
        
          //algoritmo fin
        TFin = System.currentTimeMillis(); //Tomamos la hora en que finalizó el algoritmo y la almacenamos en la variable T
        tiempo = TFin - TInicio; //Calculamos los milisegundos de diferencia
        System.out.println("Tiempo de ejecución en milisegundos Quick Hull: " + tiempo + " ms"); //Mostramos en pantalla el tiempo de ejecución en milisegundos
        }
    }
    
    
    public static LinkedList<Point2D> cargarPuntos(String archivo) {
        LinkedList<Point2D> listaPuntos = new LinkedList<>();

//        try (Scanner lector = new Scanner(String.format("%s\\%s", RUTA_ARCHIVO, archivo))){
        try (Scanner lector = new Scanner(new File(String.format("%s\\%s", RUTA_ARCHIVO, archivo)))) {
//        try (Scanner lector = new Scanner(new File(String.format("dat/%s", archivo)))){
            while (lector.hasNext()) {
                String linea[] = lector.nextLine().split("   ");
                double x = Double.parseDouble(linea[0].trim().replace(',', '.'));
                double y = Double.parseDouble(linea[1].trim().replace(',', '.'));
                listaPuntos.add(new Point2D(x, y));
            }

        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return listaPuntos;
    }
}
