
package implementacion;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Scanner;
import java.util.Stack;
import java.io.File;
import java.math.MathContext;
import java.util.LinkedList;

public class GrahamScan{
    
    private Stack<Point2D> hull = new Stack<Point2D>();    
    private static final String RUTA_ARCHIVO = "C:\\Users\\sandra\\Documents\\NetBeansProjects\\PlantillaGraficos\\src\\main\\resources\\dat";
       

 
    public GrahamScan(Point2D[] pts)
    {
 
        // defensive copy
        int N = pts.length;
        Point2D[] points = new Point2D[N];
        for (int i = 0; i < N; i++)
            points[i] = pts[i];
        Arrays.sort(points);
 
        Arrays.sort(points, 1, N, points[0].POLAR_ORDER);
 
        hull.push(points[0]); // p[0] primer punto extremo
        int k1;
        for (k1 = 1; k1 < N; k1++)
            if (!points[0].equals(points[k1]))
                break;
        if (k1 == N)
            return; // todos los puntos
 
        int k2;
        for (k2 = k1 + 1; k2 < N; k2++)
            if (Point2D.ccw(points[0], points[k1], points[k2]) != 0)
                break;
        hull.push(points[k2 - 1]); // points[k2-1] segundo punto extremo
 
        for (int i = k2; i < N; i++)
        {
            Point2D top = hull.pop();
            while (Point2D.ccw(hull.peek(), top, points[i]) <= 0)
            {
                top = hull.pop();
            }
            hull.push(top);
            hull.push(points[i]);
        }
 
        assert isConvex();
    }
 
    public Iterable<Point2D> hull()
    {
        Stack<Point2D> s = new Stack<Point2D>();
        for (Point2D p : hull)
            s.push(p);
        return s;
    }
 
    private boolean isConvex()
    {
        int N = hull.size();
        if (N <= 2)
            return true;
 
        Point2D[] points = new Point2D[N];
        int n = 0;
        for (Point2D p : hull())
        {
            points[n++] = p;
        }
 
        for (int i = 0; i < N; i++)
        {
            if (Point2D
                    .ccw(points[i], points[(i + 1) % N], points[(i + 2) % N]) <= 0)
            {
                return false;
            }
        }
        return true;
    }
 
    // test client
    public static void main(String[] args)
    {
        
        LinkedList<Point2D> vertices;
        LinkedList<Point2D> convertido; // son los convex para dibujarlos
    
        vertices = new LinkedList<>();
        vertices = cargarPuntos("100000.dat");

        System.out.println("Graham Algorithm");

        long TInicio, TFin, tiempo; //Variables para determinar el tiempo de ejecución
        for(int j=0;j<10;j++){
        TInicio = System.currentTimeMillis(); //Tomamos la hora en que inicio el algoritmo y la almacenamos en la variable inicio
        //algoritmo inicio
        int N = 0;

        N = vertices.size();

        Point2D[] puntos = new Point2D[N];
        //llena el vector con los datos de los vertices obtenidos en pantalla
        for (int i = 0; i < vertices.size(); i++) {
            puntos[i] = new Point2D((double) vertices.get(i).x(), (double) vertices.get(i).y());
        }

        GrahamScan grahami = new GrahamScan(puntos);

        convertido = new LinkedList<>();

        for (Point2D p : grahami.hull()) {

            Point2D nuevoPunto = new Point2D(p.x(), p.y());
            convertido.add(nuevoPunto);
        }

        int tamano = 0;
        double Area = 0;
        tamano = convertido.size();
        Point[] puntosarea = new Point[tamano];

        for (int i = 0; i < convertido.size(); i++) {
            puntosarea[i] = new Point();
            puntosarea[i].x = (int) convertido.get(i).x();
            puntosarea[i].y = (int) convertido.get(i).y();
        }

        //algoritmo fin
        TFin = System.currentTimeMillis(); //Tomamos la hora en que finalizó el algoritmo y la almacenamos en la variable T
        tiempo = TFin - TInicio; //Calculamos los milisegundos de diferencia
        System.out.println("Tiempo de ejecución en milisegundos Graham: " + tiempo + "ms"); //Mostramos en pantalla el tiempo de ejecución en milisegundos
        
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
    
    

