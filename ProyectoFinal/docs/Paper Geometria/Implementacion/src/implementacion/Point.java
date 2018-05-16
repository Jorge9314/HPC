/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package implementacion;

/**
 *
 * @author sandra
 */
public class Point implements Comparable<Point> {
	public int x, y;
        public double x2,y2;

    public String getPos() {
        String s_pos = "(" + String.valueOf(x) + ","  + String.valueOf(y) + ")";
        return s_pos;
    }

        
	public int compareTo(Point p) {
		if (this.x == p.x) {
			return this.y - p.y;
		} else {
			return this.x - p.x;
		}
	}

	public String toString() {
		return "("+x + "," + y+")";
	}
        
        
    public  double left( Point b, Point c) {
        double area2 = (b.x2 - this.x2) * (c.y2 - this.y2) - (c.x2 - this.x2) * (b.y2 - this.y2);
        if (area2 < 0) {
            return -1;
        } else if (area2 > 0) {
            return +1;
        } else {
            return 0;
        }
    }

}