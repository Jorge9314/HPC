// A C++ program to find convex hull of a set of points. Refer
// http://www.geeksforgeeks.org/orientation-3-ordered-points/
// for explanation of orientation()

#include <iostream>
#include <stack>
#include <stdlib.h>
#include <cuda.h>
#include "mergeCuda.cu"

using namespace std;

struct Point
{
    int x, y;
};

// A globle point needed for  sorting points with reference
// to  the first point Used in compare function of qsort()
Point p0;

// A utility function to find next to top in a stack
Point nextToTop(stack<Point> &S)
{
    Point p = S.top();
    S.pop();
    Point res = S.top();
    S.push(p);
    return res;
}

// A utility function to swap two points
void swap(Point &p1, Point &p2)
{
    Point temp = p1;
    p1 = p2;
    p2 = temp;
}

// A utility function to return square of distance
// between p1 and p2
int distSq(Point p1, Point p2)
{
    return (p1.x - p2.x)*(p1.x - p2.x) +
          (p1.y - p2.y)*(p1.y - p2.y);
}

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(Point p, Point q, Point r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

// A function used by library function qsort() to sort an array of
// points with respect to the first point
int compare(const void *vp1, const void *vp2)
{
   Point *p1 = (Point *)vp1;
   Point *p2 = (Point *)vp2;

   // Find orientation
   int o = orientation(p0, *p1, *p2);
   if (o == 0)
     return (distSq(p0, *p2) >= distSq(p0, *p1))? -1 : 1;

   return (o == 2)? -1: 1;
}

void mainMergeSortCuda(long *v, long n){

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    //
    // Read numbers from stdin
    //
    long* data;
    long size = n;
    data = v;
    
    for(int i = 0; i < n; i++){
      cout<<data[i]<<endl;
    }

    // merge-sort the data
    //mergesort(data, size, threadsPerBlock, blocksPerGrid);

    //
    // Print out the list
    //
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << '\n';
    }
}

// Prints convex hull of a set of n points.
void convexHull(Point points[], int n)
{
   // Find the bottommost point
   int ymin = points[0].y, min = 0;
   for (int i = 1; i < n; i++)
   {
     int y = points[i].y;

     // Pick the bottom-most or chose the left
     // most point in case of tie
     if ((y < ymin) || (ymin == y &&
         points[i].x < points[min].x))
        ymin = points[i].y, min = i;
   }

   // Place the bottom-most point at first position
   swap(points[0], points[min]);

   // Sort n-1 points with respect to the first point.
   // A point p1 comes before p2 in sorted ouput if p2
   // has larger polar angle (in counterclockwise
   // direction) than p1
   p0 = points[0];
   cout<<points[0].x<<" "<<points[0].y<<endl;

   //qsort(&points[1], n-1, sizeof(Point), compare);

   cout << "sacando distancias"<<endl; 
   long *distance;
   distance = (long*)malloc(n-1*sizeof(long));

   //extract distances
   for(int i = 0; i < n-1; i++){
    distance[i] = distSq(p0,points[i+1]);
    cout<<"("<<distance[i]<<")"<<endl;
   }

   long size = n-1;
   dim3 dimBlock(32,1,1);
   dim3 dimGrid(8,1,1);

   cout<<"ordenando distancias"<<endl;

   mainMergeSortCuda(distance, size);

   for(int i = 0; i < n-1; i++){
      cout<<"["<<distance[i]<<"]"<<endl;
   }

   cout<<"datos ordenados"<<endl;

   // If two or more points make same angle with p0,
   // Remove all but the one that is farthest from p0
   // Remember that, in above sorting, our criteria was
   // to keep the farthest point at the end when more than
   // one points have same angle.
   int m = 1; // Initialize size of modified array
   for (int i=1; i<n; i++)
   {
       // Keep removing i while angle of i and i+1 is same
       // with respect to p0
       while (i < n-1 && orientation(p0, points[i],
                                    points[i+1]) == 0)
          i++;


       points[m] = points[i];
       m++;  // Update size of modified array
   }

   // If modified array of points has less than 3 points,
   // convex hull is not possible
   if (m < 3) return;

   // Create an empty stack and push first three points
   // to it.
   stack<Point> S;
   S.push(points[0]);
   S.push(points[1]);
   S.push(points[2]);

   // Process remaining n-3 points
   for (int i = 3; i < m; i++)
   {
      // Keep removing top while the angle formed by
      // points next-to-top, top, and points[i] makes
      // a non-left turn
      while (orientation(nextToTop(S), S.top(), points[i]) != 2)
         S.pop();
      S.push(points[i]);
   }

   cout << S.size() << endl;

   // Now stack has the output points, print contents of stack
   while (!S.empty())
   {
       Point p = S.top();
       cout << p.x << " " << p.y << endl;
       S.pop();
   }
   free(distance);
}

// Driver program to test above functions
int main(){
    int n;
    cin >> n;
    cout << n << endl;
    Point points[n];

    for(int i = 0;  i < n; i++){
        cin >> points[i].x;
        cin >> points[i].y;
        cout << points[i].x << " " << points[i].y << endl;
    }

    convexHull(points, n);
    return 0;
}
