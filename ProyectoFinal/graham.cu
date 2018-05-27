#include <iostream>
#include <stack>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

using namespace std;

struct Point {
  /* it will represent a point */
  int x, y;
};

__constant__ Point d_p0[1];

__device__ int distSq(Point p1, Point p2)
{
    return (p1.x - p2.x)*(p1.x - p2.x) +
          (p1.y - p2.y)*(p1.y - p2.y);
}

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
__device__ int d_orientation(Point p, Point q, Point r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

__device__ int compare(Point p1,Point p2){
  // Find orientation with respect to the first point
  int o = d_orientation(d_p0[0], p1, p2);
  if(o == 0)
    return (distSq(d_p0[0], p2) >= distSq(d_p0[0],p1))? -1 : 1;

  return (o == 2)? -1: 1;
}

__device__ void swap(Point *points,uint lowIndex,uint upIndex){
  /* it will swap two points */
  Point aux = points[lowIndex];
  points[lowIndex] = points[upIndex];
  points[upIndex] = aux;
}

__global__ void sort(Point* points,uint phase,uint n){
  /* it will sort with points array with respect to phase*/
  uint ti = blockIdx.x*blockDim.x+threadIdx.x;
  if(ti >= n || ti == 0) return;

  if(ti%phase == 0){ // multiplier phase
    uint top = ti, lower = (top - phase) + 1;
    uint middle = lower + phase/2;
    uint lowG1 = lower, lowG2 = middle, topG1 = middle-1, topG2 = top;
    while(true){
      if(lowG1 > topG1 && lowG2 > topG2) break;

      // --------------------- case 1 ---------------------
      if(lowG1 <= topG1 && lowG2 <= topG2){
        Point p1 = points[lowG1];
        Point p2 = points[lowG2];
        if(compare(p1,p2) == 1){
          swap(points,lowG1,lowG2);
          lowG2++;
        }
        else lowG1++;

      }

      // --------------------- case 2 ---------------------
      else if(lowG1 < topG1 && lowG2 > topG2){
        uint next = lowG1 + 1;
        Point p1 = points[lowG1];
        Point p2 = points[next];
        if(compare(p1,p2) == 1)
          swap(points,lowG1,next);
        lowG1++;
      }

      // --------------------- case 3 ---------------------
      else if(lowG2 < topG2 && lowG1 > topG1){
        uint next = lowG2 + 1;
        Point p1 = points[lowG2];
        Point p2 = points[next];
        if(compare(p1,p2) == 1)
          swap(points,lowG2,next);
        lowG2++;
      }

      else if(lowG1 == topG1)
        lowG1++;
      else if(lowG2 == topG2)
        lowG2++;
    }
  }
}

__host__ void checkCudaState(cudaError_t& cudaState,const char *message){
  /* it will print an error message if there is */
  if(cudaState != cudaSuccess) cout << message;
}

__host__ Point nextToTop(stack<Point> &S){
  /* it will find next to top in a stack */
  Point p = S.top();
  S.pop();
  Point res = S.top();
  S.push(p);
  return res;
}

__host__ void swap(Point &p1, Point &p2){
  /* it will swap two points */
  Point temp = p1;
  p1 = p2;
  p2 = temp;
}

__host__ int h_orientation(Point p, Point q, Point r){
  /* it will determine orientation between three points */
  int val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);

  if(val == 0) return 0;  // colinear
  return (val > 0)? 1: 2; // clock or counterclock wise
}

// Prints convex hull of a set of n points.
__host__ void convexHull(Point *h_points, int n){
  /* it will find convex hull of a set of points */

  cudaError_t cudaState = cudaSuccess;
  size_t size = n*sizeof(Point);
  Point *h_result = NULL, *d_points = NULL;

  h_result = (Point *) malloc(size);
  cudaState = cudaMalloc((void**)&d_points,size);
  checkCudaState(cudaState,"Impossible allocate data for d_points\n");

  if(d_points != NULL){

    // Find the bottommost point
    int ymin = h_points[0].y, min = 0;
    for(int i = 1; i < n; i++){
     int y = h_points[i].y;

      // Pick the bottom-most or chose the left
      // most point in case of tie
      if((y < ymin) || (ymin == y && h_points[i].x < h_points[min].x))
        ymin = h_points[i].y, min = i;
    }

    // Place the bottom-most point at first position
    swap(h_points[0], h_points[min]);

    // Sort n-1 points with respect to the first point.
    // A point p1 comes before p2 in sorted ouput if p2
    // has larger polar angle (in counterclockwise
    // direction) than p1

    cudaState = cudaMemcpyToSymbol(d_p0,h_points,sizeof(Point));
    checkCudaState(cudaState,"Impossible copy data from host to device\n");

    cudaState = cudaMemcpy(d_points,h_points,size,cudaMemcpyHostToDevice);
    checkCudaState(cudaState,"Impossible copy data from host to device\n");
    dim3 gridSize((int)(ceil(n/1024.0)),1,1);
    dim3 blockSize(1024,1,1);
    uint i = 1;
    while(pow(2,i) <= n){
      sort<<<gridSize,blockSize>>>(d_points,pow(2,i),n);
      cudaDeviceSynchronize();
      i++;
    }

    cudaState = cudaMemcpy(h_result,d_points,size,cudaMemcpyDeviceToHost);
    checkCudaState(cudaState,"Impossible copy data from device to host\n");
    h_result[0] = h_points[0];
    // If two or more points make same angle with p0,
    // Remove all but the one that is farthest from p0
    // Remember that, in above sorting, our criteria was
    // to keep the farthest point at the end when more than
    // one points have same angle.
    int m = 1; // Initialize size of modified array
    for(int i=1; i<n; i++){
      // Keep removing i while angle of i and i+1 is same
      // with respect to p0
      while(i < n-1 && h_orientation(h_points[0],h_result[i],h_result[i+1]) == 0)
        i++;

      h_result[m] = h_result[i];
      m++;  // Update size of modified array
    }

    // If modified array of points has less than 3 points,
    // convex hull is not possible
    if (m < 3) return;

    // Create an empty stack and push first three points
    // to it.
    stack<Point> S;
    S.push(h_result[0]);
    S.push(h_result[1]);
    S.push(h_result[2]);

    // Process remaining n-3 points
    for(int i = 3; i < m; i++){
      // Keep removing top while the angle formed by
      // points next-to-top, top, and h_result[i] makes
      // a non-left turn
      while(h_orientation(nextToTop(S), S.top(), h_result[i]) != 2)
        S.pop();
      S.push(h_result[i]);
    }

    cout << S.size() << endl;
    // Now stack has the output points, print contents of stack
    while(!S.empty()){
      Point p = S.top();
      cout << p.x << " " << p.y << endl;
      S.pop();
    }

  }

  if(h_result != NULL) free(h_result);
  if(d_points != NULL) cudaFree(d_points);

}

// Driver program to test above functions
int main() {
  int n;
  cin >> n;
  cout << n << endl;
  //uint n = 9;
  size_t size = n*sizeof(Point);
  Point *h_points = NULL;
  h_points = (Point *) malloc(size);

  for(int i=0; i<n; i++){
    cin >> h_points[i].x;
    cin >> h_points[i].y; 
    cout << h_points[i].x << " " << h_points[i].y << endl;
  }
  /*
  if(h_points != NULL){
    h_points[0].x = 0; h_points[0].y = 3;
    h_points[1].x = 1; h_points[1].y = 1;
    h_points[2].x = 2; h_points[2].y = 2;
    h_points[3].x = 4; h_points[3].y = 4;
    h_points[4].x = 0; h_points[4].y = 0;
    h_points[5].x = 1; h_points[5].y = 2;
    h_points[6].x = 3; h_points[6].y = 1;
    h_points[7].x = 3; h_points[7].y = 3;
    h_points[8].x = 2; h_points[8].y = 1;
    convexHull(h_points, n);
  }
  */
  convexHull(h_points, n);
  if(h_points != NULL) free(h_points);
  return 0;
}
