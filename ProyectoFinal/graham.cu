// A C++ program to find convex hull of a set of points. Refer
// http://www.geeksforgeeks.org/orientation-3-ordered-points/
// for explanation of orientation()

#include <iostream>
#include <stack>
#include <stdlib.h>
#define min(a, b) (a < b ? a : b)


using namespace std;

struct Point{
    int x, y;
};

// A globle point needed for  sorting points with reference
// to  the first point Used in compare function of qsort()

// A utility function to find next to top in a stack
Point nextToTop(stack<Point> &S){
    Point p = S.top();
    S.pop();
    Point res = S.top();
    S.push(p);
    return res;
}

// A utility function to swap two points
void swap(Point &p1, Point &p2){
    Point temp = p1;
    p1 = p2;
    p2 = temp;
}

// A utility function to return square of distance
// between p1 and p2
int distSq(Point p1, Point p2){
    return (p1.x - p2.x)*(p1.x - p2.x) +
          (p1.y - p2.y)*(p1.y - p2.y);
}

// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
__device__ int orientation_cuda(Point p, long q[], long r[]){
    int val = (q[1] - p.y) * (r[0] - q.x) -
              (q[0] - p.x) * (r[1] - q.y);

    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

// A function used by library function qsort() to sort an array of
// points with respect to the first point
__device__ bool compare_cuda(long p1[], long p2[], Point p0){

   // Find orientation
   int o = orientation_cuda(p0, p1, p2);
   if (o == 0)
     return false;

   return (o == 2)? true: false;
}

__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end, Point p0){
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || compare_cuda(source[i],source[j],p0))) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks, Point p0) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices,
         middle,
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end, p0);
        start += width;
    }
}

int orientation(Point p, Point q, Point r){
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

void Cuda_Main(Point p[], int s, Point p0){

  long size = 0;
  long** points;
  points = (long**)malloc(s-1 * sizeof(long*));

  for(int i = 0; i < s-1; i++){
    points[i] = (long*)malloc(2*sizeof(long));
    size++;
  }

  for(int i = 0; i < s-1; i++){
    for(int j = 0; j < 2; j++){
      if(j == 0){
        points[i][j] = p[i].x; 
      }else{
        points[i][j] = p[i].y;
      }
    }
  }

  long** D_data;
  long** D_swp;
  cout<<"cuda malloc data and swp"<<endl;
  cudaMalloc((void**)&D_data, size * sizeof(long*));
  cudaMalloc((void**)&D_swp, size * sizeof(long*));
  cout<<"cuda malloc fin..."<<endl;

  cudaMemcpy(D_data, data, size * sizeof(Point), cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(32,1,1);
  dim3 blocksPerGrid(8,1,1);

  dim3* D_threads;
  dim3* D_blocks;
  cudaMalloc((void**)&D_threads, size * sizeof(dim3));
  cudaMalloc((void**)&D_blocks , size * sizeof(dim3));

  cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
  cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

  long *A = D_data;
  long *B = D_swp;

  long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                  blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

  for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        // Actually call the kernel
        std::cout<< "llamando a a GPU"<<std::endl;
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks, p0);
        std::cout<< "saliendo de la GPU"<<std::endl;

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

}

// Prints convex hull of a set of n points.
void convexHull(int argc, char* argv[], Point points[], int n)
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
   Point p0 = points[0];

   Cuda_Main(points, n, p0);

   for(int i = 0; i < n; i++){
     //points[i] = point[i];
     cout << "(" << points[i].x << "," << points[i].y << ")" <<endl;
   }

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
}

void imp_points(int p[][2], int size){

  for (int i = 0; i < size; ++i){
    cout<<p[i][0]<<" "<<p[i][1]<<endl;
  }
}

// Driver program to test above functions
int main(int argc, char* argv[]){

    int n;
    cin >> n;
    cout << n << endl;
    int points[n][2];

    for(int i = 0;  i < n; i++){
        cin >> points[i][0];
        cin >> points[i][1];
    }

    imp_points(points,n);

    //convexHull(argc, argv, points, n);
    return 0;
}
