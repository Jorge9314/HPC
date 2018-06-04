#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <cuda.h>
#include <stack>
#include <sys/time.h>
#define min(a, b) (a < b ? a : b)

bool verbose;
timeval tStart;
int tm() {
    timeval tEnd;
    gettimeofday(&tEnd, 0);
    int t = (tEnd.tv_sec - tStart.tv_sec) * 1000000 + tEnd.tv_usec - tStart.tv_usec;
    tStart = tEnd;
    return t;
}

struct Point
{
    int x, y;
};

// helper for main()

typedef struct {
    Point v;
    void* next;
} LinkNode;

// helper function for reading numbers from stdin
// it's 'optimized' not to check validity of the characters it reads in..
long readList(Point* list,int n) {
    tm();
    Point v;
    long size = 0;
    LinkNode* node = 0;
    LinkNode* first = 0;
    while (size < n) {
        LinkNode* next = new LinkNode();
        v = list[size];
        next->v = v;
        if (node)
            node->next = next;
        else
            first = next;
        node = next;
        size++;
    }


    if (size) {
        list = (Point*)malloc(n * sizeof(Point));
        LinkNode* node = first;
        long i = 0;
        while (node) {
            list[i] = node->v;
            node = (LinkNode*) node->next;
            i++;
        }

    }

    if (verbose)
        std::cout << "read stdin: " << tm() << " microseconds\n";

    return size;
}

__device__ int distSq_cuda(Point p1, Point p2){
    return (p1.x - p2.x)*(p1.x - p2.x) +
          (p1.y - p2.y)*(p1.y - p2.y);
}

__device__ int orientation_cuda(Point p, Point q, Point r){
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

__device__ bool compare_cuda(Point p1, Point p2, Point p0){

   // Find orientation
   int o = orientation_cuda(p0, p1, p2);
   if (o == 0)
     return (distSq_cuda(p0, p2) >= distSq_cuda(p0, p1))? true : false;

   return (o == 2)? true : false;
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(Point* source, Point* dest, long start, long middle, long end, Point* p0){
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || compare_cuda(source[i],source[j],p0[0]))) {
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
__global__ void gpu_mergesort(Point* source, Point* dest, long size, long width, long slices, dim3* threads, dim3* blocks, Point* p0) {
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

void mergesort(Point* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid, Point* P0) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    Point* D_data;
    Point* D_swp;
    Point* p0;
    dim3* D_threads;
    dim3* D_blocks;
    cudaError_t error = cudaSuccess;
    // Actually allocate the two arrays
    tm();

    error = cudaMalloc((void**) &p0, sizeof(Point));
    if(error != cudaSuccess){
           std::cout<<"Error reservando memoria para D_data"<<std::endl;
    }

    error = cudaMalloc((void**) &D_data, size * sizeof(Point));
    if(error != cudaSuccess){
           std::cout<<"Error reservando memoria para D_data"<<std::endl;
     }

    error = cudaMalloc((void**) &D_swp, size * sizeof(Point));
    if(error != cudaSuccess){
           std::cout<<"Error reservando memoria para D_swp"<<std::endl;
     }

    if (verbose)
        std::cout << "cudaMalloc device lists: " << tm() << " microseconds\n";

    // Copy from our input list into the first array
    cudaMemcpy(D_data, data, size * sizeof(Point), cudaMemcpyHostToDevice);

    if (verbose)
        std::cout << "cudaMemcpy list to device: " << tm() << " microseconds\n";

    //
    // Copy the thread / block info to the GPU as well
    //
    error = cudaMalloc((void**) &D_threads, sizeof(dim3));
    if(error != cudaSuccess){
           std::cout<<"Error reservando memoria para D_threads"<<std::endl;
     }
    error = cudaMalloc((void**) &D_blocks, sizeof(dim3));
    if(error != cudaSuccess){
           std::cout<<"Error reservando memoria para D_blocks"<<std:: endl;
     }

    if (verbose)
        std::cout << "cudaMalloc device thread data: " << tm() << " microseconds\n";
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    if (verbose)
        std::cout << "cudaMemcpy thread data to device: " << tm() << " microseconds\n";

    Point* A = D_data;
    Point* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //

    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        if (verbose) {
            std::cout << "mergeSort - width: " << width
                      << ", slices: " << slices
                      << ", nThreads: " << nThreads << '\n';
            tm();
        }

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks, p0);

        if (verbose)
            std::cout << "call mergesort kernel: " << tm() << " microseconds\n";

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    //
    // Get the list back from the GPU
    //
    tm();
    cudaMemcpy(data, A, size * sizeof(Point), cudaMemcpyDeviceToHost);
    if (verbose)
        std::cout << "cudaMemcpy list back to host: " << tm() << " microseconds\n";


    // Free the GPU memory
    cudaFree(D_data);
    cudaFree(D_swp);
    cudaFree(p0);
    if (verbose)
        std::cout << "cudaFree: " << tm() << " microseconds\n";
}

int Cuda_Main(int argc, char *argv[], Point* points, int tamanio, Point* p0) {
    
    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    //
    // Parse argv
    //
    tm();
    
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-' && argv[i][1] && !argv[i][2]) {
            char arg = argv[i][1];
            unsigned int* toSet = 0;
            switch(arg) {
                case 'x':
                    toSet = &threadsPerBlock.x;
                    break;
                case 'y':
                    toSet = &threadsPerBlock.y;
                    break;
                case 'z':
                    toSet = &threadsPerBlock.z;
                    break;
                case 'X':
                    toSet = &blocksPerGrid.x;
                    break;
                case 'Y':
                    toSet = &blocksPerGrid.y;
                    break;
                case 'Z':
                    toSet = &blocksPerGrid.z;
                    break;
                case 'v':
                    verbose = true;
                    break;
                default:
                    std::cout << "unknown argument: " << arg << '\n';
                    return -1;
            }

            if (toSet) {
                i++;
                *toSet = (unsigned int) strtol(argv[i], 0, 10);
            }
        }
        else {
            if (argv[i][0] == '?' && !argv[i][1])
                std::cout << "help:\n";
            else
                std::cout << "invalid argument: " << argv[i] << '\n';
            return -1;
        }
    }

    if (verbose) {
        std::cout << "parse argv " << tm() << " microseconds\n";
        std::cout << "\nthreadsPerBlock:"
                  << "\n  x: " << threadsPerBlock.x
                  << "\n  y: " << threadsPerBlock.y
                  << "\n  z: " << threadsPerBlock.z
                  << "\n\nblocksPerGrid:"
                  << "\n  x:" << blocksPerGrid.x
                  << "\n  y:" << blocksPerGrid.y
                  << "\n  z:" << blocksPerGrid.z
                  << "\n\n total threads: "
                  << threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                     blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z
                  << "\n\n";
    }

    //
    // Read numbers from stdin
    //
    long size = readList(points,tamanio);
    if (!size) return -1;

    if (verbose)
        std::cout << "sorting " << size << " numbers\n\n";

    // merge-sort the data
    mergesort(points, size, threadsPerBlock, blocksPerGrid, p0);

    tm();

    if (verbose) {
        std::cout << "print list to stdout: " << tm() << " microseconds\n";
    }
    return 0;
}

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
   p0 = points[0];

   Point* point;
   point = (Point*)malloc(n * sizeof(Point));
   Point* P0;
   P0 = (Point*)malloc(sizeof(Point));
   P0[0].x = p0.x;
   P0[0].y = p0.y;

   for (int i = 0; i < n; i++){
     point[i] = points[i];
   }

   Cuda_Main(argc, argv, point, n, P0);

   for(int i = 0; i < n; i++){
     points[i] = point[i];
     cout << "(" << points[i].x << "," << points[i].y << ")" <<endl;
   }

   free(point);
   free(P0);

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

// Driver program to test above functions
int main(int argc, char* argv[]){

    int n;
    cin >> n;
    cout << n << endl;
    Point points[n];

    for(int i = 0;  i < n; i++){
        cin >> points[i].x;
        cin >> points[i].y;
        cout << points[i].x << " " << points[i].y << endl;
    }

    convexHull(argc, argv, points, n);
    return 0;
}
