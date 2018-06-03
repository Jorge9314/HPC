#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <cuda.h>
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
        v.x = list[size].x;
        v.y = list[size].y;
        next->v.x = v.x;
        next->v.y = v.y;
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
            list[i].x = node->v.x;
            list[i].y = node->v.y;
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

__device__ bool compare_cuda(const void *vp1, const void *vp2, const void *P0){
   Point *p1 = (Point *)vp1;
   Point *p2 = (Point *)vp2;
   Point *p0 = (Point *)P0;

   // Find orientation
   int o = orientation_cuda(p0, *p1, *p2);
   if (o == 0)
     return (distSq_cuda(p0, *p2) >= distSq_cuda(p0, *p1))? true : false;

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
    cudaFree(A);
    cudaFree(B);
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

    //
    // Print out the list
    //

    if (verbose) {
        std::cout << "print list to stdout: " << tm() << " microseconds\n";
    }
    return 0;
}
/*
int main(int argc, char *argv[]){

    int n;
    std::cin >> n;
    std::cout << n << std::endl;
    Point points[n];

    Point *p;
    p = (Point*)malloc(n * sizeof(Point));

    for(int i = 0;  i < n; i++){
        std::cin >> points[i].x;
        std::cin >> points[i].y;
        std::cout << points[i].x << " " << points[i].y << std::endl;
        p[i] = points[i];
    }

    Point* p0;
    p0 = (Point*)malloc(sizeof(Point));
    p0[0].x = 300;
    p0[0].y = 300;

    return Cuda_Main(argc,argv,p, n,p0);
}*/