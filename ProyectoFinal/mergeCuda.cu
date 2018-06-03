#include <iostream>
#include <stdio.h>
#include <malloc.h>
#include <cuda.h>
#include <sys/time.h>

// helper for main()
long readList(long**);

// data[], size, threads, blocks,
void mergesort(long*, long, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(long*, long*, long, long, long, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(long*, long*, long, long, long);

// profiling
int tm();

#define min(a, b) (a < b ? a : b)

bool verbose;

struct Point
{
    int x, y;
};

int distSq(Point p1, Point p2);

void Cuda_main(Point points[], int n) {
    
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
    /*
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
    }*/

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
    Point* data;
    long size = readList(&data,points,n);

    if (verbose)
        std::cout << "sorting " << size << " numbers\n\n";

    // merge-sort the data
    //mergesort(data, size, threadsPerBlock, blocksPerGrid);

    tm();

    //
    // Print out the list
    //
    /*
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << '\n';
    }
    */
    if (verbose) {
        std::cout << "print list to stdout: " << tm() << " microseconds\n";
    }
}

void mergesort(long* data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid) {

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    long* D_data;
    long* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    cudaError_t error = cudaSuccess;
    // Actually allocate the two arrays
    tm();
    std::cout<<"reservando memoria con cudamalloc"<<std::endl;
    error = cudaMalloc((void**) &D_data, size * sizeof(long));
    if(error != cudaSuccess){
           std::cout<<"Error reservando memoria para D_data"<<std::endl;
     }
    std::cout<<"pass 1"<<std::endl;
    error = cudaMalloc((void**) &D_swp, size * sizeof(long));
    if(error != cudaSuccess){
           std::cout<<"Error reservando memoria para D_swp"<<std::endl;
     }
    std::cout<<"pass 2"<<std::endl;
    if (verbose)
        std::cout << "cudaMalloc device lists: " << tm() << " microseconds\n";

    // Copy from our input list into the first array
    cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice);
    std::cout<<"copy 1"<<std::endl;
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

    std::cout<<"pass t and b"<<std::endl;

    if (verbose)
        std::cout << "cudaMalloc device thread data: " << tm() << " microseconds\n";
    cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice);
    cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice);

    std::cout<<"copy t and b"<<std::endl;

    if (verbose)
        std::cout << "cudaMemcpy thread data to device: " << tm() << " microseconds\n";

    long* A = D_data;
    long* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    std::cout<<"antes del ciclo for extraño"<<std::endl;

    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        if (verbose) {
            std::cout << "mergeSort - width: " << width
                      << ", slices: " << slices
                      << ", nThreads: " << nThreads << '\n';
            tm();
        }

        // Actually call the kernel
        std::cout<< "llamando a a GPU"<<std::endl;
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);
        std::cout<< "saliendo de la GPU"<<std::endl;

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
    cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost);
    if (verbose)
        std::cout << "cudaMemcpy list back to host: " << tm() << " microseconds\n";


    // Free the GPU memory
    cudaFree(A);
    cudaFree(B);
    if (verbose)
        std::cout << "cudaFree: " << tm() << " microseconds\n";
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
__global__ void gpu_mergesort(long* source, long* dest, long size, long width, long slices, dim3* threads, dim3* blocks) {
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices,
         middle,
         end;

    for (long slice = 0; slice < slices; slice++) {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(long* source, long* dest, long start, long middle, long end) {
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

// read data into a minimal linked list
typedef struct {
    Point v;
    void* next;
} LinkNode;

// helper function for reading numbers from stdin
// it's 'optimized' not to check validity of the characters it reads in..
long readList(Point** list,Point points[],int s) {
    tm();
    Point v; 
    long size = 0;
    LinkNode* node = 0;
    LinkNode* first = 0;
    while (size < s) {
        LinkNode* next = new LinkNode();
        v.x = points[size].x;
        v.y = points[size].y;
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
        *list = new Point[size];
        LinkNode* node = first;
        long i = 0;
        while (node) {

            (*list)[i++].x = node->v.x;
            (*list)[i++].y = node->v.y;
            node = (LinkNode*) node->next;
        }

    }

    if (verbose)
        std::cout << "read stdin: " << tm() << " microseconds\n";

    return size;
}


//
// Get the time (in microseconds) since the last call to tm();
// the first value returned by this must not be trusted
//
timeval tStart;
int tm() {
    timeval tEnd;
    gettimeofday(&tEnd, 0);
    int t = (tEnd.tv_sec - tStart.tv_sec) * 1000000 + tEnd.tv_usec - tStart.tv_usec;
    tStart = tEnd;
    return t;
}

// A utility function to return square of distance
// between p1 and p2
int distSq(Point p1, Point p2)
{
    return (p1.x - p2.x)*(p1.x - p2.x) +
          (p1.y - p2.y)*(p1.y - p2.y);
}

