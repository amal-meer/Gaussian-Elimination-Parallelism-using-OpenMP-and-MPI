# Gaussian Elimination Parallelism using OpenMP and MPI

This is a High Performance Computing course project, an elective course of the Computer Science bechalor's degree from King AbdulAziz Univirsity.

## Provided Code and Required Task

A working C++ code of the Gaussian Elimination algorithm is provided. Gaussian Elimination algorithm is used to solve a system of linear equations. The task is to reduce the running time of that code using MPI/OpenMP and using the available compute resources on the thin or fat queue on Aziz Supercomputer.

Below is the C++ code followed by the resources in the thin and fat queue on Aziz Supercomputer. 

### C++ code of the Gaussian Elimination algorithm
```C++
// GaussianElimination.cpp : This file contains the 'main' function. 
#include<stdio.h> 
#include<conio.h> 
#include<math.h> 
#include<stdlib.h>
#define SIZE 10
int main() {
    
    float a[SIZE][SIZE], x[SIZE], ratio; int i, j, k, n;

    /* 1. Reading number of unknowns */ 
    printf("Enter number of unknowns: "); 
    scanf("%d", &n); 

    /* 2. Reading Augmented Matrix */ 
    for (i = 0; i < n; i++) {
        for (j = 0; j < n + 1; j++) {
            printf("a[%d][%d] = ", i, j); scanf("%f", &a[i][j]);
        }
    } 

    /* Applying Gauss Elimination */ 
    for (i = 0; i < n - 1; i++) {
        if (a[i][i] == 0.0) {
            printf("Mathematical Error!"); 
            exit(0); 
        }

        for (j = i + 1; j < n; j++) {
            ratio = a[j][i] / a[i][i];
            for (k = 0; k < n + 1; k++) {
                a[j][k] = a[j][k] - ratio * a[i][k]; 
            } 
        }
    } 

    /* Obtaining Solution by Back Subsitution */ 
    x[n-1] = a[n-1][n] / a[n-1][n-1];
    
    for (i = n - 2; i >= 0; i--) {
        x[i] = a[i][n]; 
        for (j = i + 1; j < n; j++) {
            x[i] = x[i] - a[i][j] * x[j];
        }
        x[i] = x[i] / a[i][i];
    }

    /* Displaying Solution */ 
    printf("\nSolution:\n"); 
    for (i = 0; i < n; i++) {
        printf("x[%d] = %0.3f\n", i, x[i]); 
    } 
    return(0); 
} 
```

### Available Resources in the thin and fat queue on Aziz Supercomputer.
* **Thin queue:** 380 Compute Nodes, 24 CPU on each node, 96 GB memory on each node, Total of 9120 cores. 
* **Fat queue:** 112 Compute Nodes, 24 CPU on each node, 256 GB memory on each node, Total of 2688 cores.

## Implementation Files description
1. **GaussianElimination.cpp**

The parallelized C++ code.

2. **Report.pdf**

The report that documents the optimizations and strategies in reducing the runtime. It includes the following:
* The OpenMP directives used and the justification to use them over others. 
* The MPI collective communication calls used and the justification to use them over other collective or point-to-point communication calls.
* The test environment used, i.e., name of the queue and compiler used, the PBS job script, and the Best speedup obtained. 

3. **job.PBS**

The PBS job scripts that is used to allocate resources and submit a job to Aziz supercomputer.

## Compile and Run instructions
To compile the C++ code on Aziz, use the following commands on Aziz terminal
```
module load impi/5.0.3.048
mpicc -fopenmp -o Gaussian GaussianElimination.cpp -lm -lstdc++
```

where GaussianElimination.cpp is the c++ code, and Gaussian is the resulted executable file. To run the executable file on Aziz, edit the PBS script based on the needed resources and run it using ```qsub job.pbs```

## Results
The result is a parallelized code that uses multithreaded and distributed computing and an obtained average speedup of 23.79

## Resources
[Gaussian Example to check code correctence](https://www.matesfacil.com/english/high/solving-systems-by-Gaussian-Elimination.html)

[practical_parallelism_in_cpp](https://github.com/CoffeeBeforeArch/practical_parallelism_in_cpp/tree/master/parallel_algorithms/gaussian_elimination/mpi/naive)