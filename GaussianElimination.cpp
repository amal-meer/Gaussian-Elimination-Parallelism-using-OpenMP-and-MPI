// GaussianElimination.cpp : This file contains the 'main' function. Program execution begins and ends there.
// Amal Meer 1706921

#include <omp.h>
#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <string.h>
#define SIZE 800

//float b[SIZE][SIZE + 1] = { {1,2,-3,-1,0} , {0, -3,2,6,-8}, {-3,-1,3,1,0}, {2,3,2,-1,-8} };
float a[SIZE][SIZE + 1], b[SIZE][SIZE + 1];
int i, j, k, n;
time_t start, end;

void sequential_run() {

	float x[SIZE], ratio;

	start = clock(); //Start the timing

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
	/* Note: This has been changed. The original was a[n][n+1]/a[n][n], but it gives incorrect results as a has size n*n+1,
	therfore indices [0:n-1] for rows and [0:n] for columns*/
	x[n - 1] = a[n - 1][n] / a[n - 1][n - 1];

	for (i = n - 2; i >= 0; i--) {
		x[i] = a[i][n];
		for (j = i + 1; j < n; j++) {
			x[i] = x[i] - a[i][j] * x[j];
		}
		x[i] = x[i] / a[i][i];
	}

	end = clock(); //End the timing
	printf("\nExecution time of the sequential code: %f seconds \n", ((double)(end - start)) / CLOCKS_PER_SEC);

	/* Displaying Solution 
	printf("Solution of the sequential code:\n");
	for (i = 0; i < n; i++) {
		printf("x[%d] = %0.3f\n", i, x[i]);
	}
	printf("\n");*/

}

void parallel_run(int argc, char* argv[]) {

	int nproc, rank, num_rows, num_pad;
	float x[SIZE];

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nproc);

	num_rows = (int)ceil(n / (double)nproc); //compute the block size
	num_pad = nproc - n % nproc;  // we need to add rows of zeros to be able to divide the rows evenly between processors

	/* Copy the a array to a pointer to 1D to make partitioning easier */
	float* new_a = new float[(n + num_pad) * (n + 1)];

	if (rank == 0) {

#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n + 1; j++) {
				new_a[i * (n + 1) + j] = b[i][j];
			}
		}

		// Pad the extra rows with zeros
#pragma omp parallel for
		for (int i = n; i < n + num_pad; i++) {
			for (int j = 0; j < n + 1; j++) {
				new_a[i * (n + 1) + j] = 0;
			}
		}

		start = clock(); //Start the timing

	}

	// Declare a variable to store part of the array for each process
	float* sub_a = new float[num_rows * (n + 1)];

	// Partiotion, Data decomposition. Send part of the rows to each process, after this step sub_a in each process will have values
	MPI_Scatter(new_a, num_rows * (n + 1), MPI_FLOAT, sub_a, num_rows * (n + 1), MPI_FLOAT, 0, MPI_COMM_WORLD);

	// row is the normalized row that will be sent to this rank
	float* row = new float[n + 1];

	/* Applying Gauss Elimination */

	// Receivers side. Do the elimination after getting the normalized row
	int starting_row = rank * num_rows;
	for (i = 0; i < starting_row; i++) {						// For each row above the first row assigned to this processor.

		// Wait for recieving the normalized loop
		// i / num_rows to get the right root
		MPI_Bcast(row, n + 1, MPI_FLOAT, i / num_rows, MPI_COMM_WORLD);

		// Start elimination after recieving the normalized row
#pragma omp parallel for
		for (j = 0; j < num_rows; j++) { // here we start from 0 rather than i+1 because index 0 here is the i+1 index in the normal array
   
     //int w = omp_get_thread_num();
     //printf(" thread num = %d \n", w);
   
   
			for (k = i + 1; k < n + 1; k++) {
				sub_a[j * (n + 1) + k] = sub_a[j * (n + 1) + k] - (sub_a[j * (n + 1) + i]) * row[k];
			}
			sub_a[j * (n + 1) + i] = 0; // trivial case, no need to compute
		}
	}


	// Sender side.
	// Compute normalized loop.
	// To make use of parallelization, broadcast the normalized loop to other processors so that they can start elimination.
	// then continue elimination on the next rows of the matrix assigned to this processor. 
	
	int pivot_column; // it is not only i as we are not working with the whole matrix.
	for (i = 0; i < num_rows; i++) {
		pivot_column = rank * num_rows + i;

		if (sub_a[i * (n + 1) + pivot_column] == 0.0 && pivot_column<n) { // If pivot column > n-1, then this is the padded row of zeros, do not exit
			printf("Mathematical Error!");
			exit(0);
		}

#pragma omp parallel for
		for (j = pivot_column + 1; j < n + 1; j++) {
			sub_a[i * (n + 1) + j] = sub_a[i * (n + 1) + j] / sub_a[i * (n + 1) + pivot_column]; // normalize row i for each column j
		}
		sub_a[i * (n + 1) + pivot_column] = 1; // trivial case, no need to compute

		// Copy the normalized row
		memcpy(row, sub_a + (i * (n + 1)), (n + 1) * sizeof(float));

		// Broadcast the normalized row to other ranks
		MPI_Bcast(row, (n + 1), MPI_FLOAT, rank, MPI_COMM_WORLD);

		// Continue elimination for this rank
#pragma omp parallel for
		for (j = i + 1; j < num_rows; j++) {
			for (k = pivot_column + 1; k < n + 1; k++) {
				sub_a[j * (n + 1) + k] = sub_a[j * (n + 1) + k] - sub_a[j * (n + 1) + pivot_column] * row[k];
			}
			sub_a[j * (n + 1) + pivot_column] = 0; // trivial case, no need to compute
		}

	}

	// To avoid deadlock
	for (i = rank * num_rows + 1; i < n; i++) {
		MPI_Bcast(row, n + 1, MPI_FLOAT, i / num_rows, MPI_COMM_WORLD);
	}

	// Create a barrier to finish elimination befor backpropagation
	MPI_Barrier(MPI_COMM_WORLD);

	// Collect all the data to one processor
	MPI_Gather(sub_a, (n + 1) * num_rows, MPI_FLOAT, new_a, (n + 1) * num_rows, MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Finalize();

	/* Make sure the upper triangular matrix is correct 
	if (rank == 0) {

		printf("\nFinal result before backsubstitution.\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n + 1; j++) {
				printf("%.1f ", new_a[i * (n + 1) + j]);
			}
			printf("\n");
		}
	}
	*/

	/* Obtaining Solution by Back Subsitution */
	if (rank == 0) {
		x[n - 1] = new_a[(n - 1) * (n + 1) + n];

		for (i = n - 2; i >= 0; i--) {
			x[i] = new_a[i * (n + 1) + n];
			for (j = i + 1; j < n; j++) {
				x[i] = x[i] - new_a[i * (n + 1) + j] * x[j];
			}
		}
		
		end = clock(); //End the timing;
		printf("Execution time of the parallel code: %f seconds \n", ((double)(end - start)) / CLOCKS_PER_SEC);

		/* Displaying Solution 
		printf("Solution of the parallel code:\n");
		for (i = 0; i < n; i++) {
			printf("x[%d] = %0.3f\n", i, x[i]);
		}
		printf("\n");*/
	}
}

int main(int argc, char* argv[]) {
	

	/* Number of unknowns */ 
	n = SIZE;

	/* Initalize Matrix */
	i = j = 0;
	int Max_Element = 50; // range = 1-50 unsigned 
	int iseed = (unsigned int)time(NULL); 
	srand (iseed); 
	for (i=0; i<n; i++) 
		for (j=0; j<n+1; j++)
			a[i][j] = b[i][j] = (rand() / (((float)RAND_MAX + 1) /Max_Element) + 1);


	/* Choose sequential, parallel, or both */
	
	/* MPI Initialization to run sequential code only once */
	int rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if (rank == 0) {
		sequential_run();
	}

	parallel_run(argc, argv);

	return(0);
} 

