/**
 * @file fft_hybrid.c
 * @brief Hybrid MPI+OpenMP implementation of 2D Fast Fourier Transform
 *
 * This file implements a parallel version of 2D FFT using both MPI for
 * distributed memory parallelization and OpenMP for shared memory
 * parallelization. The implementation distributes rows across MPI processes and
 * uses OpenMP to parallelize operations within each process.
 */

#include <complex.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cshift.h"
#include "pgm.h"

#define PI 3.14159265358979323846

typedef double complex cplx;

cplx* mat2vet(cplx** mat, int width, int height) {
    cplx* v = (cplx*)malloc(height * width * sizeof(cplx));
#pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            v[i * width + j] = mat[i][j];
        }
    }
    return v;
}

cplx** vet2mat(cplx* v, int width, int height) {
    cplx** mat = (cplx**)malloc(height * sizeof(cplx*));
    for (int i = 0; i < height; i++) {
        mat[i] = (cplx*)malloc(width * sizeof(cplx));
    }
#pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            mat[i][j] = v[i * width + j];
        }
    }
    return mat;
}

int nextPowerOf2(int num) {
    int power = 1;
    while (power < num) {
        power *= 2;
    }
    return power;
}

int is_power_of_two(int x) { 
    return (x != 0) && ((x & (x - 1)) == 0); 
}

/**
 * @brief Apply zero padding to make image square and power-of-2 sized
 * @param image Input image structure
 * @return Padded image structure
 */
pgm_t zeroPadding(const pgm_t image) {
    pgm_t paddedImage;
    int newWidth = nextPowerOf2(image.width);
    int newHeight = nextPowerOf2(image.height);

    /* If the image is not square, pad to make it square */
    if (newWidth != newHeight) {
        newWidth = newHeight = (newWidth > newHeight) ? newWidth : newHeight;
    }

    /* Allocate memory for the padded image */
    paddedImage.data = (cplx**)malloc(newHeight * sizeof(cplx*));
    for (int i = 0; i < newHeight; i++) {
        paddedImage.data[i] = (cplx*)calloc(newWidth, sizeof(cplx));
    }

    /* Copy the original image into the padded image with OpenMP */
#pragma omp parallel for
    for (int i = 0; i < image.height; i++) {
        for (int j = 0; j < image.width; j++) {
            paddedImage.data[i][j] = image.data[i][j];
        }
    }

    /* Copy image metadata */
    paddedImage.width = newWidth;
    paddedImage.height = newHeight;
    paddedImage.max = image.max;
    strcpy(paddedImage.type, image.type);

    return paddedImage;
}

/**
 * @brief Cooley-Tukey FFT algorithm with OpenMP parallelization
 * @param x       Input/output complex array (modified in-place)
 * @param N       Size of the input array (must be power of 2)
 * @param inverse 0 for forward FFT, 1 for inverse FFT
 */
void cooley_tukey_fft(cplx x[], int N, int inverse) {
    /* Bit-reversal permutation */
    int i, j, k;
    for (i = 1, j = N / 2; i < N - 1; i++) {
        if (i < j) {
            cplx temp = x[i];
            x[i] = x[j];
            x[j] = temp;
        }
        k = N / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    /* Iterative FFT or IFFT computation with OpenMP */
    double sign = (inverse) ? 1.0 : -1.0; /* Sign for IFFT */
    for (int s = 1; s <= log2(N); s++) {
        int m = 1 << s; /* Subproblem size */
        cplx omega_m = cexp(sign * I * 2.0 * PI / m);

#pragma omp parallel for
        for (int k = 0; k < N; k += m) {
            cplx omega = 1.0;

            for (int j = 0; j < m / 2; j++) {
                cplx t = omega * x[k + j + m / 2];
                cplx u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m / 2] = u - t;
                omega *= omega_m;
            }
        }
    }
}

/**
 * @brief Transpose a matrix represented as a flattened vector with OpenMP
 * @param v      Input vector (flattened matrix)
 * @param width  Number of columns
 * @param height Number of rows
 * @return Pointer to transposed vector
 */
cplx* transpose(cplx* v, int width, int height) {
    cplx* tmp = (cplx*)malloc(height * width * sizeof(cplx));

#pragma omp parallel for
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            tmp[j * height + i] = v[i * width + j];
        }
    }

    return tmp;
}

/**
 * @brief Main function for hybrid MPI+OpenMP 2D FFT processing
 * @param argc Argument count
 * @param argv Argument vector (argv[1] should be input image filename)
 * @return Exit status
 */
int main(int argc, char** argv) {
    int rank, size;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    pgm_t img;
    cplx* v_send;
    cplx* v_revc;
    int len_info[5]; /* [width, height, max, orig_width, orig_height] */

    if (rank == 0) {
        /* Master process reads and prepares data */
        img = pgm_read(argv[1]);
        
        /* Store original dimensions */
        len_info[3] = img.width;
        len_info[4] = img.height;
        
        /* Apply zero padding if necessary */
        if (!is_power_of_two(img.width * img.height) || img.width != img.height) {
            img = zeroPadding(img);
        }
        
        /* Convert image matrix to vector format */
        v_send = mat2vet(img.data, img.width, img.height);
        
        /* Store processed dimensions */
        len_info[0] = img.width;
        len_info[1] = img.height;
        len_info[2] = img.max;
    }

    /* Broadcast image information to all processes */
    MPI_Bcast(len_info, 5, MPI_INT, 0, MPI_COMM_WORLD);

    /* Calculate data distribution among processes */
    int rows_per_processor = len_info[1] / size;
    int remainder = len_info[1] % size;

    /* Determine if this process gets an extra row */
    int processors_with_extra_rows = (rank < remainder) ? 1 : 0;
    int my_num_rows = rows_per_processor + processors_with_extra_rows;

    /* Calculate displacements and receive counts for MPI_Scatterv */
    int* displacements = (int*)malloc(size * sizeof(int));
    int* recvcounts = (int*)malloc(size * sizeof(int));

    int displacement = 0;
    for (int i = 0; i < size; i++) {
        int extra_rows = (i < remainder) ? 1 : 0;
        recvcounts[i] = (rows_per_processor + extra_rows) * len_info[0];
        displacements[i] = displacement;
        displacement += recvcounts[i];
    }

    /* Allocate memory for local data chunk */
    v_revc = (cplx*)malloc(my_num_rows * len_info[0] * sizeof(cplx));

    /* Distribute data to all processes */
    MPI_Scatterv(v_send, recvcounts, displacements, MPI_C_DOUBLE_COMPLEX, 
                 v_revc, my_num_rows * len_info[0], MPI_C_DOUBLE_COMPLEX, 
                 0, MPI_COMM_WORLD);

    double fft_start = MPI_Wtime();

    /*================== START 2D FFT ==================*/
    /* Perform 1D FFT on each row with OpenMP parallelization */
#pragma omp parallel for
    for (int i = 0; i < my_num_rows; i++) {
        cooley_tukey_fft(v_revc + i * len_info[0], len_info[0], 0);
    }

    /* Gather results back to master process */
    MPI_Gatherv(v_revc, my_num_rows * len_info[0], MPI_C_DOUBLE_COMPLEX, 
                v_send, recvcounts, displacements, MPI_C_DOUBLE_COMPLEX, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* Transpose the data */
        cplx* v_transposed = transpose(v_send, len_info[0], len_info[1]);
        free(v_send);
        v_send = v_transposed;
    }

    /* Redistribute transposed data */
    MPI_Scatterv(v_send, recvcounts, displacements, MPI_C_DOUBLE_COMPLEX, 
                 v_revc, my_num_rows * len_info[0], MPI_C_DOUBLE_COMPLEX, 
                 0, MPI_COMM_WORLD);

    /* Perform 1D FFT on each column (now row after transpose) with OpenMP */
#pragma omp parallel for
    for (int i = 0; i < my_num_rows; i++) {
        cooley_tukey_fft(v_revc + i * len_info[0], len_info[0], 0);
    }
    /*================== END 2D FFT ==================*/

    double fft_end = MPI_Wtime();

    /* Gather FFT results */
    MPI_Gatherv(v_revc, my_num_rows * len_info[0], MPI_C_DOUBLE_COMPLEX, 
                v_send, recvcounts, displacements, MPI_C_DOUBLE_COMPLEX, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* Transpose back and apply FFT shift for visualization */
        cplx* v_transposed = transpose(v_send, len_info[0], len_info[1]);
        free(v_send);
        v_send = v_transposed;
        
        img.data = vet2mat(fftshift(v_send, len_info[0], len_info[1]), 
                          len_info[0], len_info[1]);
        pgm_write_fft(img, "fft.pgm", "");
        
        /* Prepare for inverse FFT */
        v_send = ifftshift(mat2vet(img.data, len_info[0], len_info[1]), 
                          len_info[0], len_info[1]);
    }

    /*================== START 2D IFFT ==================*/
    /* Distribute data for IFFT */
    MPI_Scatterv(v_send, recvcounts, displacements, MPI_C_DOUBLE_COMPLEX, 
                 v_revc, my_num_rows * len_info[0], MPI_C_DOUBLE_COMPLEX, 
                 0, MPI_COMM_WORLD);

    /* Perform 1D IFFT on each row with OpenMP parallelization */
#pragma omp parallel for
    for (int i = 0; i < my_num_rows; i++) {
        cooley_tukey_fft(v_revc + i * len_info[0], len_info[0], 1);
    }

    /* Gather and transpose */
    MPI_Gatherv(v_revc, my_num_rows * len_info[0], MPI_C_DOUBLE_COMPLEX, 
                v_send, recvcounts, displacements, MPI_C_DOUBLE_COMPLEX, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        cplx* v_transposed = transpose(v_send, len_info[0], len_info[1]);
        free(v_send);
        v_send = v_transposed;
    }

    /* Redistribute for column IFFT */
    MPI_Scatterv(v_send, recvcounts, displacements, MPI_C_DOUBLE_COMPLEX, 
                 v_revc, my_num_rows * len_info[0], MPI_C_DOUBLE_COMPLEX, 
                 0, MPI_COMM_WORLD);

    /* Perform 1D IFFT on each column (now row after transpose) with OpenMP */
#pragma omp parallel for
    for (int i = 0; i < my_num_rows; i++) {
        cooley_tukey_fft(v_revc + i * len_info[0], len_info[0], 1);
    }

    /* Gather final results */
    MPI_Gatherv(v_revc, my_num_rows * len_info[0], MPI_C_DOUBLE_COMPLEX, 
                v_send, recvcounts, displacements, MPI_C_DOUBLE_COMPLEX, 
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        /* Master process finalizes the results */
        cplx* v_transposed = transpose(v_send, len_info[0], len_info[1]);
        free(v_send);
        v_send = v_transposed;

        /* Normalize by dividing by total number of pixels with OpenMP */
#pragma omp parallel for
        for (int i = 0; i < len_info[0] * len_info[1]; i++) {
            v_send[i] /= (len_info[0] * len_info[1]);
        }

        /* Convert back to matrix and restore original dimensions */
        img.data = vet2mat(v_send, len_info[0], len_info[1]);
        
        /* Restore original image dimensions */
        img.data = (cplx**)realloc(img.data, len_info[4] * sizeof(cplx*));
        for (int i = 0; i < len_info[4]; i++) {
            img.data[i] = (cplx*)realloc(img.data[i], len_info[3] * sizeof(cplx));
        }
        
        img.width = len_info[3];
        img.height = len_info[4];
        
        pgm_write(img, "ifft.pgm", "");
        
        printf("Pure FFT Time: %.10lf\n", fft_end - fft_start);
        
        free(v_send);
        free(img.data);
    }

    free(v_revc);
    free(displacements);
    free(recvcounts);

    MPI_Finalize();
    return 0;
}
