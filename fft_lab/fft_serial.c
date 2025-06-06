/**
 * @file fft_serial.c
 * @brief Serial implementation of 2D Fast Fourier Transform
 *
 * This file implements a serial version of 2D FFT using the Cooley-Tukey
 * algorithm. It processes PGM images, applies zero-padding if necessary,
 * performs forward and inverse FFT, and outputs the results.
 */

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cshift.h"
#include "pgm.h"

#define PI 3.14159265358979323846

typedef double complex cplx;

cplx* mat2vet(cplx** mat, int width, int height) {
    cplx* v = (cplx*)malloc(height * width * sizeof(cplx));
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

int is_power_of_two(int x) { return (x != 0) && ((x & (x - 1)) == 0); }

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

    /* Copy the original image into the padded image */
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
 * @brief Cooley-Tukey FFT algorithm implementation
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

    /* Iterative FFT or IFFT computation */
    double sign = (inverse) ? 1.0 : -1.0; /* Sign for IFFT */
    for (int s = 1; s <= log2(N); s++) {
        int m = 1 << s; /* Subproblem size */
        cplx omega_m = cexp(sign * I * 2.0 * PI / m);

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
 * @brief Transpose a matrix represented as a flattened vector
 * @param v      Input vector (flattened matrix)
 * @param width  Number of columns
 * @param height Number of rows
 * @return Pointer to transposed vector
 */
cplx* transpose(cplx* v, int width, int height) {
    cplx* tmp = (cplx*)malloc(height * width * sizeof(cplx));

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            tmp[j * height + i] = v[i * width + j];
        }
    }

    return tmp;
}

/**
 * @brief Main function for serial 2D FFT processing
 * @param argc Argument count
 * @param argv Argument vector (argv[1] should be input image filename)
 * @return Exit status
 */
int main(int argc, char** argv) {
    pgm_t img;
    cplx* v_data;
    int o_width, o_height;

    /* Read input image */
    img = pgm_read(argv[1]);

    /* Store original dimensions */
    o_height = img.height;
    o_width = img.width;

    /* Apply zero padding if necessary */
    if (!is_power_of_two(img.width * img.height) || img.width != img.height) {
        img = zeroPadding(img);
    }

    /* Convert image matrix to vector format */
    v_data = mat2vet(img.data, img.width, img.height);

    clock_t fft_start = clock();

    /*=============== START 2D FFT ===============*/
    /* Perform 1D FFT on each row */
    for (int i = 0; i < img.height; i++) {
        cooley_tukey_fft(v_data + i * img.width, img.width, 0);
    }

    /* Transpose the data */
    v_data = transpose(v_data, img.width, img.height);

    /* Perform 1D FFT on each column (now row after transpose) */
    for (int i = 0; i < img.height; i++) {
        cooley_tukey_fft(v_data + i * img.width, img.width, 0);
    }
    /*=============== END 2D FFT ===============*/

    clock_t fft_end = clock();

    /* Convert back to matrix and apply FFT shift for visualization */
    img.data =
        vet2mat(fftshift(v_data, img.width, img.height), img.width, img.height);

    pgm_write_fft(img, "fft.pgm", "");

    /* Prepare for inverse FFT by applying inverse shift */
    v_data = ifftshift(mat2vet(img.data, img.width, img.height), img.width,
                       img.height);

    clock_t ifft_start = clock();

    /*=============== START 2D IFFT ===============*/
    /* Perform 1D IFFT on each row */
    for (int i = 0; i < img.height; i++) {
        cooley_tukey_fft(v_data + i * img.width, img.width, 1);
    }

    /* Transpose the data */
    v_data = transpose(v_data, img.width, img.height);

    /* Perform 1D IFFT on each column (now row after transpose) */
    for (int i = 0; i < img.height; i++) {
        cooley_tukey_fft(v_data + i * img.width, img.width, 1);
    }

    /* Normalize by dividing by total number of pixels */
    for (int i = 0; i < img.width * img.height; i++) {
        v_data[i] /= (img.height * img.width);
    }
    /*=============== END 2D IFFT ===============*/

    clock_t ifft_end = clock();

    /* Convert vector back to matrix */
    img.data = vet2mat(v_data, img.width, img.height);

    free(v_data);

    /* Restore original image dimensions */
    img.data = (cplx**)realloc(img.data, o_height * sizeof(cplx*));
    for (int i = 0; i < o_height; i++) {
        img.data[i] = (cplx*)realloc(img.data[i], o_width * sizeof(cplx));
    }

    img.width = o_width;
    img.height = o_height;

    pgm_write(img, "ifft.pgm", "");

    free(img.data);

    printf("Pure FFT Time: %.10lf\n",
           (double)(fft_end - fft_start) / CLOCKS_PER_SEC);
    printf("Pure IFFT Time: %.10lf\n",
           (double)(ifft_end - ifft_start) / CLOCKS_PER_SEC);

    return 0;
}