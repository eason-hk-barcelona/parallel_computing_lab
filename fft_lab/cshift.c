/**
 * @file cshift.c
 * @brief Circular shift functions for FFT operations
 *
 * This file implements circular shift functions used to center the
 * zero-frequency component in FFT results for better visualization
 * and analysis. Based on MATLAB's fftshift and ifftshift functions.
 *
 * Reference:
 * https://stackoverflow.com/questions/5915125/fftshift-ifftshift-c-c-source-code
 */

#include <complex.h>
#include <stdlib.h>

#include "pgm.h"

typedef double complex cplx;

/**
 * @brief Generic circular shift function for 2D complex arrays
 * @param in     Input complex array (flattened 2D array)
 * @param xdim   Width dimension
 * @param ydim   Height dimension
 * @param xshift Horizontal shift amount
 * @param yshift Vertical shift amount
 * @return Pointer to new shifted complex array
 */
static cplx* _circshift(const cplx* in, int xdim, int ydim, int xshift,
                        int yshift) {
    cplx* out = (cplx*)malloc(xdim * ydim * sizeof(cplx));

    for (int i = 0; i < xdim; i++) {
        int ii = (i + xshift) % xdim;
        for (int j = 0; j < ydim; j++) {
            int jj = (j + yshift) % ydim;
            out[ii * ydim + jj] = in[i * ydim + j];
        }
    }

    return out;
}

cplx* fftshift(cplx* in, int x, int y) {
    return _circshift(in, x, y, (x / 2), (y / 2));
}

cplx* ifftshift(cplx* in, int x, int y) {
    return _circshift(in, x, y, ((x + 1) / 2), ((y + 1) / 2));
}