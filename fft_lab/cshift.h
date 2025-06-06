#ifndef C_SHIFT_H_
#define C_SHIFT_H_

#include <complex.h>

typedef double complex cplx;

cplx* fftshift(cplx* in, int x, int y);
cplx* ifftshift(cplx* in, int x, int y);

#endif