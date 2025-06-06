#ifndef PGM_H_
#define PGM_H_

#include <complex.h>

typedef double complex cplx;

/**
 * @struct pgm
 * @brief Structure representing a PGM image with complex pixel data
 *
 * @var type    PGM format type identifier (e.g., "P2", "P5")
 * @var width   Image width in pixels
 * @var height  Image height in pixels
 * @var max     Maximum pixel value
 * @var data    2D array of complex pixel values
 */
typedef struct pgm {
    char type[3];
    int width;
    int height;
    int max;
    cplx **data;
} pgm_t;


void pgm_write(pgm_t img, char *fabs, char *farg);

/**
 * @brief Write FFT result to PGM file with logarithmic scaling
 * @param img    PGM image structure containing FFT data
 * @param fabs   Output filename for magnitude data
 * @param farg   Output filename for phase data (empty string to skip)
 */
void pgm_write_fft(pgm_t img, char *fabs, char *farg);

pgm_t pgm_read(char *filename);

#endif