/**
 * @file pgm.c
 * @brief PGM image format I/O operations with complex number support
 *
 * This file implements functions for reading and writing PGM image files
 * with complex pixel values, including specialized functions for FFT output
 * with logarithmic scaling for better visualization.
 */

#include "pgm.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double complex cplx;

/**
 * @brief Apply logarithmic scaling to image data
 * @param img Input image structure
 * @return Image with logarithmically scaled pixel values
 */
pgm_t log_scale(pgm_t img) {
    double c = 255 / log(1 + img.max);

    for (int i = 0; i < img.height; i++) {
        for (int j = 0; j < img.width; j++) {
            img.data[i][j] = c * log(1 + cabs(img.data[i][j]));
        }
    }

    return img;
}

void pgm_write(pgm_t img, char *fabs, char *farg) {
    if (strcmp(farg, "") != 0) {
        /* Write both magnitude and phase files */
        FILE *fpabs = fopen(fabs, "wb");
        FILE *fparg = fopen(farg, "wb");

        if (fpabs == NULL || fparg == NULL) {
            printf("Error opening file\n");
            exit(1);
        }

        /* Write headers */
        fprintf(fpabs, "%s\n", img.type);
        fprintf(fparg, "%s\n", img.type);

        fprintf(fpabs, "%d %d\n", img.width, img.height);
        fprintf(fparg, "%d %d\n", img.width, img.height);

        fprintf(fpabs, "%d\n", img.max);
        fprintf(fparg, "%d\n", img.max);

        /* Write pixel data */
        for (int i = 0; i < img.height; i++) {
            for (int j = 0; j < img.width; j++) {
                if (isinf(cabs(img.data[i][j]))) {
                    printf("inf\n");
                    exit(1);
                }

                fprintf(fpabs, "%0.lf ", cabs(img.data[i][j]));
                fprintf(fparg, "%0.lf ", carg(img.data[i][j]));
            }
            fprintf(fpabs, "\n");
            fprintf(fparg, "\n");
        }

        fclose(fpabs);
        fclose(fparg);
    } else {
        /* Write magnitude file only */
        FILE *fpabs = fopen(fabs, "wb");

        if (fpabs == NULL) {
            printf("Error opening file\n");
            exit(1);
        }

        /* Write header */
        fprintf(fpabs, "%s\n", img.type);
        fprintf(fpabs, "%d %d\n", img.width, img.height);
        fprintf(fpabs, "%d\n", img.max);

        /* Write pixel data */
        for (int i = 0; i < img.height; i++) {
            for (int j = 0; j < img.width; j++) {
                if (isinf(cabs(img.data[i][j]))) {
                    printf("inf\n");
                    exit(1);
                }

                fprintf(fpabs, "%0.lf ", cabs(img.data[i][j]));
            }
            fprintf(fpabs, "\n");
        }

        fclose(fpabs);
    }
}

/**
 * @brief Write FFT result to PGM file with logarithmic scaling
 * @param img  PGM image structure containing FFT data
 * @param fabs Output filename for magnitude data
 * @param farg Output filename for phase data (empty string to skip phase
 * output)
 */
void pgm_write_fft(pgm_t img, char *fabs, char *farg) {
    if (strcmp(farg, "") != 0) {
        /* Write both magnitude and phase files */
        FILE *fpabs = fopen(fabs, "wb");
        FILE *fparg = fopen(farg, "wb");

        if (fpabs == NULL || fparg == NULL) {
            printf("Error opening file\n");
            exit(1);
        }

        /* Write headers */
        fprintf(fpabs, "%s\n", img.type);
        fprintf(fparg, "%s\n", img.type);

        fprintf(fpabs, "%d %d\n", img.width, img.height);
        fprintf(fparg, "%d %d\n", img.width, img.height);

        fprintf(fpabs, "%d\n", img.max);
        fprintf(fparg, "%d\n", img.max);

        /* Find maximum magnitude for logarithmic scaling */
        double img_max = 0;
        for (int i = 0; i < img.height; i++) {
            for (int j = 0; j < img.width; j++) {
                if (cabs(img.data[i][j]) > img_max) {
                    img_max = cabs(img.data[i][j]);
                }
            }
        }

        double c = 255 / log(1 + img_max);

        /* Write pixel data with logarithmic scaling */
        for (int i = 0; i < img.height; i++) {
            for (int j = 0; j < img.width; j++) {
                double img_tmp = c * log(1 + cabs(img.data[i][j]));

                fprintf(fpabs, "%0.lf ", img_tmp);
                fprintf(fparg, "%0.lf ", carg(img.data[i][j]));
            }
            fprintf(fpabs, "\n");
            fprintf(fparg, "\n");
        }

        fclose(fpabs);
        fclose(fparg);
    } else {
        /* Write magnitude file only */
        FILE *fpabs = fopen(fabs, "wb");

        if (fpabs == NULL) {
            printf("Error opening file\n");
            exit(1);
        }

        /* Write header */
        fprintf(fpabs, "%s\n", img.type);
        fprintf(fpabs, "%d %d\n", img.width, img.height);
        fprintf(fpabs, "%d\n", img.max);

        /* Find maximum magnitude for logarithmic scaling */
        double img_max = 0;
        for (int i = 0; i < img.height; i++) {
            for (int j = 0; j < img.width; j++) {
                if (cabs(img.data[i][j]) > img_max) {
                    img_max = cabs(img.data[i][j]);
                }
            }
        }

        double c = 255 / log(1 + img_max);

        /* Write pixel data with logarithmic scaling */
        for (int i = 0; i < img.height; i++) {
            for (int j = 0; j < img.width; j++) {
                double img_tmp = c * log(1 + cabs(img.data[i][j]));
                fprintf(fpabs, "%0.lf ", img_tmp);
            }
            fprintf(fpabs, "\n");
        }

        fclose(fpabs);
    }
}


pgm_t pgm_read(char *filename) {
    FILE *fp;
    pgm_t img;

    fp = fopen(filename, "rb");

    if (fp == NULL) {
        printf("Error opening file\n");
        exit(1);
    }

    char buff[2048];
    int tmp;

    /* Read PGM header */
    fgets(buff, sizeof(buff), fp);
    sscanf(buff, "%s", img.type);

    fgets(buff, sizeof(buff), fp);
    sscanf(buff, "%d %d", &img.width, &img.height);

    fgets(buff, sizeof(buff), fp);
    sscanf(buff, "%d", &img.max);

    /* Allocate memory for image data */
    img.data = (cplx **)malloc(img.height * sizeof(cplx *));

    /* Read pixel data */
    for (int i = 0; i < img.height; i++) {
        img.data[i] = (cplx *)malloc(img.width * sizeof(cplx));

        for (int j = 0; j < img.width; j++) {
            fscanf(fp, "%d", &tmp);
            img.data[i][j] = (cplx)tmp;
        }
    }

    fclose(fp);
    return img;
}