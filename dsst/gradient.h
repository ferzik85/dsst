#pragma once

#ifndef GRADIENT_H
#define GRADIENT_H

#include "DSSTSettings.h"

void gradientMagnitude( float *I, float *M, float *O, int h, int w, int d, bool full );
void gradMagNormalization( float *M, float *S, int h, int w, float norm );
void gradientHist( float *M, float *O, float *H, int h, int w, int bin, int nOrients, int softBin, bool full );
void hog(float *M, float *O, float *H, int h, int w, int binSize, int nOrients, int softBin, bool full, float clip);
void fhog(float *M, float *O, float *H, int h, int w, int binSize, int nOrients, int softBin, float clip);
#endif