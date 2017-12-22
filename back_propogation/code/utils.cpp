/* 
 * File:   utils.cpp
 * Author: dmitry
 *
 * Created on December 1, 2017, 3:25 PM
 */

#include "utils.h"
#include <cmath>
#include <cstdlib>

void reverseBytes(Byte* bytes, int numBytes)
{
    Byte tmp;
    for (int i = 0; i < numBytes / 2; i++) {
        tmp = bytes[i];
        bytes[i] = bytes[numBytes - i - 1];
        bytes[numBytes - i - 1] = tmp;
    }
}

int getIndexOfMaximum(DoubleVector x) {
    int resIndex = 0;
    Double max = x[resIndex];
    for (int i = 0; i < x.size(); i++) {
        if (x[i] > max) {
            resIndex = i;
            max = x[resIndex ];
        }
    }
    return resIndex;
}

Double hyperTan(Double x) {
    const Double TRESHOLD = 19.0;
    Double retValue;
    if (x < -TRESHOLD) {
        retValue = -1.0;
    }
    else if (x > TRESHOLD) {
        retValue = 1.0;
    }
    else {
        retValue = tanh(x);
    }
    return retValue;
}

DoubleVector softmax(DoubleVector z)  {
    Double max = z[0];
    for (int i = 0; i < z.size(); i++) {
        if (z[i] > max) {
            max = z[i];
        }
    }

    Double scale = 0.0;
    for (int i = 0; i < z.size(); i++) {
        scale += exp(z[i] - max);
    }

    DoubleVector result(z.size(), 0.0);
    for (int i = 0; i < z.size(); i++) {
        result[i] = exp(z[i] - max) / scale;
    }
    return result; 
}

Double rand01() {
    return rand() / Double(RAND_MAX);
}

void setRandomSeed(int seed) {
    srand(seed);
}