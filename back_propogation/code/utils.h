/* 
 * File:   utils.h
 * Author: dmitry
 *
 * Created on December 1, 2017, 3:25 PM
 */

#ifndef UTILS_H
#define UTILS_H

#include "types.h"

void reverseBytes(Byte* bytes, int numBytes);
int getIndexOfMaximum(DoubleVector x);

Double hyperTan(Double x);
DoubleVector softmax(DoubleVector z);

void setRandomSeed(int seed);
Double rand01();
        
#endif /* UTILS_H */

