/* 
 * File:   main.cpp
 * Author: dmitry
 *
 * Created on December 1, 2017, 3:25 PM
 */

#include "utils.h"
#include "network.h"

#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iostream>

//MNIST parameters
const string MNIST_TRAIN_IMAGES_PATH = "../data/train-images-idx3-ubyte"; 
const string MNIST_TRAIN_LABELS_PATH = "../data/train-labels-idx1-ubyte";
const string MNIST_TEST_IMAGES_PATH = "../data/t10k-images-idx3-ubyte"; 
const string MNIST_TEST_LABELS_PATH = "../data/t10k-labels-idx1-ubyte";

const int MNIST_ALL_NUMBER_IMAGES = 60000;
const int MNIST_IMAGE_SIZE = 28 * 28;

//Network parameters
const int NUMBER_TEST_IMAGES = 10000;
const Double CROSS_ERROR = 7.0e-3;
const Double LEARN_RATE = 8.0e-3;

const int MAX_NUMBER_EPOCHS = 20;
const int NUMBER_INPUT_NEORONS = MNIST_IMAGE_SIZE;
const int NUMBER_HIDDEN_NEORONS = 300;
const int NUMBER_OUTPUT_NEORONS = 10;

void readoutDataFromFile(string path, VectorOfDoubleVectors &vectorToPut)
{
    const Double COEFF_TO_NORM = 255.0;
    ifstream file(path.c_str(), ios_base::binary);
    if (!file.is_open()) {
        cout << "Not found file: " << path;
        exit(-1);
    }
    
    int tmpToSkip, numImages, numRows, numColumns;
    file.read((char*)&tmpToSkip, sizeof(tmpToSkip));
    file.read((char*)&numImages, sizeof(numImages));
    reverseBytes((Byte*)&numImages, sizeof(numImages));
    file.read((char*)&numRows, sizeof(numRows));
    reverseBytes((Byte*)&numRows, sizeof(numRows));
    file.read((char*)&numColumns, sizeof(numColumns));
    reverseBytes((Byte*)&numColumns, sizeof(numColumns));
    
    Byte tmp;
    for (int i = 0; i < numImages; i++) {
        DoubleVector tmpVector;
        for (int j = 0; j < numRows; j++) {
            for (int k = 0; k < numColumns; k++) {
                file.read((char*)&tmp, sizeof(Byte));
                tmpVector.push_back((Double)tmp / COEFF_TO_NORM);
            }
        }
        vectorToPut.push_back(tmpVector);
    }
    
    file.close();
}

void readoutLabelsFromFile(string path, DoubleVector &vectorToPut)
{
    ifstream file(path.c_str(), ios_base::binary);
    if (!file.is_open()) {
        cout << "Not found file: " << path;
        exit(-1);
    }
    
    int tmpToSkip, numImages;
    file.read((char*)&tmpToSkip, sizeof(tmpToSkip));
    file.read((char*)&numImages, sizeof(numImages));
    reverseBytes((Byte*)&numImages, sizeof(numImages));
    
    Byte tmp;
    for (int i = 0; i < numImages; i++) {
        file.read((char*)&tmp, sizeof(Byte));
        vectorToPut[i] = (Double)tmp;
    }
    
    file.close();
}

void prepareTrainingData(VectorOfDoubleVectors& trainingData) {
    readoutDataFromFile(MNIST_TRAIN_IMAGES_PATH, trainingData);

    DoubleVector trainingLabels(MNIST_ALL_NUMBER_IMAGES);
    readoutLabelsFromFile(MNIST_TRAIN_LABELS_PATH, trainingLabels);

    trainingData.resize(MNIST_ALL_NUMBER_IMAGES);
    for (int i = 0; i < MNIST_ALL_NUMBER_IMAGES; i++) {
        DoubleVector tmp(NUMBER_OUTPUT_NEORONS, 0.0);
        tmp[static_cast<int>(trainingLabels[i])] = 1.0;
        for (int j = 0; j < NUMBER_OUTPUT_NEORONS; j++) {
            trainingData[i].push_back(tmp[j]);
        }
    }
}

void prepareTestingData(VectorOfDoubleVectors& testingData) {
    readoutDataFromFile(MNIST_TEST_IMAGES_PATH, testingData);

    DoubleVector testingLabels(NUMBER_TEST_IMAGES);
    readoutLabelsFromFile(MNIST_TEST_LABELS_PATH, testingLabels);

    testingData.resize(NUMBER_TEST_IMAGES);
    for (int i = 0; i < NUMBER_TEST_IMAGES; i++) {
        DoubleVector tmp(NUMBER_OUTPUT_NEORONS, 0.0);
        tmp[static_cast<int>(testingLabels[i])] = 1.0;
        for (int j = 0; j < NUMBER_OUTPUT_NEORONS; j++) {
            testingData[i].push_back(tmp[j]);
        }
    }
}

int main(int argc, char** argv) {
    cout << "************************" << endl;
    cout << "*** BACK PROPOGATION ***" << endl;
    cout << "************************" << endl;
    VectorOfDoubleVectors trainingData;
    prepareTrainingData(trainingData);
    
    Network network(NUMBER_INPUT_NEORONS, NUMBER_HIDDEN_NEORONS, NUMBER_OUTPUT_NEORONS);
    network.train(trainingData, MAX_NUMBER_EPOCHS, LEARN_RATE, CROSS_ERROR);

    VectorOfDoubleVectors testingData;
    prepareTestingData(testingData);
   
    cout << "Training error = " << 1.0 - network.getAnswer(trainingData) <<
            ", testing error = " << 1.0 - network.getAnswer(testingData) << endl;

    return 0;
}
