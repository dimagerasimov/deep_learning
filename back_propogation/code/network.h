/* 
 * File:   network.h
 * Author: dmitry
 *
 * Created on December 1, 2017, 3:25 PM
 */

#ifndef NETWORK
#define NETWORK

#include "types.h"

class Network {
public:
    Network(const int numberInputNeurons,
            const int numberHiddenNeurons, const int numberOutputNeurons);
    void train(const VectorOfDoubleVectors trainSample, const int maxNumberEpochs,
            const Double neededLearnRate, const Double neededCrossEntropyValue);
    Double getAnswer(const VectorOfDoubleVectors data);

private:
    void initializeOtherFields();
    void randomizeWeights();
    Double calcCrossEntropyValue(VectorOfDoubleVectors data);
    void backwardPass(DoubleVector ts, Double learnRate);
    void updateGradients(DoubleVector ts);
    void updateWeights(Double learnRate);
    void updateBiases(Double learnRate);
    DoubleVector calulcateHidOutputs(DoubleVector xs);
    DoubleVector calculateOutputs(DoubleVector xs);

private:
    int m_numberInputNeurons;
    int m_numberHiddenNeurons;
    int m_numberOutputNeurons;

    DoubleVector m_inputVector;
    DoubleVector m_outputVector;

    DoubleVector m_hiddenBiases;
    DoubleVector m_outputBiases;

    DoubleVector m_hiddenOutputs;

    DoubleVector m_hiddenGradients;
    DoubleVector m_outputGradients;

    VectorOfDoubleVectors m_weightsHiddenLayer;
    VectorOfDoubleVectors  m_weightsOutput;
};

#endif /* NETWORK */
