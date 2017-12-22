/* 
 * File:   network.cpp
 * Author: dmitry
 *
 * Created on December 1, 2017, 3:25 PM
 */

#include "network.h"
#include "utils.h"

#include <iostream>
#include <cmath>
#include <algorithm>

Network::Network(const int numberInputNeurons, const int numberHiddenNeurons, const int numberOutputNeurons) {
    m_numberInputNeurons = numberInputNeurons;
    m_numberHiddenNeurons = numberHiddenNeurons;
    m_numberOutputNeurons = numberOutputNeurons;

    m_weightsHiddenLayer.resize(m_numberInputNeurons);
    for (int i = 0; i < m_weightsHiddenLayer.size(); i++) {
        m_weightsHiddenLayer[i].resize(m_numberHiddenNeurons);
    }
    m_weightsOutput.resize(m_numberHiddenNeurons);
    for (int i = 0; i < m_weightsOutput.size(); i++) {
        m_weightsOutput[i].resize(m_numberOutputNeurons);
    }

    initializeOtherFields();
    randomizeWeights();
}

void Network::train(const VectorOfDoubleVectors trainSample, const int maxNumberEpochs,
        const Double neededLearnRate, const Double neededCrossEntropyValue) {
    DoubleVector xs(m_numberInputNeurons, 0.0); 
    DoubleVector ts(m_numberOutputNeurons, 0.0); 

    vector<int> seq(trainSample.size(), 0);
    for (int i = 0; i < seq.size(); i++) {
        seq[i] = i;
    }

    for(int currentEpoch = 1; currentEpoch <= maxNumberEpochs; currentEpoch++) {
        cout << "Current epoch is " << currentEpoch << 
                " (max: " << maxNumberEpochs << ")" << endl;
        
        Double currentCrossEntropyValue = calcCrossEntropyValue(trainSample);
        cout << "Current cross entropy value is " << currentCrossEntropyValue <<
                " (needed " << neededCrossEntropyValue << ")" << endl;
        if (currentCrossEntropyValue < neededCrossEntropyValue) {
            cout << "Network is trained enough" << endl;
            return;
        }

        random_shuffle(seq.begin(), seq.end()); 
        for (int i = 0; i < trainSample.size(); i++) {
            for (int j = 0; j < xs.size(); j++) {
                xs[j] = trainSample[seq[i]][j];
            }
            for (int j = 0; j < ts.size(); j++) {
                ts[j] = trainSample[seq[i]][m_numberInputNeurons + j];
            }
            calculateOutputs(xs); 
            backwardPass(ts, neededLearnRate);
        }
    }
} 

void Network::initializeOtherFields() {
    m_inputVector.resize(m_numberInputNeurons, 0.0);
    m_hiddenBiases.resize(m_numberHiddenNeurons, 0.0);
    m_hiddenOutputs.resize(m_numberHiddenNeurons, 0.0);
    m_outputBiases.resize(m_numberOutputNeurons, 0.0);
    m_outputVector.resize(m_numberOutputNeurons, 0.0);
    m_hiddenGradients.resize(m_numberHiddenNeurons, 0.0);
    m_outputGradients.resize(m_numberOutputNeurons, 0.0);
}

Double Network::getAnswer(const VectorOfDoubleVectors data) {
    DoubleVector xs(m_numberInputNeurons, 0);
    DoubleVector ts(m_numberOutputNeurons, 0);
    DoubleVector ys;

    int numberValidAnswers, numberWrongAnswers;
    numberValidAnswers = numberWrongAnswers = 0;
    for (int i = 0; i < data.size(); ++i) {
        for (int j = 0; j < xs.size(); j++) {
            xs[j] = data[i][j];
        }
        for (int j = 0; j < ts.size(); j++) {
            ts[j] = data[i][m_numberInputNeurons + j];
        }
        ys = calculateOutputs(xs);

        if (ts[getIndexOfMaximum(ys)] == 1.0) {
            numberValidAnswers += 1;
        }
        else {
            numberWrongAnswers += 1;
        }
    }
    return (numberValidAnswers * 1.0) / (numberValidAnswers + numberWrongAnswers); 
}

void Network::randomizeWeights() {
    setRandomSeed(777);

    const Double DIVIDER = 111.0;
    for (int i = 0; i < m_numberInputNeurons; i++) {
        for (int j = 0; j < m_numberHiddenNeurons; j++) {
            m_weightsHiddenLayer[i][j] = rand01() / DIVIDER;
        }
    }
    for (int i = 0; i < m_numberHiddenNeurons; i++) {
        m_hiddenBiases[i] = rand01() / DIVIDER;
    }
    for (int i = 0; i < m_numberHiddenNeurons; i++) {
        for (int j = 0; j < m_numberOutputNeurons; j++) {
            m_weightsOutput[i][j] = rand01() / DIVIDER;
        }
    }
    for (int i = 0; i < m_numberOutputNeurons; i++) {
        m_outputBiases[i] = rand01() / DIVIDER;
    }
}

Double Network::calcCrossEntropyValue(VectorOfDoubleVectors data)
{
    Double err = 0.0;
    DoubleVector xs(m_numberInputNeurons, 0.0); 
    DoubleVector ts(m_numberOutputNeurons, 0.0); 

    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < xs.size(); j++) {
            xs[j] = data[i][j];
        }
        for (int j = 0; j < ts.size(); j++) {
            ts[j] = data[i][m_numberInputNeurons + j];
        }
        DoubleVector ys = calculateOutputs(xs); 
        for (int j = 0; j < m_numberOutputNeurons; j++) {
            err += log(ys[j]) * ts[j]; 
        }
    }
    return -1.0 * err / data.size();
}

void Network::backwardPass(DoubleVector ts, Double learnRate) {
    updateGradients(ts);
    updateWeights(learnRate);
    updateBiases(learnRate);
}

void Network::updateGradients(DoubleVector ts) {
    for (int i = 0; i < m_outputGradients.size(); i++) {
        m_outputGradients[i] = (ts[i] - m_outputVector[i]); 
    }
    Double x, sum;
    for (int i = 0; i < m_hiddenGradients.size(); i++) {
        sum = 0.0;
        for (int j = 0; j < m_numberOutputNeurons; j++) {
            x = m_outputGradients[j] * m_weightsOutput[i][j];
            sum += x;
        }
        m_hiddenGradients[i] = (1.0 - m_hiddenOutputs[i]) * (1.0 + m_hiddenOutputs[i]) * sum;
    }
}

void Network::updateWeights(Double learnRate) {
    Double delta;
    for (int i = 0; i < m_weightsHiddenLayer.size(); i++) {
        for (int j = 0; j < m_weightsHiddenLayer[0].size(); j++) {
            delta = learnRate * m_hiddenGradients[j] * m_inputVector[i];
            m_weightsHiddenLayer[i][j] += delta;
        }
    }
    for (int i = 0; i < m_weightsOutput.size(); i++) {
        for (int j = 0; j < m_weightsOutput[0].size(); j++) {
            delta = learnRate * m_outputGradients[j] * m_hiddenOutputs[i];
            m_weightsOutput[i][j] += delta;
        }
    }
}

void Network::updateBiases(Double learnRate) {
    Double delta;
    for (int i = 0; i < m_hiddenBiases.size(); i++) {
        delta = learnRate * m_hiddenGradients[i] * 1.0;
        m_hiddenBiases[i] += delta;
    }

    for (int i = 0; i < m_outputBiases.size(); i++) {
        delta = learnRate * m_outputGradients[i] * 1.0;
        m_outputBiases[i] += delta;
    }
}

DoubleVector Network::calulcateHidOutputs(DoubleVector xs) {
    DoubleVector hiddenOutput(m_numberHiddenNeurons, 0.0);
    for (int i = 0; i < xs.size(); i++) {
        m_inputVector[i] = xs[i];
    }
    for (int i = 0; i < m_numberHiddenNeurons; i++) {
        for (int j = 0; j < m_numberInputNeurons; j++) {
            hiddenOutput[i] += m_inputVector[j] * m_weightsHiddenLayer[j][i];
        }
    }
    for (int i = 0; i < m_numberHiddenNeurons; i++) {
        hiddenOutput[i] += m_hiddenBiases[i];
    }
    return hiddenOutput;
}

DoubleVector Network::calculateOutputs(DoubleVector xs) {
    DoubleVector normalOutput(m_numberOutputNeurons, 0.0);
    DoubleVector hiddenOutput = calulcateHidOutputs(xs);

    for (int i = 0; i < m_numberHiddenNeurons; i++) {
        m_hiddenOutputs[i] = hyperTan(hiddenOutput[i]);
    }
    for (int j = 0; j < m_numberOutputNeurons; j++) {
        for (int i = 0; i < m_numberHiddenNeurons; i++) {
            normalOutput[j] += m_hiddenOutputs[i] * m_weightsOutput[i][j];
        }
    }
    for (int i = 0; i < m_numberOutputNeurons; i++) {
        normalOutput[i] += m_outputBiases[i];
    }

    m_outputVector = softmax(normalOutput);
    return m_outputVector;
}



