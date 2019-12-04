# Authors    : Ethan Brugger & Mitch Merrick
# Date       : 12/3/19
# Class      : AI CSCE 4613
# Assignment : Assignment 3

import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)


def sigmoid(val):
    return 1/(1 + np.exp(-val))


def sigmoidDerivative(val):
    return val * (1 - val)


# Define training data
trainInput = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expectedOuput = np.array([[0], [1], [1], [0]])

# Creating Test cases
testInput = np.array([[0, 0]])
xSpace = np.linspace(0, 1, 30)
ySpace = np.linspace(0, 1, 30)
for xP in xSpace:
    for yP in ySpace:
        if(xP == 0 and yP == 0):
            pass
        testInput = np.append(testInput, [[xP, yP]], axis=0)

# Define nn architecture
inputNodes, hiddenNodes, outputNodes = 2, 2, 1
epochs = 10000
learningRate = 0.1

# Initalize weights
# All random weights are 0 <= x <= 1
hiddenWeights = np.random.uniform(size=(inputNodes, hiddenNodes))
hiddenBias = np.random.uniform(size=(1, hiddenNodes))
outputWeights = np.random.uniform(size=(hiddenNodes, outputNodes))
outputBias = np.random.uniform(size=(1, outputNodes))


def forwardPropogation(inputData, hiddenWeights, hiddenBias, outputWeights, outputBias):
    # Calculate values of input dot weights for input->hidden
    # This section aka Forward Propogation
    hiddenLayerActivation = np.dot(inputData, hiddenWeights)
    hiddenLayerActivation += hiddenBias

    # Apply this value to activation function in this case sigmoid
    hiddenLayerOutput = sigmoid(hiddenLayerActivation)

    # Using the newly calculated activation function output do the same operation on hidden->output
    outputLayerActivation = np.dot(hiddenLayerOutput, outputWeights)
    outputLayerActivation += outputBias
    return [outputLayerActivation, hiddenLayerOutput]


def backpropogation(inputData, hiddenWeights, hiddenBias, outputWeights, outputBias, hiddenLayerOutput, predictedOutput):
    # Calculate the error between the expected and actual output
    # This section aka Backpropogation
    error = expectedOuput - predictedOutput

    # Using gradient decent to figure out what the output needs to change by
    deltaPredictedOutput = error * sigmoidDerivative(predictedOutput)

    errorHidenLayer = deltaPredictedOutput.dot(outputWeights.T)
    deltaHiddenLayer = errorHidenLayer * \
        sigmoidDerivative(hiddenLayerOutput)

    # Updating Weights and Biases
    outputWeights += hiddenLayerOutput.T.dot(
        deltaPredictedOutput) * learningRate
    outputBias += np.sum(deltaPredictedOutput, axis=0,
                         keepdims=True) * learningRate
    hiddenWeights += inputData.T.dot(deltaHiddenLayer) * learningRate
    hiddenBias += np.sum(deltaHiddenLayer, axis=0,
                         keepdims=True) * learningRate


def x_or_neural_network(inputData, hiddenWeights, hiddenBias, outputWeights, outputBias, epoch, training=True):
    for i in range(epoch):

        # Apply an actication function to the output of the nn
        forwardProp = forwardPropogation(
            inputData, hiddenWeights, hiddenBias, outputWeights, outputBias)
        hiddenLayerOutput = forwardProp[1]
        predictedOutput = sigmoid(forwardProp[0])

        if(training):
            # If training the network allow for back propogation
            backpropogation(inputData, hiddenWeights,
                            hiddenBias, outputWeights, outputBias, hiddenLayerOutput, predictedOutput)
    return predictedOutput


def setColor(testInput, testOutput):
    testX = testInput[:, [0]]
    testY = testInput[:, [1]]

    for i in range(len(testOutput)):
        val = testOutput[i]
        if val < .3:
            plt.scatter(testX[i], testY[i], color='lightGrey')
        elif val < .6:
            plt.scatter(testX[i], testY[i], color='grey')
        elif val < .9:
            plt.scatter(testX[i], testY[i], color='darkgrey')
        else:
            plt.scatter(testX[i], testY[i], color='black')


trainingOutput = x_or_neural_network(
    trainInput, hiddenWeights, hiddenBias, outputWeights, outputBias, epochs)
print(trainingOutput)


testOutput = x_or_neural_network(
    testInput, hiddenWeights, hiddenBias, outputWeights, outputBias, 1, False)

x = trainInput[:, [0]]
y = trainInput[:, [1]]

# Training Inputs are labeled with red triangles
# Test Inputs are the circles
setColor(testInput, testOutput)
plt.scatter(x, y, color='red', marker='^')
plt.ylabel('Training Inputs')
plt.show()
