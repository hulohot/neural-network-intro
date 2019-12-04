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


# Define training i/o
trainInput = np.array([[-2, 4], [-1, 4], [0, 4], [1, 4], [2, 4],
                       [-2, 3], [-1, 3], [0, 3], [1, 3], [2, 3],
                       [-2, 2], [-1, 2], [0, 2], [1, 2], [2, 2],
                       [-2, 1], [-1, 1], [0, 1], [1, 1], [2, 1],
                       [-2, 0], [-1, 0], [0, 0], [1, 0], [2, 0],
                       [-2, -1], [-1, -1], [0, -1], [1, -1], [2, -1],
                       [-2, -2], [-1, -2], [0, -2], [1, -2], [2, -2]])
expectedOutput = np.array([[1], [1], [1], [1], [1],
                           [-1], [1], [1], [1], [-1],
                           [-1], [1], [1], [1], [-1],
                           [-1], [1], [1], [1], [-1],
                           [-1], [-1], [1], [-1], [-1],
                           [-1], [-1], [-1], [-1], [-1],
                           [-1], [-1], [-1], [-1], [-1]])


# Creating Test cases
testInput = np.array([[0, 0]])
xSpace = np.linspace(-2, 2, 30)
ySpace = np.linspace(-2, 4, 30)
for xP in xSpace:
    for yP in ySpace:
        if(xP == 0 and yP == 0):
            pass
        testInput = np.append(testInput, [[xP, yP]], axis=0)

# Define nn architecture
inputNodes, hiddenNodes, outputNodes = 2, 5, 1
epochs = 10000
learningRate = 0.1

# Weight Convergence
weightConvergence = [100]

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
    error = expectedOutput - predictedOutput

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


def x_squared_neural_network(inputData, hiddenWeights, hiddenBias, outputWeights, outputBias, epoch, training=True):
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


def getTrainCorrect(trainOuput, realOutput):
    correct = 0
    for i in range(len(realOutput)):
        realOp = 1
        if(round(realOutput[i][0]) == 0):
            realOp = -1
        if trainOuput[i] == realOp:
            correct += 1
    return correct / len(realOutput)


def getTestCorrect(testInput, testOutput):
    correct = 0
    for i in range(len(testOutput)):
        if testInput[i][0] ** 2 >= testInput[i][1]:
            if round(testOutput[i][0]) == 0:
                correct += 1
        else:
            if round(testOutput[i][0]) == 1:
                correct += 1
    return correct / len(testOutput)

trainOutput = x_squared_neural_network(
    trainInput, hiddenWeights, hiddenBias, outputWeights, outputBias, epochs)

testOutput = x_squared_neural_network(
    testInput, hiddenWeights, hiddenBias, outputWeights, outputBias, 1, False)


print('Train')
print(getTrainCorrect(expectedOutput, trainOutput))

print('Test')
print(getTestCorrect(testInput, testOutput))

# This code shows the parabola y = x^2
x = np.linspace(-2, 2, 20)

# Training Inputs are labeled with red triangles
# Test Inputs are the circles
setColor(testInput, testOutput) # Uncomment this for test data
# setColor(trainInput, trainOutput) # Uncomment this for training data
plt.plot(x, x**2, color='red') # y = x ^ 2 line
plt.ylabel('Training Inputs')
plt.show()
