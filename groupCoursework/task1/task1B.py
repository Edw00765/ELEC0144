import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# We generate the training data and the testing data
xTraining = np.arange(-1, 1.0001, 0.05)
xTest = np.arange(-0.97, 0.9301, 0.1)

# Generate noise
np.random.seed(10) # Fixed seed chosen at random
noiseSample = np.random.normal(0, 0.02, len(xTraining))

yTraining = 0.8 * xTraining**3 + 0.3 * xTraining**2 - 0.4 * xTraining + noiseSample
yTest = 0.8 * xTest**3 + 0.3 * xTest**2 - 0.4 * xTest

df = pd.DataFrame({'x': xTraining, 'y': yTraining})
df = df.sample(frac=1).reset_index(drop=True) # We are shuffling it here because it is better SGD performs better with random sampling

# We define the layers
inputSize = 1
hiddenSize = 3
outputSize = 1

# Hyperparameters
learningRate = 0.01
epochs = 3000

# For the weights, we initialize the values as a random value using normal distribution
# The bias has been initially set at 0
weights1 = np.random.randn(hiddenSize, inputSize) * 0.1
bias1 = np.zeros((hiddenSize, 1))
weights2 = np.random.randn(outputSize, hiddenSize) * 0.1
bias2 = np.zeros((outputSize, 1))


for epoch in range(epochs):
    epochLoss = 0
    for i in range(len(df)):
        xSample = df.iloc[i]['x']
        xSample = np.array([[xSample]])
        yTarget = df.iloc[i]['y']

        # From input to hidden layer
        weightedSum1 = np.dot(weights1, xSample) + bias1
        activatedVal1 = np.tanh(weightedSum1)

        # From hidden layer to output layer
        weightedSum2 = np.dot(weights2, activatedVal1) + bias2
        yPred = weightedSum2

        # Error calculation
        error = yPred[0][0] - yTarget
        epochLoss += 0.5 * (error ** 2)

        # backpropogation, where we update the weights and bias from the error
        # We use the general formula (delta = error difference x activation derivative), where delta is differentiation of Error with respect to activatedVal of that layer
        delta2 = error
        delta1 = np.dot(weights2.T, delta2) * (1 - activatedVal1**2)

        weights2 -= learningRate * delta2 * activatedVal1.T
        bias2 -= learningRate * delta2
        weights1 -= learningRate * delta1 * xSample.T
        bias1 -= learningRate * delta1

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}: MSE Loss = {epochLoss / len(df)}")
    
# Display the performance of the model against the test and the training data
def displayPerformance(xInput, inputTitle):
    yPredictedTest = []
    for val in xInput:
        xPoint = np.array([[val]])
        h = np.tanh(np.dot(weights1, xPoint) + bias1)
        y = np.dot(weights2, h) + bias2
        yPredictedTest.append(y[0][0])

    plt.figure(figsize=(10, 6)) 

    if inputTitle == "Training Data":
        plt.scatter(xTraining, yTraining, color='black', marker='+', label=inputTitle)
        plt.plot(xInput, yPredictedTest, color='red', linewidth=2, label='NN Prediction')
        plt.title('1-3-1 Network Approximation (SGD)')
        plt.xlabel('x')
        plt.ylabel('d')
        plt.legend()
        plt.grid(True)
        plt.show()
    elif inputTitle == "Test Data":
        plt.scatter(xTest, yTest, color='black', marker='+', label=inputTitle)
        plt.plot(xInput, yPredictedTest, color='red', linewidth=2, label='NN Prediction')
        plt.title('1-3-1 Network Approximation (SGD)')
        plt.xlabel('x')
        plt.ylabel('d')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Invalid Input Title")

displayPerformance(xInput=xTraining, inputTitle="Training Data")
displayPerformance(xInput=xTest, inputTitle="Test Data")