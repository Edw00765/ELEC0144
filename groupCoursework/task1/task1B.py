import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create input ranges for training and testing
xTraining = np.arange(-1, 1.0001, 0.05)
xTest = np.arange(-0.97, 0.9301, 0.1)

# Generate noise and labels based on the cubic function to learn
np.random.seed(10) 
noiseSample = np.random.normal(0, 0.02, len(xTraining)) # Add Gaussian noise to training data
yTraining = 0.8 * xTraining**3 + 0.3 * xTraining**2 - 0.4 * xTraining + noiseSample
yTest = 0.8 * xTest**3 + 0.3 * xTest**2 - 0.4 * xTest # Clean data for testing

# Prepare our data and shuffle to ensure SGD works optimally
df = pd.DataFrame({'x': xTraining, 'y': yTraining})
df = df.sample(frac=1).reset_index(drop=True) 

# We then define the network architecture
# 1 Input node -> 3 Hidden nodes (tanh) -> 1 Output node (linear)
inputSize = 1
hiddenSize = 3
outputSize = 1

# Hyperparameters
learningRate = 0.01
epochs = 3000

# Initialize weights randomly and biases at zero
weights1 = np.random.randn(hiddenSize, inputSize) * 0.1
bias1 = np.zeros((hiddenSize, 1))
weights2 = np.random.randn(outputSize, hiddenSize) * 0.1
bias2 = np.zeros((outputSize, 1))

# We enter the SGD training loop
for epoch in range(epochs):
    epochLoss = 0
    for i in range(len(df)):
        # Reshape single sample for matrix multiplication
        xSample = np.array([[df.iloc[i]['x']]])
        yTarget = df.iloc[i]['y']

        # Layer 1: linear sum -> tanh activation
        weightedSum1 = np.dot(weights1, xSample) + bias1
        activatedVal1 = np.tanh(weightedSum1)

        # Layer 2: linear sum
        weightedSum2 = np.dot(weights2, activatedVal1) + bias2
        yPred = weightedSum2

        # Next, we calculate the loss and perform backpropogation.
        error = yPred[0][0] - yTarget
        epochLoss += 0.5 * (error ** 2)

        delta2 = error 
        # Formula: (weights2 transposed * delta2) * derivative of tanh(weightedSum1)
        # Derivative of tanh(z) is (1 - tanh(z)^2)
        delta1 = np.dot(weights2.T, delta2) * (1 - activatedVal1**2)

        # After finding all of the deltas, we update the weights and biases.
        weights2 -= learningRate * delta2 * activatedVal1.T
        bias2 -= learningRate * delta2
        weights1 -= learningRate * delta1 * xSample.T
        bias1 -= learningRate * delta1

    # Print progress every 200 epochs
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}: MSE Loss = {epochLoss / len(df)}")
    
# We define a helper function to help display the performance of the model.
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
    elif inputTitle == "Test Data":
        plt.scatter(xTest, yTest, color='black', marker='+', label=inputTitle)
    
    plt.plot(xInput, yPredictedTest, color='red', linewidth=2, label='NN Prediction')
    plt.title(f'1-3-1 Network Approximation - {inputTitle}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run the display function for both sets
displayPerformance(xInput=xTraining, inputTitle="Training Data")
displayPerformance(xInput=xTest, inputTitle="Test Data")