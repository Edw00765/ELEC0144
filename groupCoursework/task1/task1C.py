import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Here, we generate the training and test data as a global variable as it is used by both methods
xTraining = np.arange(-1, 1.0001, 0.05)
xTest = np.arange(-0.97, 0.9301, 0.1)

# Generate noise
np.random.seed(10) # Fixed seed chosen at random
noiseSample = np.random.normal(0, 0.02, len(xTraining))
yTraining = 0.8 * xTraining**3 + 0.3 * xTraining**2 - 0.4 * xTraining + noiseSample
yTest = 0.8 * xTest**3 + 0.3 * xTest**2 - 0.4 * xTest


# We define a helper function to display the performance of our model
def displayPerformance(xInput, inputTitle, weights1, bias1, weights2, bias2):
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
        plt.xlabel('x')
        plt.ylabel('d')
        plt.legend()
        plt.grid(True)
        plt.show()
    elif inputTitle == "Test Data":
        plt.scatter(xTest, yTest, color='black', marker='+', label=inputTitle)
        plt.plot(xInput, yPredictedTest, color='red', linewidth=2, label='NN Prediction')
        plt.xlabel('x')
        plt.ylabel('d')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Invalid Input Title")


# In this function, we define an ADAM training algorithm.
def adam(learningRate, epochs):
    df = pd.DataFrame({'x': xTraining, 'y': yTraining})
    df = df.sample(frac=1).reset_index(drop=True) # We are shuffling it here because it is better SGD performs better with random sampling

    # We define the layers
    inputSize = 1
    hiddenSize = 3
    outputSize = 1

    # For the weights, we make the values random between -0.5 to 0.5 at random.
    # The bias has been initially set at 0
    weights1 = np.random.uniform(-0.5, 0.5, (hiddenSize, inputSize))
    bias1 = np.zeros((hiddenSize, 1))
    weights2 = np.random.uniform(-0.5, 0.5, (outputSize, hiddenSize))
    bias2 = np.zeros((outputSize, 1))

    # Hyperparameters and weights
    batchSize = 32
    beta1 = 0.9 # The multiplier for the momentum
    beta2 = 0.999 # the multiplier for the mean of squared gradients / velocity
    epsilon = 1e-8
    t = 0 # The step counter

    # We define a helper function which updates the weight or matrix
    # param is the parameter that we are actually trying to optimize, m is the momentum (a average of gradients), v is the velocity (a average of the squared gradients), g is the gradient, t is the step counter
    def adamUpdate(param, m, v, g, t):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        mHat = m / (1 - beta1 ** t)
        vHat = v / (1 - beta2 ** t)
        newParam = param + learningRate * mHat / (np.sqrt(vHat) + epsilon) # We move in the direction of the momentum, but we divide it with sqrt of the velocity because if the velocity is large, we take smaller steps, while if the velocity is low, we can take larger steps.
        return newParam, m, v

    # Initialize Adam Memory, where m is the momentum and v is the velocity
    mW1, vW1 = np.zeros_like(weights1), np.zeros_like(weights1)
    mb1, vb1 = np.zeros_like(bias1), np.zeros_like(bias1)
    mW2, vW2 = np.zeros_like(weights2), np.zeros_like(weights2)
    mb2, vb2 = np.zeros_like(bias2), np.zeros_like(bias2)

    for epoch in range(epochs):
        epochLoss = 0
        for i in range(0, len(df), batchSize):
            t += 1
            dfBatch = df.iloc[i : i + batchSize]
            currentBatchSize = len(dfBatch) # Handle the case if the last batch is smaller
            xBatch = dfBatch['x'].values.reshape(1, currentBatchSize) 
            yTarget = dfBatch['y'].values.reshape(1, currentBatchSize)

            # From input to hidden layer
            weightedSum1 = np.dot(weights1, xBatch) + bias1
            activatedVal1 = np.tanh(weightedSum1)

            # From hidden layer to output layer
            weightedSum2 = np.dot(weights2, activatedVal1) + bias2
            yPred = weightedSum2

            # Error calculation
            error = yTarget - yPred
            epochLoss += np.sum(0.5 * (error ** 2))

            # backpropogation, where we update the weights, biases, velocities, and momentum
            delta2 = error
            gradientW2 = np.dot(delta2, activatedVal1.T) / currentBatchSize 
            gradientB2 = np.sum(delta2, axis=1, keepdims=True) / currentBatchSize

            delta1 = np.dot(weights2.T, delta2) * (1 - activatedVal1**2)
            gradientW1 = np.dot(delta1, xBatch.T) / currentBatchSize
            gradientB1 = np.sum(delta1, axis=1, keepdims=True) / currentBatchSize
            
            weights2, mW2, vW2 = adamUpdate(weights2, mW2, vW2, gradientW2, t)
            bias2, mb2, vb2 = adamUpdate(bias2, mb2, vb2, gradientB2, t)
            weights1, mW1, vW1 = adamUpdate(weights1, mW1, vW1, gradientW1, t)
            bias1, mb1, vb1 = adamUpdate(bias1, mb1, vb1, gradientB1, t)
            
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}: MSE Loss = {epochLoss / len(df)}")
        
    displayPerformance(xInput=xTraining, inputTitle="Training Data", weights1=weights1, bias1=bias1, weights2=weights2, bias2=bias2)
    displayPerformance(xInput=xTest, inputTitle="Test Data", weights1=weights1, bias1=bias1, weights2=weights2, bias2=bias2)


# We define the batch training method
def batch(learningRate, epochs):
    df = pd.DataFrame({'x': xTraining, 'y': yTraining})
    # Shuffling is not that important during batch, but it is kept in regardless
    df = df.sample(frac=1).reset_index(drop=True) 

    # We define the layers
    inputSize = 1
    hiddenSize = 3
    outputSize = 1

    # Initialize Weights
    weights1 = np.random.uniform(-0.5, 0.5, (hiddenSize, inputSize))
    bias1 = np.zeros((hiddenSize, 1))
    weights2 = np.random.uniform(-0.5, 0.5, (outputSize, hiddenSize))
    bias2 = np.zeros((outputSize, 1))

    n = len(df) 

    for epoch in range(epochs):
        epochLoss = 0
        
        # On each epoch, we initialize the sums of the deltas to update after every epoch
        sumDeltaW2 = np.zeros_like(weights2)
        sumDeltaB2 = np.zeros_like(bias2)
        sumDeltaW1 = np.zeros_like(weights1)
        sumDeltaB1 = np.zeros_like(bias1)
        
        for i in range(n):
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

            # Backpropogation
            # We use the general formula (delta = error difference x activation derivative), where delta is differentiation of Error with respect to activatedVal of that layer
            delta2 = error
            delta1 = np.dot(weights2.T, delta2) * (1 - activatedVal1**2)
            
            deltaW2 = delta2 * activatedVal1.T
            deltaB2 = delta2
            deltaW1 = delta1 * xSample.T
            deltaB1 = delta1
            
            sumDeltaW2 += deltaW2
            sumDeltaB2 += deltaB2
            sumDeltaW1 += deltaW1
            sumDeltaB1 += deltaB1

        # We update the weights and the bias per epoch, with the averages of each delta
        weights2 -= learningRate * (sumDeltaW2 / n)
        bias2 -= learningRate * (sumDeltaB2 / n)
        weights1 -= learningRate * (sumDeltaW1 / n)
        bias1 -= learningRate * (sumDeltaB1 / n)

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}: MSE Loss = {epochLoss / n}")

    displayPerformance(xInput=xTraining, inputTitle="Training Data", weights1=weights1, bias1=bias1, weights2=weights2, bias2=bias2)
    displayPerformance(xInput=xTest, inputTitle="Test Data", weights1=weights1, bias1=bias1, weights2=weights2, bias2=bias2)

# adam(learningRate=0.01, epochs=3000)
batch(learningRate=1, epochs=3000)
