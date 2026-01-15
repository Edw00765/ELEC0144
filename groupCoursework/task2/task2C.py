import numpy as np
import random
import csv
import matplotlib.pyplot as plt

# Helper function to display lossHistory
def displayLoss(lossHistory):
    xAxis = range(10, (len(lossHistory) + 1) * 10, 10)

    plt.plot(xAxis, lossHistory, '.', color='red')
    plt.grid(True)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

# Data processing
np.random.seed(10)
random.seed(10)

classificationMap = {
    'Iris-setosa':     [0.6, -0.6, -0.6],
    'Iris-versicolor': [-0.6, 0.6, -0.6],
    'Iris-virginica':  [-0.6, -0.6, 0.6]
}

dataByClassification = {}
with open('irisData.txt', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row:
            # Parse features (first 4 columns)
            features = [float(val) for val in row[:4]]
            
            # Parse label (last column)
            classification = row[-1]
            
            # Convert label to One-Hot Encoding
            if classification in classificationMap:
                oneHotTarget = classificationMap[classification]
                
                # Store tuple: ( [features], [one hot] )
                if classification not in dataByClassification:
                    dataByClassification[classification] = []
                dataByClassification[classification].append((features, oneHotTarget))


TrainingList = []
TestingList = []

for species in dataByClassification:
    data = dataByClassification[species]
    random.shuffle(data)
    splitPoint = int(0.7 * len(data))
    TrainingList.extend(data[:splitPoint])
    TestingList.extend(data[splitPoint:])

random.shuffle(TrainingList)
random.shuffle(TestingList)

# Architecture: 4 Inputs -> 5 Hidden -> 3 Hidden -> 3 Output
inputSize = 4
hidden1Size = 5
hidden2Size = 3
outputSize = 3

def adam(learningRate, epochs):
    xTraining = np.array([item[0] for item in TrainingList]).T
    yTraining = np.array([item[1] for item in TrainingList]).T

    xTest = np.array([item[0] for item in TestingList]).T
    yTest = np.array([item[1] for item in TestingList]).T
    minVal = xTraining.min(axis=1, keepdims=True)
    maxVal = xTraining.max(axis=1, keepdims=True)
    rangeVal = maxVal - minVal
    rangeVal[rangeVal == 0] = 1

    xTraining = (xTraining - minVal) / rangeVal
    xTest = (xTest - minVal) / rangeVal
    numSamples = xTraining.shape[1]

    # We normalize the data from 0 to 1, which will avoid dead neurons as tanh is sensitive to large input values
    minVal = xTraining.min(axis=1, keepdims=True)
    maxVal = xTraining.max(axis=1, keepdims=True)
    # Avoid division by zero
    rangeVal = maxVal - minVal
    rangeVal[rangeVal == 0] = 1
    # Initialize Weights
    w1 = np.random.randn(hidden1Size, inputSize) * 0.1
    b1 = np.zeros((hidden1Size, 1))
    w2 = np.random.randn(hidden2Size, hidden1Size) * 0.1
    b2 = np.zeros((hidden2Size, 1))
    w3 = np.random.randn(outputSize, hidden2Size) * 0.1
    b3 = np.zeros((outputSize, 1))

    # Adam Hyperparameters
    batchSize = 32
    beta1 = 0.9 
    beta2 = 0.999 
    epsilon = 1e-8
    
    # Initialize Adam Memory
    mW1, vW1 = np.zeros_like(w1), np.zeros_like(w1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(w2), np.zeros_like(w2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
    mW3, vW3 = np.zeros_like(w3), np.zeros_like(w3)
    mb3, vb3 = np.zeros_like(b3), np.zeros_like(b3)
    
    t = 0 

    def adamUpdate(param, m, v, g, t):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        mHat = m / (1 - beta1 ** t)
        vHat = v / (1 - beta2 ** t)
        newParam = param - learningRate * mHat / (np.sqrt(vHat) + epsilon)
        return newParam, m, v

    print(f"Adam Training")
    lossHistory = []
    for epoch in range(epochs):
        epochLoss = 0

        perm = np.random.permutation(numSamples)
        xShuffled = xTraining[:, perm]
        yShuffled = yTraining[:, perm]
        
        # We step through the data in jumps of 'batchSize'
        for i in range(0, numSamples, batchSize):
            t += 1
            
            # Create Batch
            xBatch = xShuffled[:, i : i + batchSize] 
            yBatch = yShuffled[:, i : i + batchSize]
            
            # Get actual size of this batch (last batch might be smaller)
            currentBatchSize = xBatch.shape[1]
            
            # Forward Pass
            # From input to hidden 1
            weightedSum1 = np.dot(w1, xBatch) + b1
            activatedSum1 = np.tanh(weightedSum1)
            
            # From hidden1 to hidden 2
            weightedSum2 = np.dot(w2, activatedSum1) + b2
            activatedSum2 = np.tanh(weightedSum2)
            
            # From hidden 2 to output
            weightedSum3 = np.dot(w3, activatedSum2) + b3
            yPred = np.tanh(weightedSum3)
            
            # Error Calculation
            epochLoss += np.sum((yBatch - yPred) ** 2)
            
            # Backpropogation, where we update the weights, biases, velocities, and momentum
            error = yPred - yBatch
            delta3 = error * (1 - yPred**2)
            
            delta2 = np.dot(w3.T, delta3) * (1 - activatedSum2**2)
            delta1 = np.dot(w2.T, delta2) * (1 - activatedSum1**2)
            
            gW3 = np.dot(delta3, activatedSum2.T) / currentBatchSize
            gb3 = np.sum(delta3, axis=1, keepdims=True) / currentBatchSize
            
            gW2 = np.dot(delta2, activatedSum1.T) / currentBatchSize
            gb2 = np.sum(delta2, axis=1, keepdims=True) / currentBatchSize
            
            gW1 = np.dot(delta1, xBatch.T) / currentBatchSize
            gb1 = np.sum(delta1, axis=1, keepdims=True) / currentBatchSize
            
            # Adam Updates
            w3, mW3, vW3 = adamUpdate(w3, mW3, vW3, gW3, t)
            b3, mb3, vb3 = adamUpdate(b3, mb3, vb3, gb3, t)
            w2, mW2, vW2 = adamUpdate(w2, mW2, vW2, gW2, t)
            b2, mb2, vb2 = adamUpdate(b2, mb2, vb2, gb2, t)
            w1, mW1, vW1 = adamUpdate(w1, mW1, vW1, gW1, t)
            b1, mb1, vb1 = adamUpdate(b1, mb1, vb1, gb1, t)

        if (epoch + 1) % 10 == 0:
            # Average loss per sample
            avgLoss = epochLoss / numSamples
            lossHistory.append(avgLoss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avgLoss:.4f}")

    
    # Forward pass on FULL Test Set
    weightedSum1 = np.dot(w1, xTest) + b1
    activatedSum1 = np.tanh(weightedSum1)
    
    weightedSum2 = np.dot(w2, activatedSum1) + b2
    activatedSum2 = np.tanh(weightedSum2)
    
    weightedSum3 = np.dot(w3, activatedSum2) + b3
    yPred = np.tanh(weightedSum3)
    
    predClassification = np.argmax(yPred, axis=0)
    trueClassification = np.argmax(yTest, axis=0)
    
    correct = np.sum(predClassification == trueClassification)
    accuracy = (correct / xTest.shape[1]) * 100
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{xTest.shape[1]} correct)")
    displayLoss(lossHistory)


def batch(learningRate, epochs):
    xTraining = np.array([item[0] for item in TrainingList])
    yTraining = np.array([item[1] for item in TrainingList])

    xTest = np.array([item[0] for item in TestingList])
    yTest = np.array([item[1] for item in TestingList])

    # We normalize the data from 0 to 1, which will avoid dead neurons as tanh is sensitive to large input values
    minVal = xTraining.min(axis=0)
    maxVal = xTraining.max(axis=0)

    # Avoid division by zero
    rangeVal = maxVal - minVal
    rangeVal[rangeVal == 0] = 1

    xTraining = (xTraining - minVal) / rangeVal
    xTest = (xTest - minVal) / rangeVal
    numSamples = len(xTest)
    # Initialize Weights & Biases (Random small numbers)
    # Layer 1
    w1 = np.random.randn(hidden1Size, inputSize) * 0.1
    b1 = np.zeros((hidden1Size, 1))

    # Layer 2
    w2 = np.random.randn(hidden2Size, hidden1Size) * 0.1
    b2 = np.zeros((hidden2Size, 1))

    # Layer 3
    w3 = np.random.randn(outputSize, hidden2Size) * 0.1
    b3 = np.zeros((outputSize, 1))


    # Here we start training our model
    print("Batch Gradient DescentTraining")
    lossHistory = []
    for epoch in range(epochs):
        epochLoss = 0
        
        # Shuffle indices for SGD
        perm = np.random.permutation(len(xTraining))
        xTrainingShuffled = xTraining[perm]
        yTrainingShuffled = yTraining[perm]
        sumDeltaW3 = np.zeros_like(w3)
        sumDeltaB3 = np.zeros_like(b3)
        sumDeltaW2 = np.zeros_like(w2)
        sumDeltaB2 = np.zeros_like(b2)
        sumDeltaW1 = np.zeros_like(w1)
        sumDeltaB1 = np.zeros_like(b1)
        for i in range(len(xTraining)):
            x = xTrainingShuffled[i].reshape(-1, 1)
            y = yTrainingShuffled[i].reshape(-1, 1)
            
            # Layer 1
            weightedSum1 = np.dot(w1, x) + b1
            activatedSum1 = np.tanh(weightedSum1)
            
            # Layer 2
            weightedSum2 = np.dot(w2, activatedSum1) + b2
            activatedSum2 = np.tanh(weightedSum2)
            
            # Layer 3
            weightedSum3 = np.dot(w3, activatedSum2) + b3
            yPred = np.tanh(weightedSum3)
            
            epochLoss += np.sum((y - yPred) ** 2)
            
            # backpropogation, where we update the weights and bias from the error
            # We use the general formula (delta = error difference x activation derivative), where delta is differentiation of Error with respect to activatedVal of that layer
            error = yPred - y

            # Delta3 (Output Layer)
            delta3 = error * (1 - yPred**2)
            
            # Delta2 (Hidden Layer 2)
            delta2 = np.dot(w3.T, delta3) * (1 - activatedSum2**2)
            
            # Delta1 (Hidden Layer 1)
            delta1 = np.dot(w2.T, delta2) * (1 - activatedSum1**2)
            
            sumDeltaW3 += np.dot(delta3, activatedSum2.T)
            sumDeltaB3 += delta3
            sumDeltaW2 += np.dot(delta2, activatedSum1.T)
            sumDeltaB2 += delta2
            sumDeltaW1 += np.dot(delta1, x.T)
            sumDeltaB1 += delta1

        w3 -= learningRate * (sumDeltaW3 / numSamples)
        b3 -= learningRate * (sumDeltaB3 / numSamples)
        w2 -= learningRate * (sumDeltaW2 / numSamples)
        b2 -= learningRate * (sumDeltaB2 / numSamples)
        w1 -= learningRate * (sumDeltaW1 / numSamples)
        b1 -= learningRate * (sumDeltaB1 / numSamples)
        
        if (epoch + 1) % 10 == 0:
            avgLoss = epochLoss / len(xTraining)
            lossHistory.append(avgLoss)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avgLoss:.4f}")


    # Forward pass on FULL Test Set
    correct = 0
    for i in range(len(xTest)):
        x = xTest[i].reshape(-1, 1)
        target = yTest[i] # One-hot
        
        # Forward Pass only
        z1 = np.dot(w1, x) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = np.tanh(z2)
        z3 = np.dot(w3, a2) + b3
        yPred = np.tanh(z3)
        
        # Convert one-hot back to index (0, 1, or 2)
        predictedClassification = np.argmax(yPred)
        actualClassification = np.argmax(target)
        
        if predictedClassification == actualClassification:
            correct += 1

    accuracy = (correct / len(xTest)) * 100
    print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{len(xTest)} correct)")
    displayLoss(lossHistory)


adam(learningRate=0.001, epochs=1000)
batch(learningRate=0.001, epochs=1000)