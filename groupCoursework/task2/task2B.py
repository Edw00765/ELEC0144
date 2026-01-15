import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import csv


# After generating the input data, I want to update this so that the activation functions are using log sigmoid as the slides in week 4 said
# that this is a good choice. Default the training val to [0.8 on the correct one, 0.2 on the wrong one] when using log sigmoid,
# Use [0.6, -0.6, -0.6, -0.6] if using tanh


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


np.random.seed(10)  # Fixed seed chosen at random
random.seed(10)  # Fixed seed chosen at random

TrainingList = []
TestingList = []

for species in dataByClassification:
    data = dataByClassification[species]
    random.shuffle(data)
    
    # 70% split
    splitPoint = int(0.7 * len(data))
    TrainingList.extend(data[:splitPoint])
    TestingList.extend(data[splitPoint:])

random.shuffle(TrainingList)
random.shuffle(TestingList)

# Separate into Inputs (X) and Targets (Y) for Training Loop
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

# We define the layers here
# Architecture: 4 Inputs -> 5 Hidden -> 3 Hidden -> 3 Output
inputSize = 4
hidden1Size = 5
hidden2Size = 3
outputSize = 3

# Hyperparameters
learningRate = 0.001
epochs = 1000

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

# Helper Functions
def softmax(x):
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0)


# Here we start training our model
lossHistory = []

for epoch in range(epochs):
    epochLoss = 0
    
    # Shuffle indices for SGD
    perm = np.random.permutation(len(xTraining))
    xTrainingShuffled = xTraining[perm]
    yTrainingShuffled = yTraining[perm]
    
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
        
        # Here we update the weights
        w3 -= learningRate * np.dot(delta3, activatedSum2.T)
        b3 -= learningRate * delta3
        
        w2 -= learningRate * np.dot(delta2, activatedSum1.T)
        b2 -= learningRate * delta2
        
        w1 -= learningRate * np.dot(delta1, x.T)
        b1 -= learningRate * delta1

    # Calculate average loss for the epoch
    avgLoss = epochLoss / len(xTraining)
    lossHistory.append(avgLoss)
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avgLoss:.4f}")


# After training the model, we test it on our test data
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
    yPred = softmax(z3)
    
    # Convert one-hot back to index (0, 1, or 2)
    predictedClassification = np.argmax(yPred)
    actualClassification = np.argmax(target)
    
    if predictedClassification == actualClassification:
        correct += 1

accuracy = (correct / len(xTest)) * 100
print(f"Test Accuracy: {accuracy:.2f}% ({correct}/{len(xTest)} correct)")