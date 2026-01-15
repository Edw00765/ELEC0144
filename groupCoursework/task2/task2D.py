import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import csv
import random

# Helper function to display lossHistory
def displayLoss(lossHistory, title):
    plt.plot(lossHistory, '.', color='red')
    plt.grid(True)
    plt.title(f'Training Loss vs Epoch ({title})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

# Defining the training and testing data
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

# Shuffle final datasets so species aren't grouped together
random.shuffle(TrainingList)
random.shuffle(TestingList)

# Separate into Inputs (X) and Targets (Y) for Training Loop
# We convert them to NumPy arrays here to match your network code
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

# Helper function to run the model
def runExperiment(name, model, title):
    print(f"\n---Training: {name}---")
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    
    # Train
    history = model.fit(xTraining, yTraining, epochs=500, batch_size=16, verbose=0)
    lossHistory = history.history['loss']
    # Evaluate
    trainingLoss, trainingAccuracy = model.evaluate(xTraining, yTraining, verbose = 0)
    loss, accuracy = model.evaluate(xTest, yTest, verbose=0)
    print(f"Training Accuracy: {trainingAccuracy*100:.2f}%, MSE Loss: {trainingLoss}")
    print(f"Final Test Accuracy: {accuracy*100:.2f}%, MSE Loss: {loss}")
    displayLoss(lossHistory, title)


original = Sequential([
    Dense(5, input_dim=4, activation='tanh'),
    Dense(3, activation='tanh'),
    Dense(3, activation='tanh')
])

shallowWide = Sequential([
    Dense(10, input_dim=4, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(3, activation='tanh')
])

deepWide = Sequential([
    Dense(10, input_dim=4, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(10, activation='tanh'),
    Dense(3, activation='tanh')
])

modelRelu = Sequential([
    Dense(5, input_dim=4, activation='relu'),
    Dense(3, activation='relu'),
    Dense(3, activation='tanh')
])

sigmoidActivation = Sequential([
    Dense(5, input_dim=4, activation='sigmoid'),
    Dense(3, activation='sigmoid'),
    Dense(3, activation='sigmoid')
])

runExperiment("Original Architecture", original, 'Original Architecture')
runExperiment("Shallow and Wide, 10 Nodes in Each Hidden Layer", shallowWide, "10 Nodes in Each Hidden")
runExperiment("Deep and Wide, 10 Nodes in Each Hidden Layer, with 5 Hidden Layers", deepWide, "10 Nodes in Each Hidden, with 5 Hidden Layers")
runExperiment("Relu activation function in hidden layer", modelRelu, "ReLu Activation Function in Hidden Layer")

yTraining = (yTraining * 0.5) + 0.5
yTest = (yTest * 0.5) + 0.5
runExperiment("Sigmoid Activation Function in All Layers", sigmoidActivation, "Sigmoid Activation Function in All Layers")