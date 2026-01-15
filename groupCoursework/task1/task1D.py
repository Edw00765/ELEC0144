import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Here, we generate the training and test data as a global variable as it is used by both methods
xTraining = np.arange(-1, 1.0001, 0.05)
xTest = np.arange(-0.97, 0.9301, 0.1)

# Generate noise
np.random.seed(10) # Fixed seed chosen at random
noiseSample = np.random.normal(0, 0.02, len(xTraining))
yTraining = 0.8 * xTraining**3 + 0.3 * xTraining**2 - 0.4 * xTraining + noiseSample
yTest = 0.8 * xTest**3 + 0.3 * xTest**2 - 0.4 * xTest

# We define the helper function to display the prediction of our models
def displayPerformance(xInput, yInput, inputTitle, model):
    # Predict using Keras
    # We reshape input to (N, 1) before feeding it to the model
    yPredicted = model.predict(xInput.reshape(-1, 1), verbose=0)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(xInput, yInput, color='black', marker='+', label=inputTitle)
    plt.plot(xInput, yPredicted, color='red', linewidth=2, label='Keras Prediction')
    plt.title(inputTitle)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

# We define our original SGD as a function:
def original(epochs, learningRate):
    keras.backend.clear_session()

    # Unlike our version in 1.b, we let keras initialize the starting weights instead of using a uniform distribution
    model = keras.Sequential([
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(3, input_shape=(1,), activation='tanh', bias_initializer='zeros'),
        
        # Output layer with one neuron and a linear activation function
        layers.Dense(1, activation='linear', bias_initializer='zeros')
    ])

    # We compile the model to use SGD where we define the learning rate as 0.01, and calculating the loss using MSE
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learningRate), loss='mean_squared_error')

    # Here, we train the model using SGD by defining the batch size as 1, and we train it at 5000 epochs.
    print("Starting training")
    history = model.fit(xTraining, yTraining, epochs=epochs, batch_size=1, verbose=0) 

    # Print final loss
    print(f"Final MSE Loss: {history.history['loss'][-1]:.6f}")

    displayPerformance(xTraining, yTraining, "Training Data", model)
    displayPerformance(xTest, yTest, "Test Data", model)

def reluHidden(epochs, learningRate):
    keras.backend.clear_session()

    model = keras.Sequential([
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(3, input_shape=(1,), activation='relu', bias_initializer='zeros'),
        
        # Output layer with one neuron and a linear activation function
        layers.Dense(1, activation='linear', bias_initializer='zeros')
    ])

    # We compile the model to use SGD where we define the learning rate as 0.01, and calculating the loss using MSE
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learningRate), loss='mean_squared_error')

    # Here, we train the model using SGD by defining the batch size as 1, and we train it at 5000 epochs.
    print("Starting training")
    history = model.fit(xTraining, yTraining, epochs=epochs, batch_size=1, verbose=0) 

    # Print final loss
    print(f"Final MSE Loss: {history.history['loss'][-1]:.6f}")

    displayPerformance(xTraining, yTraining, "Training Data", model)
    displayPerformance(xTest, yTest, "Test Data", model)

def leakyReLUOutput(epochs, learningRate):
    keras.backend.clear_session()

    # Unlike our original version, we let keras initialize the starting weights instead of using a uniform distribution
    model = keras.Sequential([
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(3, input_shape=(1,), activation="tanh", bias_initializer='zeros'),
        
        # Output layer with one neuron
        layers.Dense(1, bias_initializer='zeros'),

        # We define the alpha as 0.01, following the values found in week2's lecture
        layers.LeakyReLU(alpha=0.01)
    ])

    # We compile the model to use SGD where we define the learning rate as 0.01, and calculating the loss using MSE
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learningRate), loss='mean_squared_error')

    # Here, we train the model using SGD by defining the batch size as 1, and we train it at 5000 epochs.
    print("Starting training")
    history = model.fit(xTraining, yTraining, epochs=epochs, batch_size=1, verbose=0) 

    # Print final loss
    print(f"Final MSE Loss: {history.history['loss'][-1]:.6f}")

    displayPerformance(xTraining, yTraining, "Training Data", model)
    displayPerformance(xTest, yTest, "Test Data", model)

# We define our original SGD as a function:
def SGD5Hidden(epochs, learningRate):
    keras.backend.clear_session()

    model = keras.Sequential([
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(3, input_shape=(1,), activation='tanh', bias_initializer='zeros'),
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(3, activation='tanh', bias_initializer='zeros'),
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(3, activation='tanh', bias_initializer='zeros'),
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(3, activation='tanh', bias_initializer='zeros'),
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(3, activation='tanh', bias_initializer='zeros'),
        
        # Output layer with one neuron and a linear activation function
        layers.Dense(1, activation='linear', bias_initializer='zeros')
    ])

    # We compile the model to use SGD where we define the learning rate, and calculating the loss using MSE
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learningRate), loss='mean_squared_error')

    # Here, we train the model using SGD by defining the batch size as 1, and we train it at 5000 epochs.
    print("Starting training")
    history = model.fit(xTraining, yTraining, epochs=epochs, batch_size=1, verbose=0) 

    # Print final loss
    print(f"Final MSE Loss: {history.history['loss'][-1]:.6f}")

    displayPerformance(xTraining, yTraining, "Training Data", model)
    displayPerformance(xTest, yTest, "Test Data", model)



# We define our original SGD as a function:
def SGD10Node(epochs, learningRate):
    keras.backend.clear_session()

    model = keras.Sequential([
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(10, input_shape=(1,), activation='tanh', bias_initializer='zeros'),
        
        # Output layer with one neuron and a linear activation function
        layers.Dense(1, activation='linear', bias_initializer='zeros')
    ])

    # We compile the model to use SGD where we define the learning rate as 0.01, and calculating the loss using MSE
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learningRate), loss='mean_squared_error')

    # Here, we train the model using SGD by defining the batch size as 1, and we train it at 5000 epochs.
    print("Starting training")
    history = model.fit(xTraining, yTraining, epochs=epochs, batch_size=1, verbose=0) 

    # Print final loss
    print(f"Final MSE Loss: {history.history['loss'][-1]:.6f}")

    displayPerformance(xTraining, yTraining, "Training Data", model)
    displayPerformance(xTest, yTest, "Test Data", model)

def SGD10Node5Hidden(epochs, learningRate):
    keras.backend.clear_session()

    model = keras.Sequential([
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(10, input_shape=(1,), activation='tanh', bias_initializer='zeros'),

        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(10, activation='tanh', bias_initializer='zeros'),
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(10, activation='tanh', bias_initializer='zeros'),

        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(10, activation='tanh', bias_initializer='zeros'),
        # Hidden layer which contains 3 neurons with the tanh activation function
        layers.Dense(10, activation='tanh', bias_initializer='zeros'),
        
        # Output layer with one neuron and a linear activation function
        layers.Dense(1, activation='linear', bias_initializer='zeros')
    ])

    # We compile the model to use SGD where we define the learning rate, and calculating the loss using MSE
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=learningRate), loss='mean_squared_error')

    # Here, we train the model
    print("Starting training")
    history = model.fit(xTraining, yTraining, epochs=epochs, batch_size=1, verbose=0) 

    # Print final loss
    print(f"Final MSE Loss: {history.history['loss'][-1]:.6f}")

    displayPerformance(xTraining, yTraining, "Training Data", model)
    displayPerformance(xTest, yTest, "Test Data", model)


epochs = 5000
learningRate = 0.01
# original(epochs, learningRate)
# leakyReLUOutput(epochs, learningRate)
reluHidden(epochs, learningRate)
# SGD5Hidden(epochs, learningRate)
# SGD10Node(epochs, learningRate)
# SGD10Node5Hidden(epochs, learningRate)