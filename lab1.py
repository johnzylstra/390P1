#!C:/Users/johnm/AppData/Local/Microsoft/WindowsApps/python.exe

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
#tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
#tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
#ALGORITHM = "custom_net"
ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)*.1
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)*.1

    # Activation function.
    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return np.exp(-x)/((np.exp(-x)+1)**2)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 10, minibatches = True, mbs = 100):
        
        for e in range(epochs):
            if (e%2) == 0:
              print("epoch: " + str(e))
            #batchGen = self.__batchGenerator(xVals, mbs)
            #batch = next(batchGen)
            for i in range(len(xVals)):
              x, y = xVals[i], yVals[i]
              l1, l2 = self.__forward(x)
              l2delta = (y-l2)*(l2*(1-l2))
              l1delta = np.dot(l2delta, self.W2.T)*(l1*(1-l1))
              self.W1 += np.reshape(x, (784,1)).dot(np.reshape(l1delta, (1,len(l1delta))))*self.lr
              self.W2 += np.reshape(l1, (len(l1),1)).dot(np.reshape(l2delta, (1, 10)))*self.lr
        pass

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        for i in range(len(layer2)):
          prediction = layer2[i].argmax(axis=-1)
          out = np.zeros(10)
          out[prediction] = 1
          layer2[i] = out
        return layer2



# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)

#Keras net
def buildNet():
    model = keras.Sequential()
    lossType = keras.losses.categorical_crossentropy
    model.add(keras.layers.Dense(512, activation = "relu"))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(10, activation = "softmax"))
    model.compile(optimizer = "adam", loss = lossType)
    return model

def trainNet(model, x, y, eps):
    model.fit(x,y,epochs = eps)
    return model

def runNet(model, x):
    preds = model.predict(x)
    for i in range(len(preds)):
      prediction = preds[i].argmax(axis=-1)
      out = np.zeros(10)
      out[prediction] = 1
      preds[i] = out
    return preds



#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = ((raw[0][0]/255,raw[0][1]), (raw[1][0]/255,raw[1][1]))            # Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain = np.reshape(xTrain, (60000, 784)) #Flattening train
    xTest = np.reshape(xTest, (10000, 784)) #Flattening test
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        print("Not yet implemented.")                   #TODO: Write code to build and train your custon neural net.

        nn = NeuralNetwork_2Layer(784, 10, 200, learningRate = 0.1)
        nn.train(xTrain, yTrain)
        return nn
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = buildNet()
        model = trainNet(model, xTrain, yTrain, 30)
        print("We'll see if this works.")                   #TODO: Write code to build and train your keras neural net.
        return model
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        print("Working on it.")                   #TODO: Write code to run your custon neural net.
        preds = model.predict(data)
        return preds
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = runNet(model, data)
        #print("Not yet implemented.")                   #TODO: Write code to run your keras neural net.
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    preds = m.predict(d[1][0])

    xTest, yTest = d[1]
    acc = 0
    F1 = np.zeros((11,11), dtype=int)
    F1scores = np.zeros(10)
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
        j = preds[i].argmax(axis=-1)
        k = yTest[i].argmax(axis=-1)
        F1[j][k] +=1

    F1[:,10] = F1.sum(axis=-1)
    F1[10] = F1.sum(axis=0)

    for i in range(10):
      precision = F1[i][i]/(F1[10][i])
      recall = F1[i][i]/(F1[i][10])
      F1scores[i] = 2*(precision*recall)/(precision+recall)

    accuracy = acc / preds.shape[0]


    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print("F1 Confusion Matrix:")
    print(F1)
    print("F1 Scores:")
    print(F1scores)
    print()



#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)

    return data, model




if __name__ == '__main__':
    d,m = main()
