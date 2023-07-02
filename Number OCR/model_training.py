import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

data = np.array(data)
m, n = data.shape

# shuffle before splitting into dev and training sets
np.random.shuffle(data) 

#Testing Set
data_dev = data[0:1000].T 
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

#Training Set
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

# Our NN will have complex three-layer architecture. Input layer A0 will have 784 units corresponding to
# the 784 pixels in each 28x28 input image. Two hidden layers A1,A2 will have 20 units each with ReLU 
# activation, and finally our output layer A3 will have 10 units corresponding to the ten digit classes
#  with softmax activation.

def init_params():                                                               #Initializing random parameters
    W1 = np.random.rand(20, 784) - 0.5
    b1 = np.random.rand(20, 1) - 0.5
    W2 = np.random.rand(20, 20) - 0.5
    b2 = np.random.rand(20, 1) - 0.5
    W3 = np.random.rand(10, 20) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):                                                                     #Rectified Linear activation Unit
    return np.maximum(Z, 0)

def softmax(Z):                                                                  #softmax activation function
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, W3, b3, X):                                     #Forward propagation
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z):                                                               #ReLU derivative
    return Z > 0

def one_hot(Y):                                                                  #one-hot encoded Label matrix for comapring the activation predicted and actual
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):                     #Backward propagation
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learn):  #updating parameters after back propagation
    W1 = W1 - learn * dW1
    b1 = b1 - learn * db1    
    W2 = W2 - learn * dW2  
    b2 = b2 - learn * db2   
    W3 = W3 - learn * dW3  
    b3 = b3 - learn * db3   
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):                                                         #Get  predictions from activation value(s)
    return np.argmax(A3,0)

def get_accuracy(predictions, Y):                                                #accuracy value
    print("Predictions: ",predictions,", Labels: ", Y)
    return np.sum(predictions == Y) / predictions.size

#***** Model Training Starts *****#

def gradient_descent(X, Y, learn, iterations): #Looping the process for every image to achieve suitable parameters to minimize the cost
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2 ,dW3, db3= backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2 , W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learn)
        
        if i % 1000 == 0:
            print("for Iterations between : ", i, " to ",i+1000)
            predictions = get_predictions(A3)
            accuracy = float(get_accuracy(predictions, Y))
            print("Accuracy = ",accuracy*100,"%")
    return W1, b1, W2, b2, W3, b3

W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.01, 100000)        #Training the model on 40,000 Images for 1,00,000 iterations

#***** Model Training Ends *****#

def predict(X, W1, b1, W2, b2, W3, b3):                                          #Predict the Image
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    return predictions


def test_prediction(index, W1, b1, W2, b2, W3, b3):                              #testing on different dataset
    current_image = X_train[:, index, None]
    prediction = predict(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)