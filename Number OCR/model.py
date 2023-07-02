#Neccessary imports
import pandas as pd
import numpy as np
import csv
import os
from PIL import Image

#Extracting the parameters of the trained model
W1 = pd.read_csv(r'Trained_Model\W1.csv')   
b1 = pd.read_csv(r'Trained_Model\b1.csv')
W2 = pd.read_csv(r'Trained_Model\W2.csv')
b2 = pd.read_csv(r'Trained_Model\b2.csv')
W3 = pd.read_csv(r'Trained_Model\W3.csv')
b3 = pd.read_csv(r'Trained_Model\b3.csv')

#Predicted Results and their correct values
predictions =  []
labels = []

def read_image(path):                                    #reading a single file

    try:
        temp = Image.open(path).convert('L')

    except FileNotFoundError:
        print('- End of Folder -')
        print("Predictions are : ", predictions)
        print("Labels are : ",labels)
        accuracy = sum([pred_i == label_i for pred_i, label_i in zip (predictions,labels)])*100/len(labels)
        print("Accuracy is : ","%.2f" %accuracy, "%")
        file = 'Input.csv'
        if(os.path.exists(file) and os.path.isfile(file)):
            os.remove(file)
        print("Temperoray files have been deleted")
        exit()

    return np.asarray(temp)

def flatten_list(l):                                     #flattens the list in a single row
  return [pixel/255 for sublist in l for pixel in sublist]  

def extract_features(X):                                 #creates a numerical list of image pixel by pixel 
  return [flatten_list(sample) for sample in X]

def filereader(foldername):                              #for reading all files and collecting the predictions
    i = 0
    while True:
        
        filename = foldername + str(i) + ".png" 

        test_img = [read_image(filename)]
        test_img = extract_features(test_img)

        print('File ',i+1, " is read.")
        
        with open('Input.csv', 'w') as f:
                csv_writer = csv.writer(f)
                csv.DictWriter(f,fieldnames = '0').writeheader()
                
                for sublist in test_img:
                    for j in range(len(sublist)):
                        # print(sublist[j])
                        csv_writer.writerow([sublist[j]])
        
        test_image = pd.read_csv(r'Input.csv')
        predictions.append(int(predict(test_image,W1,b1,W2,b2,W3,b3)))
        labels.append(i)
        i+=1

def predict(X, W1, b1, W2, b2, W3, b3):                  #Predicting the value of an image

    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    # current_image = current_image.reshape((28, 28)) * 255
    # plt.gray()
    # plt.imshow(current_image, interpolation='nearest')
    # plt.show()
    return predictions

def get_predictions(A3):                                 #Get predictions from activation value(s)
    return np.argmax(A3,0)

def ReLU(Z):                                             #Rectified Linear activation Unit
    return np.maximum(Z, 0)

def softmax(Z):                                          #softmax activation function
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, W3, b3, X):             #Forward propagation
    W1 = W1.values.tolist()
    X = X.values.tolist()
    
    Z1 = np.dot(W1,X) + b1.values.tolist()
    A1 = ReLU(Z1)
    
    W2 = W2.values.tolist()
    Z2 = np.dot(W2,A1) + b2.values.tolist()
    A2 = ReLU(Z2)

    W3 = W3.values.tolist()
    Z3 = np.dot(W3,A2) + b3.values.tolist()
    A3 = softmax(Z3)
    
    return Z1, A1, Z2, A2, Z3, A3

foldername = 'Images/'                                   #Assigning the foldername
filereader(foldername)                                   #Reading the files in a folder

