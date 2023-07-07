from tensorflow import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os
import sys



    
# Load the data from CSV
# data = pd.read_csv('94_character_TMNIST.csv')

# Separate the features and labels
# x = data.drop(columns={'names','labels'}, axis=1).values
# y = data['labels'].unique()
# import csv

# with open('Labels.csv', 'w') as file:
#                 csv_writer = csv.writer(file)
#                 csv.DictWriter(file,fieldnames = ['Label']).writeheader()
                
#                 for label in y:
#                     csv_writer.writerow(label)
#                 file.close()



# Reading Labels
labels = pd.read_csv('Labels.csv')

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels.values.ravel())


# Load the saved model
loaded_model = keras.models.load_model('my_model.h5')

predicted_label = []


# Function to get images from IMAGES folder
def Predict(folder) :
    i = 0
    temp_predicted_label = []
    while True:
        
        # Load and preprocess the image for prediction
        image_path = folder + str(i) + ".png" 

        # End of Folder exception handling
        try:
            image = keras.preprocessing.image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')

        except FileNotFoundError:
            print('- End of Folder -')
            temp_predicted_label.append(" ")
            os.system('cls')
            break

        input_arr = keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        input_arr = input_arr.astype('float32') / 255.0
        print('File ',i+1, " is read.")

        # Make predictions using the loaded model
        predictions = loaded_model.predict(input_arr)
        

        # print(predictions)
        label_to_retrieve = np.argmax(predictions)
        temp_predicted_label.append(str(label_encoder.inverse_transform([label_to_retrieve])[0]))

        i+=1

    return temp_predicted_label
            


def get_folder(j):
    folder = 'Characters/' + str(j) + '/'
    return folder

j = 0
while True:
    
    folder = get_folder(j)
    if os.path.exists(folder):
        predicted_label.append(Predict(folder))
        j+=1

    else:
        print("Read the image successfully")
        break

    





# Print the predicted label
result = ''
for x in [val for sublist in predicted_label for val in sublist]:
    result += str(x)
print('Predicted label:', result)

