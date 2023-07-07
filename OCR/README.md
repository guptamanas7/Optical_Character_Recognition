--- How to Use ---

go to cropImage.py file and insert the image path and run it, it automatically calls the other half of the program to call and predict the values.


--- Contents of this folder are as follows ---

Folder - Images
contains - sample images from where text needs to be extracted from

File - cropImage.py
contains - program to imiport a single image crop it to words then to characters and save it as .png in different folders

File - Labels.csv
contains - csv of the labels of identifyable characters

File - modeltraining.py 
contains - training algorithm of the cnn used for model

File - my_model.h5
contains - final trained model ready to use

File - OCR.py
contains - program to fetch the cropped images and feed it to the model to predict and save the output as a list

