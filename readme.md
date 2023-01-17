
# Basic AI that can detect if there a dog or a cat in the picture uploaded

This program is a AI application that using the tensorflow trained model to classify the image as either a dog or a cat.

The program starts by importing the necessary libraries such as os, numpy, tensorflow, keras, PIL

Next, the program loads the previously trained model and sets up the data generator using the dataset path defined in the dataset_path variable.

The user submit an image to predict.py, This uses the PIL library to open the image file and resize it to 150x150 pixels. Then it converts the image to grayscale and normalizes it. Then it uses the loaded model to predict whether the image is a dog or a cat.
## Deployment


IN ORDER FOR IT TO WORK YOU WILL NEED python version 3.10.0
Create Virtual Environment.

```bash
pip install virtualenv
virtualenv -p python3.10.0 env
Activate: .\env\Scripts\activate

```
# Install dependencies.

```bash
pip install -r .\requirements.txt

```
download the model - due to its size you will have to download it yourself https://drive.google.com/file/d/1Sr7HghzbAOnAZkw8vzsqo1gLMyyIE7XN/view?usp=share_link
Make a new folder and name it model and another 2 folders in it named "variables" and "assets"
# Setting up the module
```bash
move "fingerprint.pb" and "keras_metadata.pb" and "saved_model.pb" into the folder "model"
```
```bash
move "variables.data-00000-of-00001" and "variables.index" into the folder "variables"

```
# If you want to remake the module you will need datasets 
## Setting up the datasets to remake the module
remove "module" folder
download the Kaggle Cats and Dogs Dataset - https://www.microsoft.com/en-us/download/details.aspx?id=54765
make a new folder and name it "PetImages"
# REMOVE THE FOLLOWING FILES THEY ARE CORRUPTED
cats/666.jpg
dogs/11702.jpg

# Run the application.
```bash
py predict.py
```


## Authors

- [@jakmarles](https://github.com/jakmarles) 

![Credits](https://img.shields.io/badge/Credits-Ilya%20Bronfman-green)
