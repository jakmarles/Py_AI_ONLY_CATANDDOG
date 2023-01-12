from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('model')

# Load the image
# img = Image.open('PetImages/Cat/110.jpg')  here is the path to the image you want to predict
img = Image.open('cat.jpg') # here is the path to the image you want to predict
img = img.resize((150,150))
img = img.convert("L") # convert to grayscale
img = np.array(img)
img = img/255 # normalizing
img = np.expand_dims(img, axis=0)

# Make a prediction
prediction = model.predict(img)

# Print the prediction
if prediction[0] > 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")