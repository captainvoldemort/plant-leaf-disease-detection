from tkinter import *
from tkinter import filedialog
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np
from PIL import Image, ImageTk

# Function to load and preprocess the input image
def load_and_preprocess_image():
    input_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    input_image = load_img(input_image_path, target_size=(224, 224))
    input_image = img_to_array(input_image)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image / 255.0
    return input_image, input_image_path

# Function to make predictions and update the UI
def predict():
    input_image, input_image_path = load_and_preprocess_image()
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_class_name = class_names[predicted_class_index]
    prediction_label.config(text=f'Predicted Class: {predicted_class_name}\nConfidence: {confidence:.2f}')
    input_image_pil = Image.open(input_image_path)
    input_image_pil = input_image_pil.resize((400, 400), Image.ANTIALIAS)
    input_image_tk = ImageTk.PhotoImage(input_image_pil)
    input_image_label.config(image=input_image_tk)
    input_image_label.image = input_image_tk

# Create the main window
root = Tk()
root.title("Plant Disease Detection")
root.configure(bg='#F9E9E7')  # Set background color

# Load the trained model
model = load_model('Trained_Model_vgg16.h5')

# Define class names
class_names = ['Scab','Rot','Rust','Healthy'] 

# Create UI elements
title_label = Label(root, text="MIT-WPU, School of ECE\nDigital Image Processing PBL", font=("Helvetica", 18), bg='#F9E9E7')
title_label.pack(pady=10)

sub_heading_label = Label(root, text="Plant Leaf Disease Detection", font=("Helvetica", 14), bg='#F9E9E7')
sub_heading_label.pack(pady=5)

input_image_label = Label(root, bg='#F9E9E7')
input_image_label.pack()

open_button = Button(root, text="Open Image", font=("Helvetica", 14), bg='#FFA07A', command=predict)
open_button.pack(pady=10)

prediction_label = Label(root, font=("Helvetica", 14), bg='#F9E9E7', wraplength=300)
prediction_label.pack()

root.mainloop()

'''
Function descriptions:
load_and_preprocess_image(file_path): 
This function takes a file path as input, which represents the path of the image file selected by the user 
in the UI. It loads the image using the load_img() function from the PIL library, which is a Python Imaging 
Library that allows image manipulation. The loaded image is then converted to a NumPy array using the 
img_to_array() function from keras_preprocessing.image module. 
The image is then preprocessed by resizing it to the target size defined in the code (224x224 pixels in 
this case) and normalizing its pixel values to the range [0, 1]. Finally, the preprocessed image is returned.

predict(image, model): 
This function takes the preprocessed image and the trained model as inputs. It performs a prediction on the 
input image using the model.predict() method, which returns an array of predicted probabilities for each class. 
The class label with the highest predicted probability is determined using the argmax() function from NumPy, 
and the corresponding confidence value is extracted. The function returns the predicted class label and the 
confidence value as a tuple.
'''
