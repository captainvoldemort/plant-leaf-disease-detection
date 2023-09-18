from tkinter import *
from tkinter import filedialog
from tkinter import ttk
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
    model_name = model_var.get()
    if model_name == 'VGG16':
        model_path = 'Trained_Model_vgg16.h5'
    elif model_name == 'ResNet50':
        model_path = 'Trained_Model_resnet50.h5'
    elif model_name == 'InceptionV3':
        model_path = 'Trained_Model_inceptionv3.h5'
    else:
        print("Invalid model selection!")
        return
    
    model = load_model(model_path)
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    predicted_class_name = class_names[predicted_class_index]
    prediction_label.config(text=f'Predicted Class: {predicted_class_name}\nConfidence: {confidence:.2f}')
    input_image_pil = Image.open(input_image_path)
    input_image_pil = input_image_pil.resize((400, 400), Image.Resampling.LANCZOS)
    input_image_tk = ImageTk.PhotoImage(input_image_pil)
    input_image_label.config(image=input_image_tk)
    input_image_label.image = input_image_tk

# Create the main window
root = Tk()
root.title("Plant Disease Detection")
root.configure(bg='#F9E9E7')  # Set background color

# Define class names
class_names = ['Scab','Rot','Rust','Healthy']

# Create UI elements
title_label = Label(root, text="MIT-WPU, School of ECE\nMinor in CSE PBL", font=("Helvetica", 18), bg='#F9E9E7')
title_label.pack(pady=10)

sub_heading_label = Label(root, text="Plant Leaf Disease Detection", font=("Helvetica", 14), bg='#F9E9E7')
sub_heading_label.pack(pady=5)

input_image_label = Label(root, bg='#F9E9E7')
input_image_label.pack()

style = ttk.Style()
style.configure('TButton', font=("Helvetica", 14), padding=5, width=20)
open_button = ttk.Button(root, text="Open Image", style='TButton', command=predict)
open_button.pack(pady=10)

prediction_label = Label(root, font=("Helvetica", 14), bg='#F9E9E7', wraplength=300)
prediction_label.pack()

sub_sub_heading_label = Label(root, text="Select Model Using Dropdown Below:", font=("Helvetica", 10), bg='#F9E9E7')
sub_sub_heading_label.pack(pady=1)

# Create dropdown for model selection with customized style
model_var = StringVar()
model_var.set('VGG16')  # Default model selection
style = ttk.Style()
style.configure('TCombobox', padding=10, font=('Helvetica', 14), width=20)  # Customize padding, font, and width
model_dropdown = ttk.Combobox(root, textvariable=model_var, values=['VGG16', 'ResNet50', 'InceptionV3'], state='readonly')
model_dropdown.pack(pady=5)

root.mainloop()
