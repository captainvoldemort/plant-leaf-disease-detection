# **Plant Leaf Disease Detection Project**

## **Overview**

This project uses a VGG16-based model to classify apple plant leaf diseases. The model is trained on the Plant Village dataset. The project consists of two main files: **`GUI.py`** for testing the model using a graphical user interface and **`Train.py`** for training the model.

## **Prerequisites**

Ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- Pillow (PIL)
- tkinter

## **Usage**

### **Training the Model**

1. Organize your dataset into three folders: training, validation, and testing, and update the paths in **`Train.py`** accordingly:
    
    ```
    train_data_dir = 'Path_to_train_folder'
    validation_data_dir = 'Path_to_validation_folder'
    test_data_dir = 'Path_to_test_folder'
    
    ```
    
2. Run the training script:
    
    ```bash
    python Train.py
    
    ```
    
3. The trained model will be saved to the specified path.

### **Testing the Model**

1. Run the graphical user interface (GUI) using **`GUI.py`**:
    
    ```bash
    python GUI.py
    
    ```
    
2. Click the "Open Image" button to select an image for disease classification. The GUI will display the predicted class and confidence.

## **Model Architecture**

The model is based on the VGG16 architecture with custom classification layers. It is trained to classify images into four classes: Scab, Rot, Rust, and Healthy.

---

# About Model Training

This code is a Python script for performing transfer learning using the VGG16 architecture with the Keras deep learning library. Transfer learning involves using a pre-trained neural network model on a new task by fine-tuning it for a specific problem.

## **Libraries and Imports**

The code begins by importing necessary libraries and modules:

- **`os`**: The Python **`os`** module for operating system-related functionalities.
- **`ImageDataGenerator`**: A class from Keras used for data augmentation and preprocessing of image data.
- **`VGG16`**: A pre-trained deep neural network architecture.
- **`Dense`** and **`Flatten`**: Layers from Keras used to customize the network.
- **`Model`**: The Keras class for defining a neural network model.

## **Configuration and Data Paths**

Several configuration parameters and file paths are defined:

- **`num_classes`**: The number of output classes in the target problem (needs to be replaced with the actual number).
- **`train_data_dir`**: The path to the training data folder.
- **`test_data_dir`**: The path to the test data folder.
- **`validation_data_dir`**: The path to the validation data folder.
- **`save_path`**: The path where the trained VGG16 model will be saved.

## **Data Generators**

ImageDataGenerators are set up for training, validation, and testing data. These generators are responsible for loading and preprocessing image data, such as rescaling pixel values to a range between 0 and 1.

## **Model Definition**

The VGG16 model is loaded with pre-trained weights using **`VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))`**. Then, custom classification layers are added on top of the VGG16 base model:

- The output from the VGG16 base is flattened.
- A fully connected layer with 256 neurons and ReLU activation is added.
- The final output layer with a softmax activation function is defined with **`num_classes`** neurons.

## **Model Compilation**

The model is compiled with the following settings:

- Optimizer: 'adam'
- Loss function: 'categorical_crossentropy' (suitable for multi-class classification)
- Metrics to monitor during training: 'accuracy'

## **Model Training**

The model is trained using the training and validation data generators. It is trained for a specified number of epochs (in this case, 10).

## **Model Evaluation**

The trained model is evaluated on the test data using the **`model.evaluate()`** function. This provides information about the model's performance on unseen data.

## **Model Saving**

Finally, the trained model is saved to the specified **`save_path`**.
## **Project Structure**

- **`GUI.py`**: The graphical user interface for testing the model.
- **`Train.py`**: Script for training the model.
- **`Trained_Model_vgg16.h5`**: The trained model file.

## **Dataset**

The model is trained on the Plant Village dataset, which contains images of apple plant leaves with various diseases. In our project we have limited our study to four apple leaf classes. That are 'Scab','Rot','Rust' and 'Healthy'.

## Using Pretrained Models

You can choose the model that best suits your needs for disease classification. To use a different model, simply replace the model file in the GUI or use the respective model file when deploying this project for your own applications.

To specify the model when running the GUI or conducting batch predictions, update the `load_model` function call in `GUI.py` with the desired model file:

```python
# Load the desired model - NOT PROVIDED 
# For VGG16 model:
model = load_model('Trained_Model_vgg16.h5')
# For ResNet50 model:
model = load_model('Trained_Model_resnet50.h5')
# For InceptionV3 model:
model = load_model('Trained_Model_inceptionv3.h5')
```

## **Acknowledgments**

- **[Plant Village Dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)**
- **[VGG16 Model](https://arxiv.org/abs/1409.1556)**
