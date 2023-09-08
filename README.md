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

## **Project Structure**

- **`GUI.py`**: The graphical user interface for testing the model.
- **`Train.py`**: Script for training the model.
- **`Trained_Model_vgg16.h5`**: The trained model file.

## **Dataset**

The model is trained on the Plant Village dataset, which contains images of apple plant leaves with various diseases.

## Trained Models

In addition to the VGG16 model, we provide trained models using other architectures for your convenience:

- **VGG16 Model**: This is the default model trained for plant leaf disease detection.
- **ResNet50 Model**: A model based on the ResNet50 architecture.
- **InceptionV3 Model**: A model based on the InceptionV3 architecture.

You can choose the model that best suits your needs for disease classification. To use a different model, simply replace the model file in the GUI or use the respective model file when deploying this project for your own applications.

To specify the model when running the GUI or conducting batch predictions, update the `load_model` function call in `GUI.py` with the desired model file:

```python
# Load the desired model
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
