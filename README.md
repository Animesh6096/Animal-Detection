# 🐾 Animal Detection Using Deep Learning

A comprehensive deep learning project for multi-class animal image classification using state-of-the-art Convolutional Neural Network (CNN) architectures.

## 📋 Project Overview

This project implements and compares three different deep learning architectures to classify images of five animal species:
- 🦏 Bighorn
- 🦋 Butterfly
- 🐫 Camel
- 🦍 Chimpanzee
- 🐷 Pig

## 🏗️ Architectures Implemented

### 1. AlexNet-inspired CNN
- Custom CNN architecture inspired by AlexNet
- Multiple convolutional and pooling layers
- Dense layers with 4096 neurons
- Trained for 50 epochs

### 2. ResNet-34
- 34-layer Residual Network
- Identity and convolutional blocks
- Skip connections to prevent vanishing gradient
- Trained for 20 epochs

### 3. InceptionV3
- Google's Inception architecture
- Multi-scale feature extraction
- Input size: 299×299×3
- Trained for 10 epochs

## 📊 Dataset

The dataset is organized into training and testing directories with the following structure:

```
├── train/
│   ├── bighorn/
│   ├── butterfly/
│   ├── camel/
│   ├── chimpanzee/
│   └── pig/
└── test/
    ├── bighorn/
    ├── butterfly/
    ├── camel/
    ├── chimpanzee/
    └── pig/
```

- **Image Format**: JPEG
- **Image Dimensions**: Resized to 224×224×3 (AlexNet, ResNet) or 299×299×3 (InceptionV3)
- **Classes**: 5 animal categories

## ✨ Key Features

### Data Preprocessing
- ✅ Grayscale to RGB conversion for consistency
- ✅ Image resizing to match model input requirements
- ✅ One-hot encoding for multi-class classification
- ✅ Data augmentation to handle class imbalance

### Data Augmentation
The project uses `ImageDataGenerator` to balance the dataset:
- Rotation (±10 degrees)
- Width and height shifting (10%)
- Shear transformation (10%)
- Zoom range (10%)
- Horizontal flipping

### Visualizations
- 📊 Class distribution plots
- 📈 Training/validation loss curves
- 📈 Training/validation accuracy curves
- 🖼️ Misclassified image analysis

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Detailed class-wise performance
- **F1 Score**: Weighted average for imbalanced classes

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow
pip install keras
pip install scikit-image
pip install scikit-learn
pip install matplotlib
pip install numpy
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Animal-Detection
```

2. Ensure your dataset is organized in the following structure:
```
Animal-Detection/
├── train/
│   ├── bighorn/
│   ├── butterfly/
│   ├── camel/
│   ├── chimpanzee/
│   └── pig/
└── test/
    ├── bighorn/
    ├── butterfly/
    ├── camel/
    ├── chimpanzee/
    └── pig/
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook milestone2.ipynb
```

## 📖 Usage

### Training Models

The notebook contains three main model implementations. Simply run the cells sequentially to:

1. **Load and preprocess data**
2. **Visualize class distributions**
3. **Apply data augmentation**
4. **Train AlexNet-inspired model**
5. **Train ResNet-34 model**
6. **Train InceptionV3 model**
7. **Evaluate and compare results**

### Model Evaluation

Each model section includes:
- Training history visualization
- Prediction on test set
- Misclassified image analysis
- Performance metrics (Accuracy, Confusion Matrix, F1 Score)

## 📈 Model Performance

The notebook generates detailed performance metrics including:

- **Training/Validation Curves**: Monitor model learning and detect overfitting
- **Confusion Matrix**: Understand class-wise prediction patterns
- **Misclassified Images**: Visual inspection of prediction errors
- **F1 Score**: Balanced performance metric for multi-class classification

## 🔍 Model Architecture Visualizations

The project generates PNG visualizations of each model architecture:
- `model1.png` - AlexNet-inspired CNN
- `model2.png` - ResNet-34
- `model3.png` - InceptionV3

## 📁 Project Structure

```
Animal-Detection/
├── milestone2.ipynb        # Main Jupyter notebook
├── README.md              # Project documentation
├── train/                 # Training images directory
├── test/                  # Testing images directory
├── model1.png            # AlexNet architecture diagram
├── model2.png            # ResNet-34 architecture diagram
└── model3.png            # InceptionV3 architecture diagram
```

## 🛠️ Technical Details

### Data Processing Pipeline
1. Load images from directories
2. Convert grayscale images to RGB
3. Apply data augmentation to minority classes
4. Resize images to target dimensions
5. Normalize pixel values
6. Convert labels to one-hot encoding

### Model Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Validation Split**: Separate test set

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is available for educational and research purposes.

## 👨‍💻 Author

Animesh

## 🙏 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- scikit-image for image processing utilities
- The research community for developing these architectures

## 📚 References

- **AlexNet**: Krizhevsky et al., "ImageNet Classification with Deep Convolutional Neural Networks"
- **ResNet**: He et al., "Deep Residual Learning for Image Recognition"
- **InceptionV3**: Szegedy et al., "Rethinking the Inception Architecture for Computer Vision"

---

⭐ If you found this project helpful, please consider giving it a star!
