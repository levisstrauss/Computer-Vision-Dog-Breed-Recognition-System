# 🐕 Computer Vision Intelligence: Production-Ready Dog Breed Identification System

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Project Overview
This personal project demonstrates the implementation of a dog breed classification system using transfer learning with pre-trained CNN models. 
The project explores multiple deep learning architectures (VGG, ResNet, AlexNet) to compare their performance in identifying dog breeds from images.

## Key Project Goals:

- Implement and compare multiple CNN architectures for image classification
- Demonstrate transfer learning techniques with pre-trained models
- Build a complete image classification pipeline from preprocessing to results analysis
- Compare model performance across accuracy, precision, and computational efficiency
- Showcase practical Python implementation of computer vision concepts
  
## 🧠 Technical Approach

This project leverages transfer learning from three pre-trained convolutional neural network architectures:

## 📊 Model Comparison

| **Model**   | **Dog Detection** | **Breed Classification** | **Processing Speed** | **Model Size** |
|-------------|-------------------|---------------------------|----------------------|----------------|
| VGG         | 100%              | 93.3%                     | 1.7 sec/image        | 553 MB         |
| AlexNet     | 100%              | 80.0%                     | 0.9 sec/image        | 244 MB         |
| ResNet      | 90%               | 82.0%                     | 1.3 sec/image        | 102 MB         |

## Implementation Highlights:

1. Transfer Learning Application: Utilizes pre-trained weights from models trained on ImageNet
2. Two-Stage Classification: Performs both dog detection and breed identification
3. Performance Analysis: Comprehensive metrics calculation for model comparison
4. Modular Implementation: Clean, well-structured code with separation of concerns
5. Flexible Testing: Support for both batch processing and individual image classification

## 🛠️ Project Structure

```bash
dog_breed_identification/
├── pet_images/                  # Test images directory
├── uploaded_images/             # Directory for user uploaded images
├── dognames.txt                 # Reference file of valid dog names
├── check_images.py              # Main program
├── classifier.py                # CNN classifier implementation
├── get_input_args.py            # Command line argument handling
├── get_pet_labels.py            # Image label processing
├── classify_images.py           # Image classification logic
├── adjust_results4_isadog.py    # Dog validation
├── calculates_results_stats.py  # Statistics calculation
├── print_results.py             # Results output formatting
├── run_models_batch.sh          # Batch processing script for test images
└── run_models_batch_uploaded.sh # Batch processing for uploaded images
```
## 💻 Implementation Details

### Classification Pipeline
The project implements a complete image classification pipeline with the following components:

1. Image Preprocessing: Standardizing images for model input
2. Feature Extraction: Using pre-trained CNN architectures
3. Classification: Identifying dog breeds from extracted features
4. Validation: Checking classification against known labels
5. Performance Analysis: Calculating statistics and metrics

## Code Examples

### Command Line Argument Handling:

```python
def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these command line arguments.
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create command line arguments
    parser.add_argument('--dir', type=str, default='pet_images/', 
                        help='path to folder of images')
    parser.add_argument('--arch', type=str, default='vgg',
                        help='chosen model architecture (resnet, alexnet, vgg)')
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='file with dog names')
    
    # Return parsed arguments
    return parser.parse_args()
```
### Image Classification Logic:

```python
def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares pet labels to 
    the classifier labels, and adds the classifier label and the comparison of 
    the labels to the results dictionary.
    """
    results_dic = dict()
    
    # Process all files in the petlabels_dic
    for key in petlabel_dic:
        # Set path to current image
        image_path = images_dir + key
        
        # Run classifier function to classify the images
        model_label = classifier(image_path, model).lower().strip()
        
        # Extract pet label
        pet_label = petlabel_dic[key]
        
        # If pet image label is found within classifier label
        if pet_label in model_label:
            results_dic[key] = [pet_label, model_label, 1]
        else:
            results_dic[key] = [pet_label, model_label, 0]
    
    # Return results dictionary
    return results_dic
```

## 📊 Results and Analysis

The project compares the performance of three different CNN architectures:

### Key Findings:

- VGG: Achieved the highest breed classification accuracy (93.3%) but with the slowest processing time
- AlexNet: Fastest processing time with reasonable accuracy (80.0%)
- ResNet: Balanced performance with medium processing time and good accuracy (82.0%)
- Model Size: Significant differences in model size impact deployment considerations

## Performance Metrics Calculation:
```python
def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the program run using classifier's model 
    architecture to classify pet images.
    """
    # Initialize dictionary to hold results statistics
    results_stats_dic = dict()
    
    # Initialize counters for accurate classifications
    n_images = len(results_dic)
    n_correct_dogs = 0
    n_correct_notdogs = 0
    n_correct_breed = 0
    
    # Calculate statistics by iterating through results dictionary
    for key in results_dic:
        # Parse classification results
        if results_dic[key][3] == 1 and results_dic[key][4] == 1:
            n_correct_dogs += 1
            
            # If breed is correctly classified
            if results_dic[key][2] == 1:
                n_correct_breed += 1
                
        # If correctly classified as NOT a dog
        if results_dic[key][3] == 0 and results_dic[key][4] == 0:
            n_correct_notdogs += 1
    
    # Calculate percentages
    if results_stats_dic['n_dogs_img'] > 0:
        results_stats_dic['pct_correct_dogs'] = (n_correct_dogs / 
                                               results_stats_dic['n_dogs_img']) * 100.0
        results_stats_dic['pct_correct_breed'] = (n_correct_breed / 
                                                results_stats_dic['n_dogs_img']) * 100.0
    
    # Return statistics dictionary
    return results_stats_dic
```

### 🚀 Usage Instructions

## Batch Processing:

```bash
sh run_models_batch.sh
```
This executes classification with all three architectures:

```bash
python check_images.py --dir pet_images/ --arch resnet --dogfile dognames.txt > resnet_pet-images.txt
python check_images.py --dir pet_images/ --arch alexnet --dogfile dognames.txt > alexnet_pet-images.txt
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt > vgg_pet-images.txt
```
## Testing Custom Images:

```bash
# Add your images to uploaded_images/
sh run_models_batch_uploaded.sh
```
## ⚙️ Command Line Options

| **Parameter** | **Description**                                   | **Default**        |
|---------------|---------------------------------------------------|--------------------|
| `--dir`       | Directory containing images                       | `pet_images/`      |
| `--arch`      | CNN Model Architecture (`resnet`, `alexnet`, `vgg`)| `vgg`             |
| `--dogfile`   | File with valid dog names                         | `dognames.txt`     |

## 🔍 Key Learnings

This project provided valuable insights into several important aspects of deep learning and computer vision:

1.  Transfer Learning Benefits: Demonstrated how pre-trained models can be applied to new domains without extensive retraining
2. Architecture Comparison: Gained understanding of the tradeoffs between different CNN architectures
3. Image Classification Pipeline: Implemented a complete workflow from data ingestion to results analysis
4. Performance Optimization: Analyzed efficiency considerations for different models
5. Python Best Practices: Applied modular design, error handling, and clean coding principles

## 🛠️ Installation Requirements

```bash
# Clone the repository
git clone https://github.com/levisstrauss/Computer-Vision-Dog-Breed-Recognition-System.git
cd Dog-Breed-Classification-CNN-Transfer-Learning

# Install dependencies
pip install torch torchvision pillow argparse
```
## 🙏 Acknowledgments

- Amazon AWS - For providing the framework and computational resources
- Udacity - For educational content and project guidance
- PyTorch team - For the deep learning framework
- ImageNet dataset - Training data foundation for the pre-trained models
- Original CNN architecture developers (VGG, ResNet, AlexNet teams)

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
