# Embedded-AI---Fashion-MNIST-Classification-with-MLP
This repository contains the implementation of a multilayer perceptron (MLP) for classifying the Fashion-MNIST dataset as part of the **Embedded AI in Systems** course at the University of Tehran. 
## Project Overview

### Objective
The goal of this exercise was to:
1. Design, train, and evaluate an MLP model for classifying Fashion-MNIST images.
2. Implement the model both using **PyTorch** and manually (without high-level libraries).
3. Compare the outputs, accuracy, and performance on **CPU** and **GPU**.

### Dataset: Fashion-MNIST
The Fashion-MNIST dataset consists of grayscale images (28x28 pixels) of clothing items, divided into:
- **Training set**: 60,000 images
- **Test set**: 10,000 images
The task is to classify each image into one of 10 categories, such as T-shirt, trousers, or shoes.

---

## Implementation Details

### 1. PyTorch-Based MLP

- **Architecture**:
  - Input Layer: 784 nodes (flattened 28x28 images)
  - Hidden Layer: 128 nodes with ReLU activation
  - Output Layer: 10 nodes (one for each class)
- **Training Configuration**:
  - Optimizer: Adam
  - Loss Function: CrossEntropyLoss
  - Epochs: 60
  - Test Accuracy: **88.88%**
- **Code Example**:
  ```python
  class MLP(nn.Module):
      def __init__(self):
          super(MLP, self).__init__()
          self.fc1 = nn.Linear(28 * 28, 128)
          self.fc2 = nn.Linear(128, 10)

      def forward(self, x):
          x = x.view(-1, 28 * 28)
          x = torch.relu(self.fc1(x))
          x = self.fc2(x)
          return x
   ```
### 2. Manual MLP Implementation
- **Process**:
  -Manual computation of matrix multiplications and ReLU activation.
  -Extracted weights and biases from the PyTorch model to ensure consistency.
- **Test Accuracy**: 88.88%
- **Code Example**:
  ```python
  def MyModel(x):
    x = x.view(-1, 28 * 28)
    x = relu(x @ weights_fc1.T + biases_fc1)
    x = x @ weights_fc2.T + biases_fc2
    return x
  ```
### 3. Comparing Outputs and Accuracy

- **Layer Outputs**:
  -Hidden and output layers of both implementations produce identical results.
- **Accuracy**:
  -Both models achieved 88.88% accuracy on test data.
---
### 4. Performance on CPU and GPU

- **Inference Times**:
  -PyTorch on CPU: 1.9224 seconds
  -PyTorch on GPU: 1.4352 seconds
  -Manual Implementation on CPU: 1.7467 seconds
  -Manual Implementation on GPU: 1.3453 seconds
---
### 5. Model Parameters

- **Both implementations had the same number of parameters**:
  -fc1.weight: 100,352
  -fc1.bias: 128
  -fc2.weight: 1,280
  -fc2.bias: 10
  -Total Parameters: 101,770
---
### Challenges and Solutions

- **Matrix Operations**: Addressed tensor shape mismatches by using PyTorch's tensor operations.
- **ReLU Function**: Replaced the custom NumPy-based ReLU with PyTorch's implementation due to compatibility issues.
---
### How to Run

- **Prerequisites**
  -Python 3.8+
  -PyTorch
  -NumPy
---
### Results and Insights

- **Accuracy**: Both models achieved identical test accuracy of 88.88%, demonstrating the correctness of the manual implementation.
- **Efficiency**: GPU processing reduced inference time significantly compared to CPU processing due to its parallelism capabilities.
- **Manual vs PyTorch**: While PyTorch simplifies implementation, manual implementation provides valuable insights into neural network operations.
---
###Author
Ali Ghorbani

