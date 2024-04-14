# Cena-Anomaly-Detection
Privacy Preserving Anomaly Detection using FHE

# Table of Contents
1. [Overview: anomalyDetection_autoEncoder.go](#overview-anomaly-detection-autoencoder)
2. [Key Components](#key-components-anomaly-detection-autoencoder)
   - [Data Loading](#data-loading)
   - [Data Preprocessing](#data-preprocessing)
   - [Autoencoder Model](#autoencoder-model)
   - [Training](#training)
   - [Anomaly Detection](#anomaly-detection)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Saving and Loading Weights](#saving-and-loading-weights)
3. [Usage and Modifications](#usage-and-modifications)
4. [Overview: fhe_stuffs.go](#overview-fhe-stuffs)
5. [Key Components](#key-components-fhe-stuffs)
   - [Parameter Initialization](#parameter-initialization)
   - [Key Generation](#key-generation)
   - [Encryption Process](#encryption-process)
   - [Homomorphic Operations](#homomorphic-operations)
   - [Decryption and Decoding](#decryption-and-decoding)
6. [Usage and Modifications](#usage-and-modifications-fhe-stuffs)
7. [Security Considerations](#security-considerations)
8. [Run Code](#run-code)


## Overview anomaly detection autoencoder: 
- The code implements an autoencoder-based anomaly detection system for cardiac Single Proton Emission Computed Tomography (SPECT) images. The autoencoder is trained using normal data and is then used to detect anomalies by comparing the reconstruction loss with a predefined threshold.

## Key Components anomaly detection autoencoder:
### Data Loading:
- The code loads the training and validation datasets from CSV files (SPECTF_train.csv and SPECTF_test.csv).
- Each data point consists of features and a label indicating whether it is normal (label=1) or anomalous (label=0).
### Data Preprocessing:
- The input features are normalized to the range [0, 1] using the normalizeData function.
- The data points are shuffled randomly during training using the shuffleData function.
### Autoencoder Model:
- The autoencoder model is defined in the Autoencoder struct, which consists of input size, hidden size, weights, and biases.
- The NewAutoencoder function creates a new autoencoder model with random initialization of weights and biases.
- The Forward function performs the forward pass of the autoencoder, encoding the input and then decoding it.
- The Backward function performs the backward pass, calculating gradients for weights and biases.
### Training:
- The autoencoder is trained using only normal data points.
- The training process iterates for a specified number of epochs and uses mini-batch gradient descent.
- The mean squared error (MSE) loss is used to measure the reconstruction error.
- The weights and biases are updated based on the calculated gradients and the learning rate.
### Anomaly Detection:
- After training, the autoencoder is used for anomaly detection.
- A threshold value is set to determine the boundary between normal and anomalous data points.
- The reconstruction loss (MSE) is calculated for each data point, and if it exceeds the threshold, the data point is classified as anomalous.
### Evaluation Metrics:
- The code calculates various evaluation metrics such as accuracy, precision, recall, F1 score, false positive rate, and false negative rate.
- These metrics provide insights into the performance of the anomaly detection system.
### Saving and Loading Weights:
- The trained weights of the autoencoder can be saved to a file using the saveWeights function.
- The weights can be loaded from a file using the loadWeights function, allowing the model to be reused without retraining.
## Usage and Modifications:
- Prepare the training and validation datasets in CSV format with features and labels.
- Set the appropriate file paths for the dataset files (SPECTF_train.csv and SPECTF_test.csv).
- Adjust the hyperparameters such as the number of epochs, batch size, learning rate, and anomaly detection threshold if needed.
- Run the code to train the autoencoder and perform anomaly detection on the validation dataset.
- The code will output the evaluation metrics and provide predictions for normal and anomalous data points.

## Overview fhe stuffs: 
This code uses the hefloat and RLWE modules in the Lattigo v5 library. The code demonstrates fully homomorphic encryption (FHE) operations, particularly focusing on the encryption, multiplication, and decryption of matrices. The operations are implemented within the context of securely processing data using homomorphic techniques, which allow computations on encrypted data without needing to decrypt it first. The example provided illustrates matrix multiplication between two encrypted matrices.

## Key Components fhe stuffs:
### Parameter Initialization:
- The code sets up FHE parameters tailored for a specific security level and operational efficiency, including parameters for the polynomial ring degree (LogN), ciphertext modulus (LogQ), auxiliary modulus (LogP), and the default scaling factor (LogDefaultScale).

### Key Generation:
- Keys necessary for the encryption process are generated, including a secret key (sk), a public key (pk), and a relinearization key (rlk), which is used to keep ciphertext sizes manageable during multiplication operations.

### Encryption Process:
- The code includes functions for encoding and encrypting two input matrices (client_values and server_values). Each element of the matrices is encoded into plaintexts and then encrypted into ciphertexts using the generated keys.

### Homomorphic Operations:
- Encrypted matrix multiplication is performed on the ciphertexts. The code demonstrates the use of addition and multiplication operations on encrypted data, managing complexity such as ciphertext size and noise growth through optional relinearization (commented out as not making much difference in this context).

### Decryption and Decoding:
- The result of the FHE operations is decrypted and decoded back into readable floating-point numbers. The code also includes scaling the resulting matrix to adjust for precision lost during encryption and multiplication.

## Usage and Modifications fhe stuffs:
- The code is designed for educational and experimental purposes, showcasing how matrix operations can be securely executed on encrypted data.
- Users can modify the matrix dimensions, scaling factors, and FHE parameters based on the security requirements and computational resources available.
- The inclusion of performance timing helps in understanding the computational cost and could guide optimizations such as parameter tuning or parallelization.

## Security Considerations:
- The example sets specific parameters that determine the security and efficiency of the encryption. Adjusting these parameters can impact both the security level of the operations and the computational overhead involved.

## Run code:
-* If the autoencoder_weights.gob file exists0, then the code will not retrain the model. Model weights and bias' within "autoencoder_weights.gob" will be used instead.
- Run command line --> go run anomalyDetection_autoEncoder.go fhe_stuffs.go
