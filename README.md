# Cena-Anomaly-Detection
Privacy Preserving Anomaly Detection using FHE

## Overview: 
This code uses the hefloat and RLWE modules in the Lattigo v5 library. The code demonstrates fully homomorphic encryption (FHE) operations, particularly focusing on the encryption, multiplication, and decryption of matrices. The operations are implemented within the context of securely processing data using homomorphic techniques, which allow computations on encrypted data without needing to decrypt it first. The example provided illustrates matrix multiplication between two encrypted matrices.

## Key Components:
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

## Usage and Modifications:
- The code is designed for educational and experimental purposes, showcasing how matrix operations can be securely executed on encrypted data.
- Users can modify the matrix dimensions, scaling factors, and FHE parameters based on the security requirements and computational resources available.
- The inclusion of performance timing helps in understanding the computational cost and could guide optimizations such as parameter tuning or parallelization.

## Security Considerations:
- The example sets specific parameters that determine the security and efficiency of the encryption. Adjusting these parameters can impact both the security level of the operations and the computational overhead involved.

## Conclusion:
- This sample code serves as a practical example of implementing secure, privacy-preserving computations using the Lattigo v5 library. It is suitable for developers and researchers interested in exploring advanced cryptographic techniques for secure data processing.
