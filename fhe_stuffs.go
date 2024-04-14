package main

import (
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func scaleMatrix(matrix [][]float64, scaleFactor float64, scaleUp bool) [][]float64 {
	/*
		ScaleMatrix scales the elements of a given 2D matrix by a scaleFactor.
		If scaleUp is true, elements are multiplied by scaleFactor, otherwise divided.
		return scaledMatrix
	*/
	rows := len(matrix)
	cols := len(matrix[0])

	scaledMatrix := make([][]float64, rows)
	for i := range scaledMatrix {
		scaledMatrix[i] = make([]float64, cols)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if scaleUp {
				scaledMatrix[i][j] = matrix[i][j] * scaleFactor
			} else {
				scaledMatrix[i][j] = matrix[i][j] / scaleFactor
			}
		}
	}

	return scaledMatrix
}

func fhe_mat_mult(client_values, server_values [][]float64, server_bias0 []float64) [][]float64 {
	var err error
	var params hefloat.Parameters

	// 128-bit secure parameters enabling depth-5 circuits.
	// LogN:14, LogQP: 431.
	if params, err = hefloat.NewParametersFromLiteral(
		hefloat.ParametersLiteral{
			LogN:            10,                        // log2(ring degree): Specifies the degree of the polynomial ring, where the degree is 2^LogN. A smaller value (10 in this case) can lead to more efficient operations but might reduce security.
			LogQ:            []int{50, 50, 50, 50, 50}, // log2(primes Q): Sets the bit-lengths of the primes forming the ciphertext modulus Q. Each prime has a bit-length of 50, configuring the modulus to handle ciphertext operations securely and efficiently.
			LogP:            []int{61},                 // log2(primes P): Defines the bit-length of the auxiliary primes used in operations such as relinearization, with a bit-length of 61 here for added security during these operations.
			LogDefaultScale: 45,                        // log2(scale): Determines the default scaling factor used in plaintext-ciphertext multiplications, balancing precision and noise growth in the ciphertext.
		}); err != nil {
		panic(err)
	}

	// Key Generator
	kgen := rlwe.NewKeyGenerator(params)

	// Secret Key
	sk := kgen.GenSecretKeyNew()

	// Secret Key
	pk := kgen.GenPublicKeyNew(sk)

	// Encoder
	ecd := hefloat.NewEncoder(params)

	// RelinearizationKey: an evaluation key which is used during ciphertext x ciphertext multiplication to ensure ciphertext compactness.
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Encryptor
	encr := rlwe.NewEncryptor(params, sk)
	encr0 := rlwe.NewEncryptor(params, pk)

	// Evaluator keys
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	// Evaluator
	eval := hefloat.NewEvaluator(params, evk)

	// Decryptor
	dec := rlwe.NewDecryptor(params, sk)

	scaleFactor := 100000.0
	client_values = scaleMatrix(client_values, scaleFactor, true)
	server_values = scaleMatrix(server_values, scaleFactor, true)

	server_bias := make([]float64, len(server_bias0))
	for i, value := range server_bias0 {
		server_bias[i] = value * scaleFactor
	}

	numRowsA := len(client_values)
	numRowsB := len(server_values)
	numColsA := len(client_values[0])
	numColsB := len(server_values[0])

	// Make plaintexts and encrypt them
	ctsA := make([][]*rlwe.Ciphertext, numRowsA)
	for i, row := range client_values { // This gives the number rows in a matrix
		ctsA[i] = make([]*rlwe.Ciphertext, len(row)) // Creates rows of ciphertexts
		for j, value := range row {                  // This gives the number columns in a slice
			ptA := hefloat.NewPlaintext(params, params.MaxLevel())   // Creates a specific plaintext value
			if err = ecd.Encode([]float64{value}, ptA); err != nil { // Encode the plaintext value into ciphertext
				panic(err)
			}
			if ctsA[i][j], err = encr.EncryptNew(ptA); err != nil { // Set the ciphertext value
				panic(err)
			}
		}
	}

	ctsB := make([][]*rlwe.Ciphertext, numRowsB)
	for i, row := range server_values {
		ctsB[i] = make([]*rlwe.Ciphertext, len(row))
		for j, value := range row {
			ptB := hefloat.NewPlaintext(params, params.MaxLevel())
			if err = ecd.Encode([]float64{value}, ptB); err != nil {
				panic(err)
			}
			if ctsB[i][j], err = encr0.EncryptNew(ptB); err != nil {
				panic(err)
			}
		}
	}

	// Create a slice to hold the ciphertexts for the bias terms
	c_bias := make([]*rlwe.Ciphertext, len(server_bias))
	for i, biasValue := range server_bias {
		// Create a new plaintext for each bias value
		pt_bias := hefloat.NewPlaintext(params, params.MaxLevel())
		// Encode the bias value into the plaintext
		if err = ecd.Encode([]float64{biasValue}, pt_bias); err != nil {
			panic(err)
		}
		// Encrypt the plaintext to obtain the ciphertext
		if c_bias[i], err = encr0.EncryptNew(pt_bias); err != nil {
			panic(err)
		}
	}

	result := make([][]*rlwe.Ciphertext, numRowsA)
	for i := 0; i < numRowsA; i++ {
		result[i] = make([]*rlwe.Ciphertext, numColsB)
		for j := 0; j < numColsB; j++ {
			sum := ctsA[i][0].CopyNew() // Accesses the first element (at index 0) of the i-th row
			for k := 0; k < numColsA; k++ {
				temp, _ := eval.MulRelinNew(ctsA[i][k], ctsB[k][j])
				//eval.Rescale(temp, temp)      //doesnt realy make that much of a diffrence
				//eval.Relinearize(temp, temp)  //doesnt realy make that much of a diffrence
				sum, _ = eval.AddNew(sum, temp)
				eval.Relinearize(sum, sum) //doesnt realy make that much of a diffrence
			}
			result[i][j], _ = eval.AddNew(sum, c_bias[j])
			//eval.Relinearize(result[i][j], result[i][j])
			result[i][j], _ = eval.MulRelinNew(result[i][j], result[i][j])
		}
	}

	plainResult := make([][]*rlwe.Plaintext, numRowsA)
	for i := 0; i < len(plainResult); i++ {
		plainResult[i] = make([]*rlwe.Plaintext, numColsB)
		for j := 0; j < numColsB; j++ {
			plainResult[i][j] = dec.DecryptNew(result[i][j])
		}
	}

	// Process result
	//fmt.Println("Final:")

	// Declare haveSlice as a 2D array
	haveSlice := make([][]float64, numRowsA)

	for i := 0; i < numRowsA; i++ {
		// Allocate memory for each row of haveSlice
		haveSlice[i] = make([]float64, numColsB)

		for j := 0; j < numColsB; j++ {
			if err = ecd.Decode(plainResult[i][j], haveSlice[i][j:]); err != nil {
				panic(err)
			}
		}

	}

	final_out := scaleMatrix(haveSlice, scaleFactor*scaleFactor*scaleFactor*scaleFactor, false)
	//fmt.Printf("%20.15f\n", final_out)
	return final_out

}
