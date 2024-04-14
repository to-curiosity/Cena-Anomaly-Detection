package main

/*
Data on cardiac Single Proton Emission Computed Tomography (SPECT) images. Each patient classified into two categories: normal and abnormal.
*/
import (
	"encoding/csv"
	"encoding/gob"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"
)

// Autoencoder represents the autoencoder model
type Autoencoder struct {
	InputSize  int
	HiddenSize int
	Weights1   [][]float64
	Weights2   [][]float64
	Biases1    []float64
	Biases2    []float64
}

// NewAutoencoder creates a new autoencoder model with random initialization
func NewAutoencoder(inputSize, hiddenSize int) *Autoencoder {
	return &Autoencoder{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Weights1:   initializeWeights(inputSize, hiddenSize),
		Weights2:   initializeWeights(hiddenSize, inputSize),
		Biases1:    initializeBiases(hiddenSize),
		Biases2:    initializeBiases(inputSize),
	}
}

// initializeWeights initializes the weights with random values
func initializeWeights(rows, cols int) [][]float64 {
	weights := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		weights[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			weights[i][j] = rand.NormFloat64() * 0.1
		}
	}
	return weights
}

// initializeBiases initializes the biases with random values
func initializeBiases(size int) []float64 {
	biases := make([]float64, size)
	for i := 0; i < size; i++ {
		biases[i] = rand.NormFloat64() * 0.1
	}
	return biases
}

// Approximate some activation function
func squared_activation_function(x float64) float64 {
	//return 1.0 / (1.0 + math.Exp(-x))  //sigmoid
	// return math.Max(0, x) ReLU
	return x * x
}

func sliceTo2D(slice []float64) [][]float64 {
	// Create a 2D slice with one row and len(slice) columns
	result := [][]float64{slice}
	return result
}

func flatten2D(slice [][]float64) []float64 {
	var flattened []float64

	// Iterate over each element of the row matrix and append it to the flattened slice
	for _, row := range slice {
		flattened = append(flattened, row...)
	}

	return flattened
}

// Forward performs the forward pass of the autoencoder
func (a *Autoencoder) Forward(input []float64) ([]float64, []float64) {
	encoded := make([]float64, a.HiddenSize)
	for i := 0; i < a.HiddenSize; i++ {
		sum := a.Biases1[i]
		for j := 0; j < a.InputSize; j++ {
			sum += a.Weights1[j][i] * input[j]
		}
		encoded[i] = squared_activation_function(sum)
	}

	decoded := make([]float64, a.InputSize)
	for i := 0; i < a.InputSize; i++ {
		sum := a.Biases2[i]
		for j := 0; j < a.HiddenSize; j++ {
			sum += a.Weights2[j][i] * encoded[j]
		}
		decoded[i] = squared_activation_function(sum)
	}

	return encoded, decoded
}

// Forward performs the forward pass of the autoencoder
func (a *Autoencoder) fhe_Forward(input0 []float64) []float64 {
	input := sliceTo2D(input0)

	// Encoding: Matrix multiplication using separate function
	encoded := fhe_mat_mult(input, a.Weights1, a.Biases1)

	// Decoding: Matrix multiplication using separate function
	decoded := fhe_mat_mult(encoded, a.Weights2, a.Biases2)

	flat_decoded := flatten2D(decoded)
	return flat_decoded
}

// mse calculates the mean squared error loss
func mse(original, decoded []float64) float64 {
	var sum float64
	for i := 0; i < len(original); i++ {
		diff := original[i] - decoded[i]
		sum += diff * diff
	}
	return sum / float64(len(original))
}

func (a *Autoencoder) Backward(input, encoded, decoded, gradOutput []float64) ([][]float64, []float64, [][]float64, []float64) {
	// Gradients for the decoded output
	gradDecoded := make([]float64, a.InputSize)
	for i := 0; i < a.InputSize; i++ {
		gradDecoded[i] = gradOutput[i] * 2 * decoded[i] //Use the derivative of the squared function (2*x)
	}

	// Gradients for the weights and biases of the decoding layer
	gradWeights2 := make([][]float64, a.HiddenSize)
	for i := 0; i < a.HiddenSize; i++ {
		gradWeights2[i] = make([]float64, a.InputSize)
		for j := 0; j < a.InputSize; j++ {
			gradWeights2[i][j] = gradDecoded[j] * encoded[i]
		}
	}
	gradBiases2 := make([]float64, a.InputSize)
	copy(gradBiases2, gradDecoded)

	// Gradients for the encoded output
	gradEncoded := make([]float64, a.HiddenSize)
	for i := 0; i < a.HiddenSize; i++ {
		var sum float64
		for j := 0; j < a.InputSize; j++ {
			sum += gradDecoded[j] * a.Weights2[i][j]
		}
		gradEncoded[i] = sum * 2 * encoded[i] // Use the derivative of the squared function (2*x)
	}

	// Gradients for the weights and biases of the encoding layer
	gradWeights1 := make([][]float64, a.InputSize)
	for i := 0; i < a.InputSize; i++ {
		gradWeights1[i] = make([]float64, a.HiddenSize)
		for j := 0; j < a.HiddenSize; j++ {
			gradWeights1[i][j] = gradEncoded[j] * input[i]
		}
	}
	gradBiases1 := make([]float64, a.HiddenSize)
	copy(gradBiases1, gradEncoded)

	return gradWeights1, gradBiases1, gradWeights2, gradBiases2
}

// DataSet represents a single data point
type DataSet struct {
	Features []float64
	Label    int
}

func main() {
	var start time.Time

	//rand.Seed(42)
	// Load train and validation datasets from CSV files
	trainData, err := loadDataFromCSV("./data/SPECTF_train.csv")
	if err != nil {
		log.Fatal(err)
	}
	validationData, err := loadDataFromCSV("./data/SPECTF_test.csv")
	if err != nil {
		log.Fatal(err)
	}

	// Normalize the input features
	normalizeData(trainData)
	normalizeData(validationData)

	// Create a new autoencoder
	autoencoder := NewAutoencoder(len(trainData[0].Features), 5)

	// Check if weights file exists
	weightsFile := "autoencoder_weights.gob"
	if _, err := os.Stat(weightsFile); err == nil {
		// Weights file exists, load the weights
		err = loadWeights(autoencoder, weightsFile)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("Loaded weights from file.")
	} else {
		numEpochs := 30000
		batchSize := 28
		learningRate := 0.00005

		var normalTrainData []DataSet
		for _, d := range trainData {
			if d.Label == 1 {
				normalTrainData = append(normalTrainData, d)
			}
		}

		// Train the autoencoder using only normal data
		for epoch := 0; epoch < numEpochs; epoch++ {
			shuffleData(normalTrainData)

			var batchLoss float64
			for i := 0; i < len(normalTrainData); i += batchSize {
				batchData := normalTrainData[i:min(i+batchSize, len(normalTrainData))]

				// Initialize gradients
				gradWeights1 := make([][]float64, autoencoder.InputSize)
				for i := range gradWeights1 {
					gradWeights1[i] = make([]float64, autoencoder.HiddenSize)
				}
				gradBiases1 := make([]float64, autoencoder.HiddenSize)
				gradWeights2 := make([][]float64, autoencoder.HiddenSize)
				for i := range gradWeights2 {
					gradWeights2[i] = make([]float64, autoencoder.InputSize)
				}
				gradBiases2 := make([]float64, autoencoder.InputSize)

				// Iterate over each data point in the batch
				for _, d := range batchData {
					// Forward pass
					encoded, decoded := autoencoder.Forward(d.Features)

					// Calculate loss
					loss := mse(d.Features, decoded)
					batchLoss += loss

					// Backward pass
					gradOutput := make([]float64, len(d.Features))
					for i := 0; i < len(d.Features); i++ {
						gradOutput[i] = 2 * (decoded[i] - d.Features[i]) / float64(len(d.Features))
					}
					gw1, gb1, gw2, gb2 := autoencoder.Backward(d.Features, encoded, decoded, gradOutput)

					// Accumulate gradients
					addGradients(gradWeights1, gw1)
					addGradients(gradWeights2, gw2)
					addSlices(gradBiases1, gb1)
					addSlices(gradBiases2, gb2)
				}

				// Normalize gradients by batch size
				scaleGradients(gradWeights1, 1.0/float64(len(batchData)))
				scaleGradients(gradWeights2, 1.0/float64(len(batchData)))
				scaleSlice(gradBiases1, 1.0/float64(len(batchData)))
				scaleSlice(gradBiases2, 1.0/float64(len(batchData)))

				// Update weights and biases
				updateParameters(autoencoder, gradWeights1, gradBiases1, gradWeights2, gradBiases2, learningRate)
			}
			// Calculate average batch loss
			avgBatchLoss := batchLoss / float64(len(normalTrainData)/batchSize)

			// Calculate validation loss
			var validationLoss float64
			for _, d := range validationData {
				_, decoded := autoencoder.Forward(d.Features)
				validationLoss += mse(d.Features, decoded)
			}
			avgValidationLoss := validationLoss / float64(len(validationData))

			fmt.Printf("Epoch %d - Batch Loss: %.4f - Validation Loss: %.4f\n", epoch+1, avgBatchLoss, avgValidationLoss)
		}

		// Save the weights after training
		err = saveWeights(autoencoder, weightsFile)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("Saved weights to file.")
	}

	// Set a threshold for anomaly detection, seems if
	threshold := .596

	// Evaluate the trained model on both normal and anomalous examples
	truePositive := 0
	trueNegative := 0
	falsePositive := 0
	falseNegative := 0

	totalNormal := 0
	totalAnomalous := 0

	for _, d := range trainData {
		_, decoded := autoencoder.Forward(d.Features)
		loss := mse(d.Features, decoded)

		if d.Label == 1 {
			totalNormal++
			if loss <= threshold {
				trueNegative++
			} else {
				falsePositive++
			}
		} else {
			totalAnomalous++
			if loss > threshold {
				truePositive++
			} else {
				falseNegative++
			}
		}
	}

	for _, d := range validationData {
		_, decoded := autoencoder.Forward(d.Features)
		loss := mse(d.Features, decoded)

		if d.Label == 1 {
			totalNormal++
			if loss <= threshold {
				trueNegative++
			} else {
				falsePositive++
			}
		} else {
			totalAnomalous++
			if loss > threshold {
				truePositive++
			} else {
				falseNegative++
			}
		}
	}
	// Evaluate the trained model on both normal and anomalous examples

	accuracy := float64(truePositive+trueNegative) / float64(truePositive+trueNegative+falsePositive+falseNegative) * 100
	normalAccuracy := float64(trueNegative) / float64(totalNormal) * 100
	anomalousAccuracy := float64(truePositive) / float64(totalAnomalous) * 100
	precision := float64(truePositive) / float64(truePositive+falsePositive)
	recall := float64(truePositive) / float64(truePositive+falseNegative)
	f1score := 2 * (precision * recall) / (precision + recall)
	falsePositiveRate := float64(falsePositive) / float64(falsePositive+trueNegative)
	falseNegativeRate := float64(falseNegative) / float64(falseNegative+truePositive)
	// https://towardsdatascience.com/anomaly-detection-how-to-tell-good-performance-from-bad-b57116d71a10

	fmt.Println("*********************************** Pre FHE performance ***********************************")
	fmt.Printf("Accuracy: %.2f%%\n", accuracy)
	fmt.Printf("Normal Accuracy: %.2f%%\n", normalAccuracy)
	fmt.Printf("Anomalous Accuracy: %.2f%%\n", anomalousAccuracy)
	fmt.Printf("Precision: %.4f\n", precision)
	fmt.Printf("Recall: %.4f\n", recall)
	fmt.Printf("F1 Score: %.4f\n", f1score)
	fmt.Printf("False Positive Rate: %.4f\n", falsePositiveRate) //Pick the system with the lowest possible False Positive rate
	fmt.Printf("False Negative Rate: %.4f\n", falseNegativeRate) //Choose the system with the lowest possible False Negatives rate.
	fmt.Println("*********************************** Pre FHE prediction ***********************************")
	// Make predictions here!
	input_for_prediction := validationData[50].Features
	start = time.Now()
	_, decoded := autoencoder.Forward(input_for_prediction)
	loss := mse(input_for_prediction, decoded)
	duration := time.Since(start)
	ms := duration.Seconds() * 1000
	fmt.Printf("PRE FHE Calc Done in %.6f ms\n", ms)
	fmt.Println()
	if loss < threshold {
		fmt.Println("The loss value is:", loss, "which is less than the threshold value of:", threshold)
		fmt.Println("Prediction of Normal Data")
	} else {
		fmt.Println("The loss value is:", loss, "which is more than the threshold value of:", threshold)
		fmt.Println("Prediction of Anomalous Data")
	}
	fmt.Println("*********************************** Post FHE prediction ***********************************")

	start = time.Now()
	// The prediction results here!

	get_fhe_prediction := autoencoder.fhe_Forward(input_for_prediction)
	loss = mse(input_for_prediction, get_fhe_prediction)
	duration2 := time.Since(start)
	ms2 := duration2.Seconds() * 1000
	fmt.Printf("POST FHE Calc Done in %.6f ms\n", ms2)
	fmt.Println()
	if loss < threshold {
		fmt.Println("The loss value is:", loss, "which is less than the threshold value of:", threshold)
		fmt.Println("Prediction of Normal Data")
	} else {
		fmt.Println("The loss value is:", loss, "which is more than the threshold value of:", threshold)
		fmt.Println("Prediction of Anomalous Data")
	}

	fmt.Println("*********************************** Diffrence in speed ***********************************")
	fmt.Println(ms2/ms, "times diff")
}

// saveWeights saves the weights of the autoencoder to a file
func saveWeights(a *Autoencoder, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(a.Weights1)
	if err != nil {
		return err
	}
	err = encoder.Encode(a.Weights2)
	if err != nil {
		return err
	}

	return nil
}

// loadWeights loads the weights of the autoencoder from a file
func loadWeights(a *Autoencoder, filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&a.Weights1)
	if err != nil {
		return err
	}
	err = decoder.Decode(&a.Weights2)
	if err != nil {
		return err
	}

	return nil
}

// normalizeData normalizes the input features to the range [0, 1]
func normalizeData(data []DataSet) {
	maxFeatures := make([]float64, len(data[0].Features))
	for _, d := range data {
		for j, val := range d.Features {
			if val > maxFeatures[j] {
				maxFeatures[j] = val
			}
		}
	}
	for i := 0; i < len(data); i++ {
		for j := 0; j < len(data[i].Features); j++ {
			data[i].Features[j] /= maxFeatures[j]
		}
	}
}

// shuffleData shuffles the order of the data points
func shuffleData(data []DataSet) {
	rand.Shuffle(len(data), func(i, j int) {
		data[i], data[j] = data[j], data[i]
	})
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// addGradients adds the gradients element-wise
func addGradients(dst, src [][]float64) {
	for i := 0; i < len(dst); i++ {
		for j := 0; j < len(dst[i]); j++ {
			dst[i][j] += src[i][j]
		}
	}
}

// addSlices adds the elements of two slices element-wise
func addSlices(dst, src []float64) {
	for i := 0; i < len(dst); i++ {
		dst[i] += src[i]
	}
}

// scaleSlice scales the elements of a slice by a factor
func scaleSlice(slice []float64, factor float64) {
	for i := 0; i < len(slice); i++ {
		slice[i] *= factor
	}
}

// scaleGradients scales the gradients by a factor
func scaleGradients(gradients [][]float64, factor float64) {
	for i := 0; i < len(gradients); i++ {
		for j := 0; j < len(gradients[i]); j++ {
			gradients[i][j] *= factor
		}
	}
}

func updateParameters(a *Autoencoder, gradWeights1 [][]float64, gradBiases1 []float64, gradWeights2 [][]float64, gradBiases2 []float64, learningRate float64) {
	// Update weights and biases of the encoding layer
	for i := 0; i < a.InputSize; i++ {
		for j := 0; j < a.HiddenSize; j++ {
			a.Weights1[i][j] -= learningRate * gradWeights1[i][j]
		}
	}
	for i := 0; i < a.HiddenSize; i++ {
		a.Biases1[i] -= learningRate * gradBiases1[i]
	}

	// Update weights and biases of the decoding layer
	for i := 0; i < a.HiddenSize; i++ {
		for j := 0; j < a.InputSize; j++ {
			a.Weights2[i][j] -= learningRate * gradWeights2[i][j]
		}
	}
	for i := 0; i < a.InputSize; i++ {
		a.Biases2[i] -= learningRate * gradBiases2[i]
	}
}

// loadDataFromCSV loads the dataset from a CSV file
func loadDataFromCSV(filename string) ([]DataSet, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	var data []DataSet
	for i, record := range records {
		if len(record) < 2 {
			return nil, fmt.Errorf("insufficient columns in record at line %d", i+1)
		}

		label, err := strconv.Atoi(record[0])
		if err != nil {
			return nil, fmt.Errorf("error parsing label value at line %d: %s", i+1, record[0])
		}

		var features []float64
		for j, val := range record[1:] {
			feature, err := strconv.ParseFloat(val, 64)
			if err != nil {
				fmt.Printf("Error parsing value at line %d, column %d: %s\n", i+1, j+2, val)
				return nil, err
			}
			features = append(features, feature)
		}

		data = append(data, DataSet{Features: features, Label: label})
	}

	return data, nil
}
