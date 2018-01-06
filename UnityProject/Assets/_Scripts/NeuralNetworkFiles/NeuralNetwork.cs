using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;

namespace NeuralNetworks.Two
{
    public class NeuralNetwork
    {

        private static System.Random rnd;

        private int numInput;
        private int numHidden;
        private int numOutput;

        private double[] inputs;

        private double[][] inputHiddenWeights; // input-hidden
        private double[] hiddenBiases;
        private double[] hiddenOutputs;

        private double[][] hiddenOutputWeights; // hidden-output
        private double[] outputBiases;

        private double[] outputs;
        
        #region Back Propagation Specific things

        // back-prop specific arrays (these could be local to method UpdateWeights)
        private double[] outputGrads; // output gradients for back-propagation
        private double[] hiddenGrads; // hidden gradients for back-propagation

        // back-prop momentum specific arrays (could be local to method Train)
        private double[][] inputHiddenPrevWeightsDelta;  // for momentum with back-propagation
        private double[] hiddenPrevBiasesDelta;
        private double[][] hiddenOutputPrevWeightsDelta;
        private double[] outputPrevBiasesDelta;

        #endregion


        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            rnd = new System.Random(0); // for InitializeWeights() and Shuffle()

            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];

            this.inputHiddenWeights = MakeMatrix(numInput, numHidden);
            this.hiddenBiases = new double[numHidden];
            this.hiddenOutputs = new double[numHidden];

            this.hiddenOutputWeights = MakeMatrix(numHidden, numOutput);
            this.outputBiases = new double[numOutput];

            this.outputs = new double[numOutput];

            // back-prop related arrays below
            this.hiddenGrads = new double[numHidden];
            this.outputGrads = new double[numOutput];

            this.inputHiddenPrevWeightsDelta = MakeMatrix(numInput, numHidden);
            this.hiddenPrevBiasesDelta = new double[numHidden];
            this.hiddenOutputPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
            this.outputPrevBiasesDelta = new double[numOutput];
        } // ctor

        private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            return result;
        }

        public override string ToString() // yikes
        {
            string s = "";
            s += "===============================\n";
            s += "numInput = " + numInput + " numHidden = " + numHidden + " numOutput = " + numOutput + "\n\n";

            s += "inputs: \n";
            for (int i = 0; i < inputs.Length; ++i)
                s += inputs[i].ToString("F2") + " ";
            s += "\n\n";

            s += "ihWeights: \n";
            for (int i = 0; i < inputHiddenWeights.Length; ++i)
            {
                for (int j = 0; j < inputHiddenWeights[i].Length; ++j)
                {
                    s += inputHiddenWeights[i][j].ToString("F4") + " ";
                }
                s += "\n";
            }
            s += "\n";

            s += "hBiases: \n";
            for (int i = 0; i < hiddenBiases.Length; ++i)
                s += hiddenBiases[i].ToString("F4") + " ";
            s += "\n\n";

            s += "hOutputs: \n";
            for (int i = 0; i < hiddenOutputs.Length; ++i)
                s += hiddenOutputs[i].ToString("F4") + " ";
            s += "\n\n";

            s += "hoWeights: \n";
            for (int i = 0; i < hiddenOutputWeights.Length; ++i)
            {
                for (int j = 0; j < hiddenOutputWeights[i].Length; ++j)
                {
                    s += hiddenOutputWeights[i][j].ToString("F4") + " ";
                }
                s += "\n";
            }
            s += "\n";

            s += "oBiases: \n";
            for (int i = 0; i < outputBiases.Length; ++i)
                s += outputBiases[i].ToString("F4") + " ";
            s += "\n\n";

            s += "hGrads: \n";
            for (int i = 0; i < hiddenGrads.Length; ++i)
                s += hiddenGrads[i].ToString("F4") + " ";
            s += "\n\n";

            s += "oGrads: \n";
            for (int i = 0; i < outputGrads.Length; ++i)
                s += outputGrads[i].ToString("F4") + " ";
            s += "\n\n";

            s += "ihPrevWeightsDelta: \n";
            for (int i = 0; i < inputHiddenPrevWeightsDelta.Length; ++i)
            {
                for (int j = 0; j < inputHiddenPrevWeightsDelta[i].Length; ++j)
                {
                    s += inputHiddenPrevWeightsDelta[i][j].ToString("F4") + " ";
                }
                s += "\n";
            }
            s += "\n";

            s += "hPrevBiasesDelta: \n";
            for (int i = 0; i < hiddenPrevBiasesDelta.Length; ++i)
                s += hiddenPrevBiasesDelta[i].ToString("F4") + " ";
            s += "\n\n";

            s += "hoPrevWeightsDelta: \n";
            for (int i = 0; i < hiddenOutputPrevWeightsDelta.Length; ++i)
            {
                for (int j = 0; j < hiddenOutputPrevWeightsDelta[i].Length; ++j)
                {
                    s += hiddenOutputPrevWeightsDelta[i][j].ToString("F4") + " ";
                }
                s += "\n";
            }
            s += "\n";

            s += "oPrevBiasesDelta: \n";
            for (int i = 0; i < outputPrevBiasesDelta.Length; ++i)
                s += outputPrevBiasesDelta[i].ToString("F4") + " ";
            s += "\n\n";

            s += "outputs: \n";
            for (int i = 0; i < outputs.Length; ++i)
                s += outputs[i].ToString("F2") + " ";
            s += "\n\n";

            s += "===============================\n";
            return s;
        }

        // ----------------------------------------------------------------------------------------

        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: ");

            int k = 0; // points into weights param

            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    inputHiddenWeights[i][j] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                hiddenBiases[i] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hiddenOutputWeights[i][j] = weights[k++];
            for (int i = 0; i < numOutput; ++i)
                outputBiases[i] = weights[k++];
        }

        public void InitializeWeights()
        {
            // initialize weights and biases to small random values
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] initialWeights = new double[numWeights];
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
            this.SetWeights(initialWeights);
        }

        public double[] GetWeights()
        {
            // returns the current set of wweights, presumably after training
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < inputHiddenWeights.Length; ++i)
                for (int j = 0; j < inputHiddenWeights[0].Length; ++j)
                    result[k++] = inputHiddenWeights[i][j];
            for (int i = 0; i < hiddenBiases.Length; ++i)
                result[k++] = hiddenBiases[i];
            for (int i = 0; i < hiddenOutputWeights.Length; ++i)
                for (int j = 0; j < hiddenOutputWeights[0].Length; ++j)
                    result[k++] = hiddenOutputWeights[i][j];
            for (int i = 0; i < outputBiases.Length; ++i)
                result[k++] = outputBiases[i];
            return result;
        }

        // ----------------------------------------------------------------------------------------

        private double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array length");

            double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
            double[] oSums = new double[numOutput]; // output nodes sums

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];

            for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += this.inputs[i] * this.inputHiddenWeights[i][j]; // note +=

            for (int i = 0; i < numHidden; ++i)  // add biases to input-to-hidden sums
                hSums[i] += this.hiddenBiases[i];

            for (int i = 0; i < numHidden; ++i)   // apply activation
                this.hiddenOutputs[i] = HyperTanFunction(hSums[i]); // hard-coded

            for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hiddenOutputs[i] * hiddenOutputWeights[i][j];

            for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
                oSums[i] += outputBiases[i];

            double[] softOut = Softmax(oSums); // softmax activation does all outputs at once for efficiency
            Array.Copy(softOut, outputs, softOut.Length);

            double[] retResult = new double[numOutput]; // could define a GetOutputs method instead
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        } // ComputeOutputs

        private static double HyperTanFunction(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        private static double[] Softmax(double[] oSums)
        {
            // determine max output sum
            // does all output nodes at once so scale doesn't have to be re-computed each time
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale;

            return result; // now scaled so that xi sum to 1.0
        }

        // ----------------------------------------------------------------------------------------

        private void UpdateWeights(double[] tValues, double learnRate, double momentum, double weightDecay)
        {
            // update the weights and biases using back-propagation, with target values, eta (learning rate),
            // alpha (momentum).
            // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays
            // and matrices have values (other than 0.0)
            if (tValues.Length != numOutput)
                throw new Exception("target values not same Length as output in UpdateWeights");

            // 1. compute output gradients
            for (int i = 0; i < outputGrads.Length; ++i)
            {
                // derivative of softmax = (1 - y) * y (same as log-sigmoid)
                double derivative = (1 - outputs[i]) * outputs[i];
                // 'mean squared error version' includes (1-y)(y) derivative
                outputGrads[i] = derivative * (tValues[i] - outputs[i]);
            }

            // 2. compute hidden gradients
            for (int i = 0; i < hiddenGrads.Length; ++i)
            {
                // derivative of tanh = (1 - y) * (1 + y)
                double derivative = (1 - hiddenOutputs[i]) * (1 + hiddenOutputs[i]);
                double sum = 0.0;
                for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
                {
                    double x = outputGrads[j] * hiddenOutputWeights[i][j];
                    sum += x;
                }
                hiddenGrads[i] = derivative * sum;
            }

            // 3a. update hidden weights (gradients must be computed right-to-left but weights
            // can be updated in any order)
            for (int i = 0; i < inputHiddenWeights.Length; ++i) // 0..2 (3)
            {
                for (int j = 0; j < inputHiddenWeights[0].Length; ++j) // 0..3 (4)
                {
                    double delta = learnRate * hiddenGrads[j] * inputs[i]; // compute the new delta
                    inputHiddenWeights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
                                              // now add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                    inputHiddenWeights[i][j] += momentum * inputHiddenPrevWeightsDelta[i][j];
                    inputHiddenWeights[i][j] -= (weightDecay * inputHiddenWeights[i][j]); // weight decay
                    inputHiddenPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 
                }
            }

            // 3b. update hidden biases
            for (int i = 0; i < hiddenBiases.Length; ++i)
            {
                double delta = learnRate * hiddenGrads[i] * 1.0; // t1.0 is constant input for bias; could leave out
                hiddenBiases[i] += delta;
                hiddenBiases[i] += momentum * hiddenPrevBiasesDelta[i]; // momentum
                hiddenBiases[i] -= (weightDecay * hiddenBiases[i]); // weight decay
                hiddenPrevBiasesDelta[i] = delta; // don't forget to save the delta
            }

            // 4. update hidden-output weights
            for (int i = 0; i < hiddenOutputWeights.Length; ++i)
            {
                for (int j = 0; j < hiddenOutputWeights[0].Length; ++j)
                {
                    // see above: hOutputs are inputs to the nn outputs
                    double delta = learnRate * outputGrads[j] * hiddenOutputs[i];
                    hiddenOutputWeights[i][j] += delta;
                    hiddenOutputWeights[i][j] += momentum * hiddenOutputPrevWeightsDelta[i][j]; // momentum
                    hiddenOutputWeights[i][j] -= (weightDecay * hiddenOutputWeights[i][j]); // weight decay
                    hiddenOutputPrevWeightsDelta[i][j] = delta; // save
                }
            }

            // 4b. update output biases
            for (int i = 0; i < outputBiases.Length; ++i)
            {
                double delta = learnRate * outputGrads[i] * 1.0;
                outputBiases[i] += delta;
                outputBiases[i] += momentum * outputPrevBiasesDelta[i]; // momentum
                outputBiases[i] -= (weightDecay * outputBiases[i]); // weight decay
                outputPrevBiasesDelta[i] = delta; // save
            }
        } // UpdateWeights

        // ----------------------------------------------------------------------------------------

        public void Train(double[][] trainData, int maxEprochs, double learnRate, double momentum,
          double weightDecay)
        {
            // train a back-prop style NN classifier using learning rate and momentum
            // weight decay reduces the magnitude of a weight value over time unless that value
            // is constantly increased
            int epoch = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // target values

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            while (epoch < maxEprochs)
            {
                double mse = MeanSquaredError(trainData);
                if (mse < 0.020) break; // consider passing value in as parameter
                                        //if (mse < 0.001) break; // consider passing value in as parameter

                Shuffle(sequence); // visit each training data in random order
                for (int i = 0; i < trainData.Length; ++i)
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], xValues, numInput);
                    Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
                    ComputeOutputs(xValues); // copy xValues in, compute outputs (store them internally)
                    UpdateWeights(tValues, learnRate, momentum, weightDecay); // find better weights
                } // each training tuple
                ++epoch;
            }
        } // Train

        private static void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        private double MeanSquaredError(double[][] trainData) // used as a training stopping condition
        {
            // average squared error per training tuple
            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput]; // first numInput values in trainData
            double[] tValues = new double[numOutput]; // last numOutput values

            // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
                double[] yValues = this.ComputeOutputs(xValues); // compute output using current weights
                for (int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }

            return sumSquaredError / trainData.Length;
        }

        // ----------------------------------------------------------------------------------------

        public double Accuracy(double[][] testData)
        {
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double[] yValues; // computed Y

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // parse test data into x-values and t-values
                Array.Copy(testData[i], numInput, tValues, 0, numOutput);
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

                if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
        }

        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i]; bigIndex = i;
                }
            }
            return bigIndex;
        }

    } // NeuralNetwork
}

