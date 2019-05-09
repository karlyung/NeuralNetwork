//Import System Components
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections;
//using NeuralNetwork.MyMath;

//File Name
namespace NeuralNetwork
{
    //Class Name
    class Neuron
    {
        //Class Field Variables
        const double DEFAULT_WEIGHT = 0d;

        //This Neuron's Layer #
        int layer = -1;

        //This Neuron's Input Neurons' Coordinates
        List<List<int>> inputNeurons = new List<List<int>>(); //m,n

        //This Neuron's Input Neurons' Values
        List<double> inputs = new List<double>();

        //Collection of Weights for All Input Neurons
        public List<double> weights = new List<double>();

        //This Neuron's Bias Value
        public double beta = 0f;

        //Activation Function Type
        public enum actFuncType { tanh, identity, logistic, ReLU };
        public actFuncType actFunc = actFuncType.logistic;

        //Pre-activation Output Save
        double output = 0f;

        //This Neuron's Final, Post-activation Output
        public double postActOutput = 0f;

        // dE/dW = dE/dOut * dOut/dNet * dNet/dW
        double dE_over_dOut = 0f;
        double dOut_over_dNet = 0f;
        List<double> dNet_over_dW = new List<double>();
        List<double> dE_over_dW = new List<double>();

        //New Weights with the Gradient Applied (stored for switching after iteration)
        public List<double> newWeights = new List<double>();

        //Empty Constructor
        public Neuron()
        {
            //this.layer = 0;
        }

        //Minimal Hidden or Output Neuron Initializing Constructor
        public Neuron(main.Network inNet, List<List<int>> inInputNs, int inLayer)
        {
            this.inputNeurons = inInputNs;
            this.layer = inLayer;

            this.weights = new List<double>();
            for(int i = 0; i < inNet.numNeuronsEachLayer[inLayer - 1]; i++)
            {
                if (inNet.nNarray[i, inLayer - 1] != null)
                {
                    this.weights.Add(DEFAULT_WEIGHT);
                }
                else
                {
                    this.weights.Add(0f);
                }
            }
        }

        //Input Neuron Constructor
        public Neuron(double finalOutput)
        {
            this.layer = 0;
            this.postActOutput = finalOutput;
        }

        //Hidden and Output Neuron Constructor
        public Neuron(int layer, actFuncType func, List<List<int>> inputNeu, List<double> weights, double beta)
        {
            this.layer = layer;
            this.actFunc = func;

            //this.inputNeurons = inputNeu;
            foreach (List<int> vec in inputNeu)
            {
                List<int> curVec = new List<int>();
                foreach(int i in vec)
                {
                    curVec.Add(i);
                }
                this.inputNeurons.Add(curVec);
            }
            
            //this.weights = weights;
            foreach (double w in weights)
            {
                this.weights.Add(w);
            }

            this.beta = beta;
        }

        public void setInputNeurons(List<List<int>> x)
        {
            this.inputNeurons = x;
        }

        public void setInputs(List<double> x)
        {
            this.inputs = x;
        }

        public void setWeights(List<double> x)
        {
            this.weights = x;
        }

        public void setActFunc(actFuncType x)
        {
            this.actFunc = x;
        }

        public void set_dNet_over_dW(List<double> x)
        {
            this.dNet_over_dW = x;
        }

        public void set_dE_over_dW(List<double> x)
        {
            this.dE_over_dW = x;
        }

        public void setNewWeights(List<double> x)
        {
            this.newWeights = x;
        }

        //Get Values from all Input Neurons' Coordinates
        public string GetInputs(Neuron[,] nn)
        {
            this.inputs.Clear();

            foreach (List<int> curNeuron in this.inputNeurons)
            {
                this.inputs.Add(nn[curNeuron[0], curNeuron[1]].postActOutput);
            }

            return null;
        }

        //Forward Propagate this Neuron
        public string ForwardPropag(Neuron[,] nn)
        {
            this.GetInputs(nn);

            this.output = MyMath.colMatMulti(this.inputs, this.weights);

            if (this.actFunc == actFuncType.identity)
            {
                this.postActOutput = MyMath.identityFunc(this.output, this.beta);
            }
            else if (this.actFunc == actFuncType.logistic){
                this.postActOutput = MyMath.logisticFunc(this.output, this.beta);
            }
            else if (this.actFunc == actFuncType.ReLU)
            {
                this.postActOutput = MyMath.reluFunc(this.output, this.beta);
            }
            else if (this.actFunc == actFuncType.tanh)
            {
                this.postActOutput = MyMath.tanhFunc(this.output, this.beta);
            }

            return null;
        }

        //Back Propagate for Output Neuron
        public int BackPropag(Neuron[,] net, double expOut, double learnRate)
        {
            if (this.actFunc == actFuncType.logistic)
            {
                this.dE_over_dOut = this.postActOutput - expOut;
                this.dOut_over_dNet = this.postActOutput * (1 - this.postActOutput);
                this.dNet_over_dW = Enumerable.Repeat(0d, net.GetLength(0)).ToList();
                this.dE_over_dW = Enumerable.Repeat(0d, net.GetLength(0)).ToList();
                this.newWeights = Enumerable.Repeat(0d, net.GetLength(0)).ToList();

                int i = 0;
                foreach (List<int> input in inputNeurons)
                {
                    this.dNet_over_dW[inputNeurons[i][0]] = net[inputNeurons[i][0], this.layer - 1].postActOutput;
                    //SHOULD SAVE NEW WEIGHTS INSTEAD OF OVER WRITING
                    if (this.weights[inputNeurons[i][0]] != 0) {
                        this.weights[inputNeurons[i][0]] -= this.dE_over_dOut * this.dOut_over_dNet * this.dNet_over_dW[i];
                        this.dE_over_dW[inputNeurons[i][0]] = (this.dE_over_dOut * this.dOut_over_dNet * this.dNet_over_dW[i]);
                        this.newWeights[inputNeurons[i][0]] = (this.weights[i] - learnRate * this.dE_over_dW[i]);
                    }

                    i++;
                }
                return 0;
            }
            else
            {
                return -1;
                throw new Exception("Activation function not ready for back propagation.");
            }
        }

        //Back Propagate for Hidden Neuron
        public int BackPropag(main.Network net, int mPos, double learnRate)
        {
            this.dE_over_dOut = 0;

            if (this.actFunc == actFuncType.logistic)
            {
                for (int i = 0; i < net.numNeuronsEachLayer[layer + 1]; i++)
                {
                    //if (net[i, this.layer + 1].weights[mPos] != 0)
                    //{
                        Neuron outNeuron = net.nNarray[i, this.layer + 1];
                        this.dE_over_dOut += outNeuron.dE_over_dOut * outNeuron.dOut_over_dNet * outNeuron.weights[mPos];
                    //}
                }

                this.dOut_over_dNet = this.postActOutput * (1 - this.postActOutput);
                for (int i = 0; i < this.inputs.Count; i++)
                {
                    this.dNet_over_dW.Add(this.inputs[i]);

                    this.dE_over_dW.Add(this.dE_over_dOut * this.dOut_over_dNet * this.dNet_over_dW[i]);

                    this.newWeights.Add(this.weights[i] - learnRate * this.dE_over_dW[i]);
                }

                return 0;
            }
            else
            {
                //return -1;
                throw new Exception("Activation function not ready for back propagation.");
            }
        }

        //Switch to New Weights, Clear New Weights Variable for Next Iteration
        public void switchToNewWeights()
        {
            for(int i = 0; i < this.weights.Count; i++)
            {
                this.weights[i] = this.newWeights[i];
            }
            this.newWeights.Clear();
        }
    }
}
