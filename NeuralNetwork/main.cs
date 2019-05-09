//Import System Components
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

//File Name
namespace NeuralNetwork
{
    //Class Name
    class main
    {
        //PLEASE ENTER THE .CSV FILES' PATH!!!
        static string csvPath = "C:\\Users\\Karl\\Desktop";
        //enter "\" twice in path above!!!
        //for exmaple: "C:\\Users\\Karl\\Desktop"

        //This collection of code includes 3 versions of code to train and run neural networks:
        //Ver 1 (Abandoned): Scalar NN (Network, Neuron), each input/hidden/output neuron takes scalar inputs, weights, and outputs
        //Ver 2: Vector NN (vNetwork, vNeuron), each neuron takes vector inputs, bias, outputs, and matrix weights
        //Ver 3: Matrix NN (mNetwork, mNeuron), each neuron takes matrix inputs, weights, bias, and outputs

        //Conventions
        //Neuron coordinates are in the form M (position in layer), N (layer #).
        //For loops, element 0 is the first; element N-1 is the last.
            //For example, 10 data points will be stored as "dataPoint[0], dataPoint[1], dataPoint[2], ... ,dataPoint[9]".
        
        //Global Random Number Generator
        public static Random rng = new Random();

        //Scalar Network (for vector, matrix networks, see bottom of this file)
        public class Network
        {
            //Class Field Variables
            public int NUM_DEPTH;
            public int NUM_LAYERS;
            public Neuron[,] nNarray = null;
            public double learnRate = 1f;
            public double[] expOut = null;
            public List<int> numNeuronsEachLayer = new List<int>();

            //Empty Initializing Constructor
            public Network()
            {
                this.NUM_DEPTH = -1;
                this.NUM_LAYERS = -1;
                this.nNarray = null;
            }

            //Standard Constructor
            public Network(int inDepth, int inLayers)
            {
                this.NUM_DEPTH = inDepth;
                this.NUM_LAYERS = inLayers;
                this.nNarray = new Neuron[inDepth, inLayers];
                this.expOut = new double[inDepth];

                for (int i = 0; i < inLayers; i++)
                {
                    for (int j = 0; j < inDepth; j++)
                    {
                        this.nNarray[j, i] = new Neuron();
                    }
                }
            }

            //Record # Neurons in Each Layer (for counting purposes)
            public void setNumNeuronsEachLayer(string x)
            {
                string[] split = x.Split(',');
                foreach (string i in split)
                {
                    numNeuronsEachLayer.Add(int.Parse(i));
                }
            }

            //Set 1 Input Neuron
            public void setInputNeuron(int m, double finalOutput) //m: neuron position in Layer, finalOutput: final output for this neuron
            {
                this.nNarray[m, 0] = new Neuron(finalOutput);
            }

            //Set 1 Hidden or Output Neuron
            public void setHidOutNeuron(int m, int n, Neuron.actFuncType func, List<List<int>> inputNeu, List<double> weights, double beta)
            { //m: neuron position in layer, n: layer #, func: activation function type, inputNeu: this neuron's input neuron coordinates, weights: collection of weights for all inputs, beta: bias variable
                this.nNarray[m, n] = new Neuron(n, func, inputNeu, weights, beta);
            }

            //Change Network Shape and Clear Neurons
            public int resetNetwork(int inDepth, int inLayers)
            {
                this.NUM_DEPTH = inDepth;
                this.NUM_LAYERS = inLayers;
                this.nNarray = new Neuron[inDepth, inLayers];
                this.expOut = new double[inDepth];

                for (int i = 0; i < inLayers; i++)
                {
                    for (int j = 0; j < inDepth; j++)
                    {
                        this.nNarray[j, i] = new Neuron();
                    }
                }

                return 0;
            }

            //Calculate Half Total Square Error of All Output Neurons
            public double calSqError()
            {
                double result = 0f;

                //if (this.expOut.Length == this.numNeuronsEachLayer[this.NUM_LAYERS - 1])
                //{
                    for (int i = 0; i < this.expOut.Length; i++)
                    {
                        result += 0.5 * Math.Pow(this.expOut[i] - this.nNarray[i, this.NUM_LAYERS - 1].postActOutput, 2);
                    }

                    return result;
                //}
                //else
                //{
                //    return -1;
                //}
            }

            //Change Weights to New Weights (which the gradient is applied to) and Clear New Weights (this method should be used after each satisfactory backward propagation run)
            public void switchToNewWeights()
            {
                for (int i = 0; i < this.NUM_LAYERS; i++)
                {
                    for (int j = 0; j < this.NUM_DEPTH; j++)
                    {
                        if (nNarray[j, i].weights.Count == nNarray[j, i].newWeights.Count)
                        {
                            nNarray[j, i].switchToNewWeights();
                        }
                    }
                }
            }

            //Forward Propagates and Backward Propagates, Continue Loop If there is a Decrease in Half Square Error Large Enough (>0.000001)
            public void train()
            {
                double lastRunSqError = 0f;
                int runNum = 1;
                double sqError = double.MaxValue;

                do
                {
                    lastRunSqError = sqError;

                    forwardPropNet(this);
                    Console.WriteLine("Forward Propagation Complete, Run Number " + runNum);

                    backPropNet(this);
                    Console.WriteLine("Back Propagation Complete, Run Number " + runNum);

                    runNum++;

                    sqError = this.calSqError();

                    for (int i = 0; i < this.NUM_DEPTH; i++)
                    {
                        if (this.nNarray[i, this.NUM_LAYERS - 1].postActOutput != 0)
                        {
                            Console.WriteLine("Output Neuron #" + i + " Ouputs Value of " + this.nNarray[i, this.NUM_LAYERS - 1].postActOutput + "(" + this.expOut[i] + ")");
                        }
                    }
                    Console.WriteLine("Half Square Error is " + sqError);
                }
                while (lastRunSqError - sqError > 0.000001);
            }
        }

        //Program Starting Point (enter the wanted method/sub-program inside)
        static void Main()
        {
            //Make Sure to Have a Pause Point Just Before the End of the Method to See Variables Inside the NN!!!
            setFinal3_Basecase(); //Base Control Case from Presentation

            //Below is one of the optimization runs for # of neurons in 1 hidden layer
            //List<double> output = new List<double>();
            //for (int i = 30; i <= 210; i+=30)
            //{
            //    output = setPaperMatCnyNasdaqNikkeiCase(6, 60, 1, i);

            //    for (int j = 0; j < output.Count; j++)
            //    {
            //        Console.Write(output[j] + ",");
            //    }
            //    Console.WriteLine("");
            //}

            Console.ReadLine(); //Adds a Pause at the End of the Program (press any key to continue)
        }

        //Create a New Blank Matrix (used in mNeuron (matrix neuron) class to solve instancing issues with matrices)
        public static List<List<double>> createNewMatrix(int mCount, int nCount)
        {
            List<List<double>> result = new List<List<double>>();

            for (int i = 0; i < mCount; i++)
            {
                result.Add(new List<double>());
                for(int j = 0; j < nCount; j++)
                {
                    result[i].Add(0d);
                }
            }
            return result;
        }

        //Forward Propagate Scalar Network
        static int forwardPropNet(Network net)
        {
            for (int i = 1; i < net.NUM_LAYERS; i++)
            {
                for (int j = 0; j < net.numNeuronsEachLayer[i]; j++)
                {
                    if (net.nNarray[j, i] != null)
                        net.nNarray[j, i].ForwardPropag(net.nNarray);
                }
            }

            return 0;
        }

        //Forward Propagate Vector Network
        static int vForwardPropNet(vNetwork net)
        {
            for (int i = 0; i < net.NUM_LAYERS; i++)
            {
                for (int j = 0; j < net.numNeuronsEachLayer[i]; j++)
                {
                    if (net.nNarray[j, i] != null)
                        net.nNarray[j, i].ForwardPropag(net.nNarray);
                }
            }

            return 0;
        }

        //Backward Propagate Scalar Network
        static int backPropNet(Network net)
        {
            //Back Prog Output Neurons
            for (int i = 0; i < net.numNeuronsEachLayer[net.NUM_LAYERS - 1]; i++)
            {
                net.nNarray[i, net.NUM_LAYERS - 1].BackPropag(net.nNarray, net.expOut[i], net.learnRate);
            }

            //Back Prog Hidden Neurons
            for (int i = net.NUM_LAYERS - 2; i > 0; i--)
            {
                for (int j = 0; j < net.numNeuronsEachLayer[net.NUM_LAYERS - 1 - i]; j++)
                {
                    if (net.nNarray[j, i].weights.Count != 0 && net.nNarray[j, i].weights != null)
                        net.nNarray[j, i].BackPropag(net.nNarray, j, net.learnRate);
                }
            }

            net.switchToNewWeights();

            return 0;
        }

        //Backward Propagate Vector Network
        static int vBackPropNet(vNetwork net)
        {
            double prevSqE = net.calHalfSqError();

            //Save All Weights to Revert to in case of Half Square Error Increasing
            List<List<List<List<List<double>>>>> prevWeights = new List<List<List<List<List<double>>>>>();
            for (int i = 0; i < net.NUM_LAYERS; i++)
            {
                prevWeights.Add(new List<List<List<List<double>>>>());
                for (int j = 0; j < net.NUM_DEPTH; j++)
                {
                    prevWeights[i].Add(new List<List<List<double>>>());
                    for (int k = 0; k < net.nNarray[j, i].weights.Count; k++)
                    {
                        prevWeights[i][j].Add(new List<List<double>>());
                        for (int l = 0; l < net.nNarray[j, i].weights[0].Count; l++)
                        {
                            prevWeights[i][j][k].Add(new List<double>());
                            for (int m = 0; m < net.nNarray[j, i].weights[0][0].Count; m++)
                            {
                                prevWeights[i][j][k][l].Add(net.nNarray[j, i].weights[k][l][m]);
                            }
                        }
                    }
                }
            }

            //Back Prog Output Neurons
            for (int i = 0; i < net.numNeuronsEachLayer[net.NUM_LAYERS - 1]; i++)
            {
                net.nNarray[i, net.NUM_LAYERS - 1].BackPropag(net.nNarray, net.expOut[i], net.learnRate, net.bLearnRate);
            }

            //Back Prog Hidden Neurons
            for (int i = net.NUM_LAYERS - 2; i >= 0; i--)
            {
                for (int j = 0; j < net.numNeuronsEachLayer[i]; j++)
                {
                    if (net.nNarray[j, i].weights.Count != 0 || net.nNarray[j, i].weights != null)
                        net.nNarray[j, i].BackPropag(net.nNarray, j, net.learnRate, net.bLearnRate);
                }
            }

            net.switchToNewWeights();

            double newSqE = net.calHalfSqError();
            if (newSqE > prevSqE)
            {
                for (int i = 0; i < net.NUM_LAYERS; i++)
                {
                    for (int j = 0; j < net.NUM_DEPTH; j++)
                    {
                        for (int k = 0; k < net.nNarray[j, i].weights.Count; k++)
                        {
                            for (int l = 0; l < net.nNarray[j, i].weights[0].Count; l++)
                            {
                                for (int m = 0; m < net.nNarray[j, i].weights[0][0].Count; m++)
                                {
                                    net.nNarray[j, i].weights[k][l][m] = prevWeights[i][j][k][l][m];
                                }
                            }
                        }
                    }
                }
            }

            return 0;
        }

        //Completely Clear Scalar Network
        static int clearNN(Network inNet)
        {
            inNet.NUM_DEPTH = -1;
            inNet.NUM_LAYERS = -1;
            inNet.nNarray = null;

            return 1;
        }

        //Scalar Input Neuron Initializer (old code, not used)
        static int setInputNeuron(Network inNet, int m, double neuronOutput)
        {
            inNet.nNarray[m, 0] = new Neuron();
            inNet.nNarray[m, 0].postActOutput = neuronOutput;
            return 0;
        }

        //Scalar Hidden Neuron Initializer (old code, not used)
        static int setHiddenNeuron(Network inNet, int n, int m)
        {
            List<List<int>> inNeuList = new List<List<int>>();
            int prevLayerDepth = inNet.numNeuronsEachLayer[n - 1];

            for (int j = 0; j < prevLayerDepth; j++)
            {
                List<int> inNeuCoor = new List<int>();
                inNeuCoor.Add(j); //m of Input Neuron Coordinates
                inNeuCoor.Add(n - 1); //n
                inNeuList.Add(inNeuCoor);
            }

            inNet.nNarray[m, n] = new Neuron(inNet, inNeuList, n);

            return 0;
        }

        //Scalar Hidden Neuron Initializer with More Parameters (old code, not used)
        static int setHiddenNeuron(Network inNet, int n, int m, double weight, double beta, Neuron.actFuncType actFunc)
        {
            List<List<int>> inNeuList = new List<List<int>>();
            int prevLayerDepth = inNet.NUM_DEPTH;

            for (int j = 0; j < prevLayerDepth; j++)
            {
                List<int> inNeuCoor = new List<int>();
                inNeuCoor.Add(j); //m of Input Neuron Coordinates
                inNeuCoor.Add(n - 1); //n
                inNeuList.Add(inNeuCoor);
            }

            Neuron currentNeuron = inNet.nNarray[m, n];
            currentNeuron = new Neuron(inNet, inNeuList, n);

            for (int i = 0; i < inNet.NUM_DEPTH; i++)
            {
                currentNeuron.weights[i] = weight;
            }

            currentNeuron.beta = beta;
            currentNeuron.actFunc = actFunc;

            return 0;
        }

        //Get All Neurons' Matrix Weights in a Network (used to save for later)
        static List<List<List<List<List<double>>>>> vCopyWeights(vNeuron[,] net, int NUM_DEPTH, int NUM_LAYERS)
        {
            List<List<List<List<List<double>>>>> result = new List<List<List<List<List<double>>>>>();

            for (int i = 0; i < NUM_LAYERS; i++)
            {
                result.Add(new List<List<List<List<double>>>>());
                for (int j = 0; j < NUM_DEPTH; j++)
                {
                    result[i].Add(new List<List<List<double>>>());
                    for (int k = 0; k < net[j, i].weights.Count; k++)
                    {
                        result[i][j].Add(new List<List<double>>());
                        for (int l = 0; l < net[j, i].weights[0].Count; l++)
                        {
                            result[i][j][k].Add(new List<double>());
                            for (int m = 0; m < net[j, i].weights[0][0].Count; m++)
                            {
                                result[i][j][k][l].Add(net[j, i].weights[k][l][m]);
                            }
                        }
                    }
                }
            }

            return result;
        }

        //Apply a Previous Save of All Neuron's Matrix Weights (used with vCopyWeights())
        static void vApplyWeights(List<List<List<List<List<double>>>>> weights, vNeuron[,] net, int NUM_DEPTH, int NUM_LAYERS)
        {
            for (int i = 0; i < NUM_LAYERS; i++)
            {
                for (int j = 0; j < NUM_DEPTH; j++)
                {
                    for (int k = 0; k < net[j, i].weights.Count; k++)
                    {
                        for (int l = 0; l < net[j, i].weights[0].Count; l++)
                        {
                            for (int m = 0; m < net[j, i].weights[0][0].Count; m++)
                            {
                                net[j, i].weights[k][l][m] = weights[i][j][k][l][m];
                            }
                        }
                    }
                }
            }
        }

        //Generic Scalar NN for Testing (Rectangular)
        static void setNNcase0(Network inNet)
        {
            //Setup
            clearNN(inNet);
            inNet.NUM_DEPTH = 3;
            inNet.NUM_LAYERS = 3;
            inNet.nNarray = new Neuron[inNet.NUM_DEPTH, inNet.NUM_LAYERS];
            inNet.setNumNeuronsEachLayer("3,3,3");

            //Input Neurons
            for (int i = 0; i < inNet.NUM_DEPTH; i++)
            {
                inNet.nNarray[i, 0] = new Neuron();
                inNet.nNarray[i, 0].postActOutput = i + 1;
            }

            //Hidden Neurons
            for (int i = 0; i < inNet.NUM_DEPTH; i++)
            {
                List<List<int>> iNl = new List<List<int>>();
                for (int j = 0; j < inNet.NUM_DEPTH; j++)
                {
                    List<int> iNc = new List<int>();
                    iNc.Add(j); //m of Input Neuron Coordinates
                    iNc.Add(0); //n
                    iNl.Add(iNc);
                }
                inNet.nNarray[i, 1] = new Neuron(inNet, iNl, 1);
                inNet.nNarray[i, 1].actFunc = Neuron.actFuncType.logistic;
            }

            //Output Neurons
            for (int i = 0; i < inNet.NUM_DEPTH; i++)
            {
                List<List<int>> iNl = new List<List<int>>();
                for (int j = 0; j < inNet.NUM_DEPTH; j++)
                {
                    List<int> iNc = new List<int>();
                    iNc.Add(j); //m
                    iNc.Add(1); //n
                    iNl.Add(iNc);
                }
                inNet.nNarray[i, 2] = new Neuron(inNet, iNl, 2);
                inNet.nNarray[i, 2].actFunc = Neuron.actFuncType.logistic;
            }

            //Expected Outputs
            inNet.expOut = new double[] { 0.3d, 0.2d, 0.1d };

            //Run
            inNet.train();
        }

        //Generic Scalar NN for Testing (Non-rectangular)
        static void setNNcase1(Network inNet)
        {
            //Setup
            clearNN(inNet);
            inNet.NUM_DEPTH = 3;
            inNet.NUM_LAYERS = 3;
            inNet.resetNetwork(3, 3);
            inNet.setNumNeuronsEachLayer("2,3,1");

            //Input Neurons
            setInputNeuron(inNet, 0, 0.2f);
            setInputNeuron(inNet, 1, -0.2f);

            //Hidden Neurons
            setHiddenNeuron(inNet, 1, 0);
            inNet.nNarray[0, 1].weights[0] = 0.5 / 2f;
            inNet.nNarray[0, 1].weights[1] = 0.5 / 2f;
            inNet.nNarray[0, 1].beta = 0f;
            inNet.nNarray[0, 1].actFunc = Neuron.actFuncType.logistic;

            setHiddenNeuron(inNet, 1, 1);
            inNet.nNarray[1, 1].weights[0] = 0.9 / 2f;
            inNet.nNarray[1, 1].weights[1] = -0.1 / 2f;
            inNet.nNarray[1, 1].beta = 0f;
            inNet.nNarray[1, 1].actFunc = Neuron.actFuncType.logistic;

            setHiddenNeuron(inNet, 1, 2);
            inNet.nNarray[2, 1].weights[0] = 0.9 / 2f;
            inNet.nNarray[2, 1].weights[1] = 0.1 / 2f;
            inNet.nNarray[2, 1].beta = 0f;
            inNet.nNarray[2, 1].actFunc = Neuron.actFuncType.logistic;

            //Output Neurons
            setHiddenNeuron(inNet, 2, 0);
            inNet.nNarray[0, 2].weights[0] = 0.5 / 3f;
            inNet.nNarray[0, 2].weights[1] = 0.6 / 3f;
            inNet.nNarray[0, 2].weights[2] = 0.7 / 3f;
            inNet.nNarray[0, 2].actFunc = Neuron.actFuncType.logistic;

            //Expected Outputs
            inNet.expOut = new double[] { 0.5d };

            //Run
            inNet.train();
        }

        //Generic Scalar NN for Testing (4 Layers)
        static void setNNcase2(Network inNet)
        {
            clearNN(inNet);
            inNet.NUM_DEPTH = 4;
            inNet.NUM_LAYERS = 4;
            inNet.resetNetwork(4, 4);
            inNet.setNumNeuronsEachLayer("2,3,4,1");

            //Input Neurons
            setInputNeuron(inNet, 0, 1f);
            setInputNeuron(inNet, 1, 0.5f);

            //Hidden Neurons
            setHiddenNeuron(inNet, 1, 0);
            inNet.nNarray[0, 1].weights[0] = 0.5 / 2f;
            inNet.nNarray[0, 1].weights[1] = 0.5 / 2f;
            inNet.nNarray[0, 1].beta = 0f;
            inNet.nNarray[0, 1].actFunc = Neuron.actFuncType.logistic;

            setHiddenNeuron(inNet, 1, 1);
            inNet.nNarray[1, 1].weights[0] = 0.9 / 2f;
            inNet.nNarray[1, 1].weights[1] = -0.1 / 2f;
            inNet.nNarray[1, 1].beta = 0f;
            inNet.nNarray[1, 1].actFunc = Neuron.actFuncType.logistic;

            setHiddenNeuron(inNet, 1, 2);
            inNet.nNarray[2, 1].weights[0] = 0.9 / 2f;
            inNet.nNarray[2, 1].weights[1] = 0.1 / 2f;
            inNet.nNarray[2, 1].beta = 0f;
            inNet.nNarray[2, 1].actFunc = Neuron.actFuncType.logistic;

            setHiddenNeuron(inNet, 2, 0);
            inNet.nNarray[0, 2].weights[0] = 0.5 / 3f;
            inNet.nNarray[0, 2].weights[1] = 0.6 / 3f;
            inNet.nNarray[0, 2].weights[2] = 0.2 / 3f;
            inNet.nNarray[0, 2].beta = 0f;
            inNet.nNarray[0, 2].actFunc = Neuron.actFuncType.logistic;

            setHiddenNeuron(inNet, 2, 1);
            inNet.nNarray[1, 2].weights[0] = 0.3 / 3f;
            inNet.nNarray[1, 2].weights[1] = 0.4 / 3f;
            inNet.nNarray[1, 2].weights[2] = 0.4 / 3f;
            inNet.nNarray[1, 2].beta = 0f;
            inNet.nNarray[1, 2].actFunc = Neuron.actFuncType.logistic;

            setHiddenNeuron(inNet, 2, 2);
            inNet.nNarray[2, 2].weights[0] = 0.2 / 3f;
            inNet.nNarray[2, 2].weights[1] = 0.3 / 3f;
            inNet.nNarray[2, 2].weights[2] = 0.5 / 3f;
            inNet.nNarray[2, 2].beta = 0f;
            inNet.nNarray[2, 2].actFunc = Neuron.actFuncType.logistic;

            setHiddenNeuron(inNet, 2, 3);
            inNet.nNarray[3, 2].weights[0] = 0.7 / 3f;
            inNet.nNarray[3, 2].weights[1] = 0.5 / 3f;
            inNet.nNarray[3, 2].weights[2] = 0.9 / 3f;
            inNet.nNarray[3, 2].beta = 0f;
            inNet.nNarray[3, 2].actFunc = Neuron.actFuncType.logistic;

            //Output Neurons
            setHiddenNeuron(inNet, 3, 0);
            inNet.nNarray[0, 3].weights[0] = 0.5 / 4f;
            inNet.nNarray[0, 3].weights[1] = 0.5 / 4f;
            inNet.nNarray[0, 3].weights[2] = 0.5 / 4f;
            inNet.nNarray[0, 3].beta = 0f;
            inNet.nNarray[0, 3].actFunc = Neuron.actFuncType.logistic;

            inNet.expOut = new double[] { 0.25 };

            inNet.train();
        }

        //Example with Calculations from the Internet (1 Cycle)
        static void setNNcase3(Network inNet)
        {
            //Setup
            clearNN(inNet);
            inNet.resetNetwork(2, 3);
            inNet.setNumNeuronsEachLayer("2,2,2");

            //Expected Output
            inNet.expOut[0] = 0.01f;
            inNet.expOut[1] = 0.99f;

            //Input Neurons
            setInputNeuron(inNet, 0, 0.05f);
            setInputNeuron(inNet, 1, 0.1f);

            //Hidden Neurons
            setHiddenNeuron(inNet, 1, 0);
            inNet.nNarray[0, 1].weights[0] = 0.15f;
            inNet.nNarray[0, 1].weights[1] = 0.2f;
            inNet.nNarray[0, 1].beta = 0.35f;
            inNet.nNarray[0, 1].actFunc = Neuron.actFuncType.logistic;

            setHiddenNeuron(inNet, 1, 1);
            inNet.nNarray[1, 1].weights[0] = 0.25f;
            inNet.nNarray[1, 1].weights[1] = 0.3f;
            inNet.nNarray[1, 1].beta = 0.35f;
            inNet.nNarray[1, 1].actFunc = Neuron.actFuncType.logistic;

            //Output Neurons
            setHiddenNeuron(inNet, 2, 0);
            inNet.nNarray[0, 2].weights[0] = 0.4f;
            inNet.nNarray[0, 2].weights[1] = 0.45f;
            inNet.nNarray[0, 2].beta = 0.6f;
            inNet.nNarray[0, 2].actFunc = Neuron.actFuncType.logistic;

            setHiddenNeuron(inNet, 2, 1);
            inNet.nNarray[1, 2].weights[0] = 0.5f;
            inNet.nNarray[1, 2].weights[1] = 0.55f;
            inNet.nNarray[1, 2].beta = 0.6f;
            inNet.nNarray[1, 2].actFunc = Neuron.actFuncType.logistic;

            //Run
            forwardPropNet(inNet);
            backPropNet(inNet);
        }

        //USD/JPY Scalar Prediction using 1 Year Data (please refer to setFinal3_Basecase() for the run used in PPT)
        static void setExchangeRateCase()
        {
            //Setup
            Network inNet = new Network();
            inNet.resetNetwork(273, 3);
            inNet.setNumNeuronsEachLayer("273,273,10");
            inNet.learnRate = 0.5d;

            //Take in File
            List<double> exRateDoubles = DataReader.readDoubleColumn(csvPath + "\\currencyData2.csv", 1);

            //Normalization
            List<double> normalExRate = new List<double>();
            for (int i = 0; i < exRateDoubles.Count; i++)
            {
                normalExRate.Add(0.5 * (exRateDoubles[i] - exRateDoubles.Min()) / (exRateDoubles.Max() - exRateDoubles.Min()));
            }

            //Expected Output
            for (int i = 0; i < 273; i++)
            {
                inNet.expOut[i] = normalExRate[10 + i];
            }

            //Input Neurons
            for (int i = 0; i < 273; i++)
            {
                inNet.setInputNeuron(i, normalExRate[i]);
            }

            //Hidden Neurons
            List<List<int>> allInput = new List<List<int>>();

            List<List<double>> allWeights = new List<List<double>>();
            List<double> noneWeights = new List<double>();

            Random rng = new Random();

            for (int i = 0; i < 273; i++)
            {
                allInput.Add(new List<int>());
                allInput[i].Add(i);
                allInput[i].Add(0);

                allWeights.Add(new List<double>());
                for (int j = 0; j < 273; j++)
                {
                    allWeights[i].Add(0.0005d * rng.NextDouble());
                }

                noneWeights.Add(0d);
            }

            for (int i = 0; i < 273; i++)
            {
                inNet.setHidOutNeuron(i, 1, Neuron.actFuncType.logistic, allInput, allWeights[i], 0d);
            }

            //Output Neurons
            allInput = new List<List<int>>();
            allWeights = new List<List<double>>();
            for (int i = 0; i < 273; i++)
            {
                allInput.Add(new List<int>());
                allInput[i].Add(i);
                allInput[i].Add(1);

                allWeights.Add(new List<double>());
                for (int j = 0; j < 273; j++)
                {
                    allWeights[i].Add(0.0001d * rng.NextDouble());
                }
            }

            for (int i = 0; i < 10; i++)
            {
                inNet.setHidOutNeuron(i, 2, Neuron.actFuncType.logistic, allInput, allWeights[i], 0d);
            }

            inNet.train();

            List<double> newInputs = normalExRate.Skip(10).ToList();

            int x = 0;
            foreach (double d in newInputs)
            {
                inNet.nNarray[x, 0].postActOutput = d;
                x++;
            }

            //Run
            forwardPropNet(inNet);

            //Output in Console
            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(inNet.nNarray[i, 2].postActOutput);
            }
        }

        //USD/JPY Scalar Prediction using 10 Years Data (expect high square error)
        static void setExRateLongCase()
        {
            //Setup
            Network network = new Network();
            network.resetNetwork(2622, 3);
            network.setNumNeuronsEachLayer("2622,273,10");
            network.learnRate = 0.5d;

            //Take in Data
            List<double> exRateDoubles = DataReader.readDoubleColumn(csvPath + "\\currencyDataLong.csv", 1);

            //Normalization
            List<double> normalExRate = new List<double>();
            for (int i = 0; i < exRateDoubles.Count; i++)
            {
                normalExRate.Add(0.5 * (exRateDoubles[i] - exRateDoubles.Min()) / (exRateDoubles.Max() - exRateDoubles.Min()));
            }

            for (int i = 0; i < 2622; i++)
            {
                network.expOut[i] = normalExRate[10 + i];
            }

            //Input Neurons
            for (int i = 0; i < 2622; i++)
            {
                network.setInputNeuron(i, normalExRate[i]);
            }

            //Hidden Neurons
            List<List<int>> allInput = new List<List<int>>();

            List<List<double>> allWeights = new List<List<double>>();
            List<double> noneWeights = new List<double>();

            Random rng = new Random();

            for (int i = 0; i < 2622; i++)
            {
                allInput.Add(new List<int>());
                allInput[i].Add(i);
                allInput[i].Add(0);

                allWeights.Add(new List<double>());
                for (int j = 0; j < 2622; j++)
                {
                    allWeights[i].Add(0.0005d * rng.NextDouble());
                }

                noneWeights.Add(0d);
            }

            for (int i = 0; i < 2622; i++)
            {
                network.setHidOutNeuron(i, 1, Neuron.actFuncType.logistic, allInput, allWeights[i], 0d);
            }

            //Output Neurons
            allInput = new List<List<int>>();
            allWeights = new List<List<double>>();
            for (int i = 0; i < 2622; i++)
            {
                allInput.Add(new List<int>());
                allInput[i].Add(i);
                allInput[i].Add(1);

                allWeights.Add(new List<double>());
                for (int j = 0; j < 2622; j++)
                {
                    allWeights[i].Add(0.0005d * rng.NextDouble());
                }
            }

            for (int i = 0; i < 10; i++)
            {
                network.setHidOutNeuron(i, 2, Neuron.actFuncType.logistic, allInput, allWeights[i], 0d);
            }

            //Run
            network.train();

            //New Inputs
            List<double> newInputs = normalExRate.Skip(10).ToList();

            int x = 0;
            foreach (double d in newInputs)
            {
                network.nNarray[x, 0].postActOutput = d;
                x++;
            }

            //Run #2
            forwardPropNet(network);

            for (int i = 0; i < 10; i++)
            {
                Console.WriteLine(network.nNarray[i, 2].postActOutput);
            }

            Console.WriteLine("");
        }

        //Horizontal Extension in Scalar Method (Abandoned!)
        static void setExRateMultiInputsCase()
        {
            //Setup
            Network network = new Network();
            network.learnRate = 1d;
            const int numTestEntries = 10;

            //Take in Data and Normalization
            List<double> data = DataReader.readDoubleColumn(csvPath + "\\usdjpycnyjpyndnk.csv", 1);
            network.resetNetwork((data.Count - 1 - numTestEntries) * 2, 3);
            network.setNumNeuronsEachLayer((data.Count - 1 - numTestEntries) * 2 + ",100," + (data.Count - 1 - numTestEntries) * 2);

            List<double> normData = new List<double>();
            for (int i = 0; i < data.Count; i++)
            {
                normData.Add(0.5 * (data[i] - data.Min()) / (data.Max() - data.Min()));
            }

            List<double> data2 = DataReader.readDoubleColumn(csvPath + "\\usdjpycnyjpyndnk.csv", 2);
            List<double> normData2 = new List<double>();
            for (int i = 0; i < data2.Count; i++)
            {
                normData.Add(0.5 * (data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
                normData2.Add(0.5 * (data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
            }

            //Expected Output
            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                if (i == 0)
                {
                    network.expOut[i] = normData.Last();
                }
                else
                {
                    network.expOut[i] = 0;
                }
            }

            //Input Neurons
            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                network.setInputNeuron(i, normData[i]);
            }

            //Hidden Neurons
            List<List<int>> input1 = new List<List<int>>();
            List<List<int>> input2 = new List<List<int>>();

            List<List<double>> weights1 = new List<List<double>>();
            List<List<double>> weights2 = new List<List<double>>();
            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                weights1.Add(new List<double>());
                weights2.Add(new List<double>());
            }

            List<double> noneWeights = new List<double>();

            Random rng = new Random();

            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                input1.Add(new List<int>());
                if (i < network.NUM_DEPTH / 2)
                {
                    input1[i].Add(i);
                    input1[i].Add(0);
                }
                else
                {
                    input1[i].Add(0);
                    input1[i].Add(0);
                }

                noneWeights.Add(0d);

                input2.Add(new List<int>());
                if (i > network.NUM_DEPTH / 2)
                {
                    input2[i].Add(i);
                    input2[i].Add(0);
                }
                else
                {
                    input2[i].Add(0);
                    input2[i].Add(0);
                }
            }

            for (int j = 0; j < network.NUM_DEPTH; j++)
            {
                for (int i = 0; i < network.NUM_DEPTH; i++)
                {
                    if (i < network.NUM_DEPTH / 2)
                    {
                        weights1[j].Add((0.05d * rng.NextDouble()));
                    }
                    else
                    {
                        weights1[j].Add(0d);
                    }

                    if (i > network.NUM_DEPTH / 2)
                    {
                        weights2[j].Add((0.05d * rng.NextDouble()));
                    }
                    else
                    {
                        weights2[j].Add(0d);
                    }
                }
            }

            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                if (i < network.NUM_DEPTH / 2)
                {
                    network.setHidOutNeuron(i, 1, Neuron.actFuncType.logistic, input1, weights1[i], 0d);
                }
                else {
                    network.setHidOutNeuron(i, 1, Neuron.actFuncType.logistic, input2, weights2[i], 0d);
                }
            }

            //Output Neurons
            List<List<int>> input3 = new List<List<int>>();
            List<double> weights3 = new List<double>();
            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                input3.Add(new List<int>());
                input3[i].Add(i);
                input3[i].Add(1);
                weights3.Add(0.001d * rng.NextDouble());
            }
            network.setHidOutNeuron(0, 2, Neuron.actFuncType.logistic, input3, weights3, 0d);

            //Correct/Expected Output #2
            List<double> correctAnswer = new List<double>();
            correctAnswer.Add((110.92 - data.Min()) / (data.Max() - data.Min()));

            //Run
            network.train();

            Console.WriteLine("Forward & Back Prog Loop Ended, Square Error did not decrease");

            List<List<double>> inputHistory = new List<List<double>>();

            for (int j = 0; j < numTestEntries; j++)
            {
                List<double> currentInputs = new List<double>();
                for (int i = 0; i < network.NUM_DEPTH; i++)
                {
                    currentInputs.Add(network.nNarray[i, 0].postActOutput);
                }
                double newIn1 = network.nNarray[0, 2].postActOutput;
                currentInputs[network.NUM_DEPTH / 2] = newIn1;
                currentInputs.Remove(currentInputs[0]);
                currentInputs.Add(normData2[normData2.Count - numTestEntries]);

                for (int i = 0; i < network.NUM_DEPTH; i++)
                {
                    network.nNarray[i, 0].postActOutput = currentInputs[i];
                }

                inputHistory.Add(currentInputs);

                forwardPropNet(network);

                double chkAnsHalfSqError = 0f;

                if (j < correctAnswer.Count)
                {
                    chkAnsHalfSqError = 0.5 * Math.Pow(newIn1 - correctAnswer[j], 2);
                }

                Console.WriteLine("Prediction " + j + ": " + newIn1 + "(" + chkAnsHalfSqError + ")");
            }

            Console.WriteLine("Run Complete");
        }

        //Replicating Experiment in Paper using Scalar NN
        static void setPaperCase()
        {
            //Total Square Error: Learn Rate, Layer 1 Weight Mean, Layer 2 Weight Mean
            //0.438: 0.01, 0.5, 0.5
            //0.425: 0.01, 0.5, 0.05
            //0.453: 0.01, 0.5, 0.005
            //3.218: 0.01, 0.05, 0.05
            //2.85: 0.01, 0.1, 0.05
            //0.475: 0.001, 0.5, 0.05
            //0.491: 0.1, 0.5, 0.05

            //Setup
            Network network = new Network();
            network.setNumNeuronsEachLayer("194,10,50");
            network.learnRate = 0.01d;
            const int numTrainEntries = 194;
            const int numTestEntries = 50;
            const int numValidateEntries = 50;

            //Take in Data
            List<double> data = DataReader.readDoubleColumn(csvPath + "\\paperData.csv", 2);
            network.resetNetwork(data.Count - numTestEntries - numValidateEntries, 3);

            //Normalization
            List<double> normData = new List<double>();
            for (int i = 0; i < data.Count; i++)
            {
                normData.Add(0.5 * (data[i] - data.Min()) / (data.Max() - data.Min()));
            }

            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                if (i < numTestEntries)
                {
                    network.expOut[i] = normData[numTrainEntries + i];
                }
                else
                {
                    network.expOut[i] = 0;
                }
            }

            //Input Neurons
            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                network.setInputNeuron(i, normData[i]);
            }

            //Hidden Neurons
            List<List<int>> inputNeurons = new List<List<int>>();
            List<List<double>> weights = new List<List<double>>();
            List<double> noneWeights = Enumerable.Repeat(0d, network.NUM_DEPTH).ToList();

            Random rng = new Random();

            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                inputNeurons.Add(new List<int>());

                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int j = 0; j < network.NUM_DEPTH; j++)
            {
                weights.Add(new List<double>());

                for (int i = 0; i < network.NUM_DEPTH; i++)
                {
                    weights[j].Add(0.5d * rng.NextDouble());
                }
            }

            for (int i = 0; i < 10; i++) //10 HIDDEN NEURONS
            {
                network.setHidOutNeuron(i, 1, Neuron.actFuncType.logistic, inputNeurons, weights[i], 0d);
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();
            List<List<double>> outNweights = new List<List<double>>();

            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                if (i < 10)
                {
                    outInputNeurons.Add(new List<int>());
                    outInputNeurons[i].Add(i);
                    outInputNeurons[i].Add(1);
                }
                else
                {
                    outInputNeurons.Add(new List<int>());
                    outInputNeurons[i].Add(0);
                    outInputNeurons[i].Add(1);
                }
            }

            for (int j = 0; j < numTrainEntries; j++)
            {
                outNweights.Add(new List<double>());
                for (int i = 0; i < 10; i++)
                {
                    outNweights[j].Add(0.05d * rng.NextDouble());
                }
            }

            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                if (i < numTrainEntries)
                {
                    network.setHidOutNeuron(i, 2, Neuron.actFuncType.logistic, outInputNeurons, weights[i], 0d);
                }
                else
                {
                    network.setHidOutNeuron(i, 2, Neuron.actFuncType.logistic, outInputNeurons, noneWeights, 0d);
                }
            }

            //Run
            network.train();

            List<double> newInputs = new List<double>();
            
            for (int i = 0; i < network.NUM_DEPTH; i++)
            {
                newInputs.Add(normData[numTestEntries + i]);
                network.nNarray[i, 0].postActOutput = newInputs[i];
            }

            //Run #2
            forwardPropNet(network);

            //Print Output
            Console.WriteLine("SQ ERROR: " + network.calSqError());
        }

        //Vector NN Test
        static void setVecTestCase()
        {
            //Setup
            vNetwork vNet = new vNetwork();
            vNet.resetNetwork(2,3);
            vNet.setNumNeuronsEachLayer("2,2,2");
            vNet.learnRate = 0.1d;

            //Expected Outputs
            vNet.expOut.Add(new List<double>(new double[] { 0.5, 0.6 }));
            vNet.expOut.Add(new List<double>(new double[] { 0.4, 0.5 }));

            //Input Neurons
            vNet.setInputNeuron(0, new List<double>(new double[] { 0.1, 0.2 }));
            vNet.setInputNeuron(1, new List<double>(new double[] { 0.4, 0.8 }));

            //Hidden Neurons
            List<List<int>> inputNeurons = new List<List<int>>();
            inputNeurons.Add(new List<int>(new int[] { 0, 0 }));
            inputNeurons.Add(new List<int>(new int[] { 1, 0 }));
            List<List<double>> weights = new List<List<double>>();
            weights.Add(new List<double>(new double[] { 0.4, 0.1 }));
            weights.Add(new List<double>(new double[] { 0.6, 0.3 }));
            vNet.setHidOutNeuron(0, 1, vNeuron.actFuncType.logistic, inputNeurons, 1d, new List<double>(new double[] { 0d, 0d }));

            weights = new List<List<double>>();
            weights.Add(new List<double>(new double[] { 0.7, 0.9 }));
            weights.Add(new List<double>(new double[] { 0.3, 0.8 }));
            vNet.setHidOutNeuron(1, 1, vNeuron.actFuncType.logistic, inputNeurons, 1d, new List<double>(new double[] { 0d, 0d }));

            //Output Neurons
            inputNeurons = new List<List<int>>();
            inputNeurons.Add(new List<int>(new int[] { 0, 1 }));
            inputNeurons.Add(new List<int>(new int[] { 1, 1 }));
            weights = new List<List<double>>();
            weights.Add(new List<double>(new double[] { 0.1, 0.3 }));
            weights.Add(new List<double>(new double[] { 0.2, 0.4 }));
            vNet.setHidOutNeuron(0, 2, vNeuron.actFuncType.logistic, inputNeurons, 1d, new List<double>(new double[] { 0d, 0d }));

            weights = new List<List<double>>();
            weights.Add(new List<double>(new double[] { 0.5, 0.7 }));
            weights.Add(new List<double>(new double[] { 0.6, 0.8 }));
            vNet.setHidOutNeuron(1, 2, vNeuron.actFuncType.logistic, inputNeurons, 1d, new List<double>(new double[] { 0d, 0d }));

            //Run
            vNet.train();
            Console.WriteLine("");
        }

        //Vector NN Run with USD/JPY and CNY/JPY
        static void set2dExRateCase()
        {
            //Read Data and Normalize
            List<double> data1 = DataReader.readDoubleColumn(csvPath + "\\usdjpycnyjpyndnk.csv", 2);
            List<double> normData1 = new List<double>();
            for (int i = 0; i < data1.Count; i++)
            {
                normData1.Add((data1[i] - data1.Min()) / (data1.Max() - data1.Min()));
            }

            List<double> data2 = DataReader.readDoubleColumn(csvPath + "\\usdjpycnyjpyndnk.csv", 3);
            List<double> normData2 = new List<double>();
            for (int i = 0; i < data2.Count; i++)
            {
                normData2.Add((data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
            }

            //Net Setup
            vNetwork vNet = new vNetwork();
            int numExpVals = 10;
            int layer1 = normData1.Count - numExpVals;
            vNet.setNumNeuronsEachLayer(layer1.ToString() + ",100," + numExpVals.ToString());
            vNet.resetNetwork(layer1, 3);
            vNet.learnRate = 0.01d;

            //Expected Values
            List<List<double>> expVals = new List<List<double>>();
            for (int i = 0; i < numExpVals; i++)
            {
                expVals.Add(new List<double>());
                expVals[i].Add(normData1[normData1.Count - numExpVals + i]);
                expVals[i].Add(normData2[normData2.Count - numExpVals + i]);
            }
            vNet.expOut = expVals;

            //Input Neurons
            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                vNet.setInputNeuron(i, new List<double>(new double[] { normData1[i], normData2[i] }));
            }

            //Hidden Neurons
            List<List<int>> inputNeurons = new List<List<int>>();
            List<List<List<double>>> weights = new List<List<List<double>>>();

            Random rng = new Random();

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                inputNeurons.Add(new List<int>());

                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < vNet.numNeuronsEachLayer[1])
                {
                    vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 0.00005, new List<double>(new double[] { 0d, 0d }));
                }
                else
                {
                    vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 0d, new List<double>(new double[] { 0d, 0d }));
                }
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();
            List<List<List<double>>> outNweights = new List<List<List<double>>>();

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < numExpVals)
                {
                    outInputNeurons.Add(new List<int>());
                    outInputNeurons[i].Add(i);
                    outInputNeurons[i].Add(1);
                }
            }

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < numExpVals)
                {
                    vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, outInputNeurons, 0.05, new List<double>(new double[] { 0d, 0d }));
                }
            }

            //Run
            vNet.train();

            //New Inputs
            List<List<double>> newInputs = new List<List<double>>();
            for (int i = 0; i < normData1.Count - numExpVals; i++)
            {
                newInputs.Add(new List<double>());
                newInputs[i].Add(normData1[i + numExpVals]);
                newInputs[i].Add(normData2[i + numExpVals]);
            }
            int index = 0;
            foreach (List<double> input in newInputs)
            {
                vNet.nNarray[index, 0].postActOutput = newInputs[index];
                index++;
            }

            //Run #2
            vForwardPropNet(vNet);

            //Print Output
            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                int j = 1;
                foreach (double vec in vNet.nNarray[i, 2].postActOutput)
                {

                    Console.WriteLine("Neuron #" + i + ": " + j + "d : " + vNet.nNarray[i, 2].postActOutput[0]);
                    Console.WriteLine("Neuron #" + i + ": " + j + "d : " + vNet.nNarray[i, 2].postActOutput[1]);
                    j++;
                }
            }

            Console.WriteLine("");
        }

        //Vector NN Run with USD/JPY, CNY/JPY and NASDAQ Composite
        static void set3dExRateCase()
        {
            //Read Data
            string filePath = csvPath + "\\usdjpycnyjpyndnk.csv";

            //Read in Data
            List<double> data1 = DataReader.readDoubleColumn(filePath, 2);
            List<double> normData1 = new List<double>();
            List<double> data2 = DataReader.readDoubleColumn(filePath, 3);
            List<double> normData2 = new List<double>();
            List<double> data3 = DataReader.readDoubleColumn(filePath, 4);
            List<double> normData3 = new List<double>();
            int numEntries = Math.Max(data1.Count, Math.Max(data2.Count, data3.Count));

            //Normalization
            for (int i = 0; i < numEntries; i++)
            {
                if (data1[i] != 0d && data2[i] != 0d && data3[i] != 0d)
                {
                    normData1.Add(0.5 * (data1[i] - data1.Min()) / (data1.Max() - data1.Min()));
                    normData2.Add(0.5 * (data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
                    normData3.Add(0.5 * (data3[i] - data3.Min()) / (data3.Max() - data3.Min()));
                }
            }

            //Net Setup
            vNetwork vNet = new vNetwork();
            int numExpVals = 10;
            int layer1 = normData1.Count - numExpVals;
            vNet.setNumNeuronsEachLayer(layer1.ToString() + ",100," + numExpVals.ToString());
            vNet.resetNetwork(layer1, 3);
            vNet.learnRate = 0.1d;

            //Expected Values
            List<List<double>> expVals = new List<List<double>>();
            for (int i = 0; i < normData1.Count; i++)
            {
                expVals.Add(new List<double>());
                if (i < numExpVals)
                {
                    expVals[i].Add(normData1[normData1.Count - numExpVals - i - 1]);
                    expVals[i].Add(normData2[normData2.Count - numExpVals - i - 1]);
                    expVals[i].Add(normData3[normData3.Count - numExpVals - i - 1]);
                }
                else
                {
                    expVals[i].Add(0d);
                    expVals[i].Add(0d);
                    expVals[i].Add(0d);
                }
            }
            vNet.expOut = expVals;

            //Input Neurons
            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                vNet.setInputNeuron(i, new List<double>(new double[] { normData1[i], normData2[i], normData3[i] }));
            }

            //Hidden Neurons
            List<List<int>> inputNeurons = new List<List<int>>();
            List<List<List<double>>> weights = new List<List<List<double>>>();
            //List<List<double>> noneWeights = new List<List<double>>();

            Random rng = new Random();

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                inputNeurons.Add(new List<int>());

                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < vNet.numNeuronsEachLayer[1])
                {
                    vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 0.00005, new List<double>(new double[] { 0d, 0d, 0d }));
                }
                else
                {
                    vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 0d, new List<double>(new double[] { 0d, 0d, 0d }));
                }
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();
            List<List<List<double>>> outNweights = new List<List<List<double>>>();

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < numExpVals)
                {
                    outInputNeurons.Add(new List<int>());
                    outInputNeurons[i].Add(i);
                    outInputNeurons[i].Add(1);
                }
                else
                {
                    outInputNeurons.Add(new List<int>());
                    outInputNeurons[i].Add(0);
                    outInputNeurons[i].Add(1);
                }
            }

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < numExpVals)
                {
                    vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, outInputNeurons, 0.05, new List<double>(new double[] { 0d, 0d, 0d }));
                }
                else
                {
                    vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, outInputNeurons, 0d, new List<double>(new double[] { 0d, 0d, 0d }));
                }
            }

            vNet.train();

            //New Inputs
            List<List<double>> newInputs = new List<List<double>>();
            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                newInputs.Add(new List<double>());
                newInputs[i].Add(normData1[numExpVals + i]);
                newInputs[i].Add(normData2[numExpVals + i]);
                newInputs[i].Add(normData3[numExpVals + i]);
            }
            int index = 0;
            foreach (List<double> input in newInputs)
            {
                vNet.nNarray[index, 0].postActOutput = newInputs[index];
                index++;
            }

            //Run #2
            vForwardPropNet(vNet);

            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                int j = 1;
                foreach (double vec in vNet.nNarray[i, 2].postActOutput)
                {

                    Console.WriteLine("Neuron #" + i + ": " + j + "d : " + vec);
                    j++;
                }
            }

            Console.WriteLine("");
        }

        //Vector NN Run with USD/JPY, CNY/JPY, NASDAQ Composite and Nikkei 225
        static void set4dPaperCase()
        {
            //Read Data
            string filePath = csvPath + "\\paperData.csv";

            //Read in Data
            List<double> data1 = DataReader.readDoubleColumn(filePath, 1);
            List<double> normData1 = new List<double>();
            List<double> data2 = DataReader.readDoubleColumn(filePath, 2);
            List<double> normData2 = new List<double>();
            List<double> data3 = DataReader.readDoubleColumn(filePath, 3);
            List<double> normData3 = new List<double>();
            List<double> data4 = DataReader.readDoubleColumn(filePath, 4);
            List<double> normData4 = new List<double>();
            //int numEntries = Math.Max(data1.Count, Math.Max(data2.Count, data3.Count));

            //Normalization
            for (int i = 0; i < data1.Count; i++)
            {
                //if (data1[i] != 0d && data2[i] != 0d && data3[i] != 0d)
                //{
                normData1.Add(0.5 * (data1[i] - data1.Min()) / (data1.Max() - data1.Min()));
                normData2.Add(0.5 * (data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
                normData3.Add(0.5 * (data3[i] - data3.Min()) / (data3.Max() - data3.Min()));
                normData4.Add(0.5 * (data4[i] - data4.Min()) / (data4.Max() - data4.Min()));
                //}
            }

            //Net Setup
            vNetwork vNet = new vNetwork();
            int numTrainInputs = normData1.Count - 100;
            const int numTrainOutputs = 50;
            const int numValidateOutputs = 50;
            int layer1 = numTrainInputs;
            vNet.setNumNeuronsEachLayer(layer1.ToString() + "," + layer1.ToString() + "," + numTrainOutputs.ToString());
            vNet.resetNetwork(layer1, 3);
            vNet.learnRate = 0.1d;

            //Expected Values
            List<List<double>> expVals = new List<List<double>>();
            for (int i = 0; i < normData1.Count; i++)
            {
                expVals.Add(new List<double>());
                if (i < numTrainOutputs)
                {
                    expVals[i].Add(normData1[numTrainInputs + i]);
                    expVals[i].Add(normData2[numTrainInputs + i]);
                    expVals[i].Add(normData3[numTrainInputs + i]);
                    expVals[i].Add(normData4[numTrainInputs + i]);
                }
                else
                {
                    expVals[i].Add(0d);
                    expVals[i].Add(0d);
                    expVals[i].Add(0d);
                    expVals[i].Add(0d);
                }
            }
            vNet.expOut = expVals;

            //Input Neurons
            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                vNet.setInputNeuron(i, new List<double>(new double[] { normData1[i], normData2[i], normData3[i], normData4[i] }));
            }

            //Hidden Neurons
            List<List<int>> inputNeurons = new List<List<int>>();
            List<List<List<double>>> weights = new List<List<List<double>>>();
            //List<List<double>> noneWeights = new List<List<double>>();

            Random rng = new Random();

            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());

                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < vNet.numNeuronsEachLayer[1])
                {
                    vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 1d / (double)vNet.numNeuronsEachLayer[0], new List<double>(new double[] { 0d, 0d, 0d, 0d }));
                }
                else
                {
                    vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 0d, new List<double>(new double[] { 0d, 0d, 0d, 0d }));
                }
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();
            List<List<List<double>>> outNweights = new List<List<List<double>>>();

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < vNet.numNeuronsEachLayer[1])
                {
                    outInputNeurons.Add(new List<int>());
                    outInputNeurons[i].Add(i);
                    outInputNeurons[i].Add(1);
                }
                else
                {
                    outInputNeurons.Add(new List<int>());
                    outInputNeurons[i].Add(0);
                    outInputNeurons[i].Add(1);
                }
            }

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < numTrainOutputs)
                {
                    vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, outInputNeurons, 1d / (double)vNet.numNeuronsEachLayer[1], new List<double>(new double[] { 0d, 0d, 0d, 0d }));
                }
                else
                {
                    vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, outInputNeurons, 0d, new List<double>(new double[] { 0d, 0d, 0d, 0d }));
                }
            }

            //Run
            vNet.train();

            //New Inputs
            List<List<double>> newInputs = new List<List<double>>();
            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                newInputs.Add(new List<double>());
                newInputs[i].Add(normData1[numTrainOutputs + i]);
                newInputs[i].Add(normData2[numTrainOutputs + i]);
                newInputs[i].Add(normData3[numTrainOutputs + i]);
                newInputs[i].Add(normData4[numTrainOutputs + i]);
            }
            int index = 0;
            foreach (List<double> input in newInputs)
            {
                vNet.nNarray[index, 0].inputs[0] = newInputs[index];
                index++;
            }

            //Run #2
            vForwardPropNet(vNet);

            //Correct Outputs used for Square Error Calculation
            List<List<double>> correctOut = new List<List<double>>();
            for (int i = 0; i < numValidateOutputs; i++)
            {
                correctOut.Add(new List<double>());
                //for(int j = 0; j < 4; j++)
                //{
                correctOut[i].Add(normData1[i + numTrainInputs + numTrainOutputs]);
                correctOut[i].Add(normData2[i + numTrainInputs + numTrainOutputs]);
                correctOut[i].Add(normData3[i + numTrainInputs + numTrainOutputs]);
                correctOut[i].Add(normData4[i + numTrainInputs + numTrainOutputs]);
                //}
            }

            vNet.expOut = correctOut;
            //for (int i = 0; i < 10/*vNet.numNeuronsEachLayer[2]*/; i++)
            //{
            //    int j = 1;
            //    foreach (double d in vNet.nNarray[i, 2].postActOutput)
            //    {

            //        Console.WriteLine("Neuron #" + i + ": " + j + "d : " + d);
            //        double sqError = Math.Pow(d - correctOut[j - 1][i], 2);
            //        j++;
            //    }
            //}

            //Write Output
            Console.WriteLine("sqError: " + vNet.calHalfSqError() * 2);

            Console.WriteLine("");
        }

        //Vector NN Run with 4 Columns and Training, Test and Validation
        static void set4dPaperConFeedCase()
        {
            //Read Data
            string filePath = csvPath + "\\paperData.csv";

            //Read in Data
            List<double> data1 = DataReader.readDoubleColumn(filePath, 1);
            List<double> normData1 = new List<double>();
            List<double> data2 = DataReader.readDoubleColumn(filePath, 2);
            List<double> normData2 = new List<double>();
            List<double> data3 = DataReader.readDoubleColumn(filePath, 3);
            List<double> normData3 = new List<double>();
            List<double> data4 = DataReader.readDoubleColumn(filePath, 4);
            List<double> normData4 = new List<double>();
            //int numEntries = Math.Max(data1.Count, Math.Max(data2.Count, data3.Count));

            //Normalization
            for (int i = 0; i < data1.Count; i++)
            {
                //if (data1[i] != 0d && data2[i] != 0d && data3[i] != 0d)
                //{
                normData1.Add(0.5 * (data1[i] - data1.Min()) / (data1.Max() - data1.Min()));
                normData2.Add(0.5 * (data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
                normData3.Add(0.5 * (data3[i] - data3.Min()) / (data3.Max() - data3.Min()));
                normData4.Add(0.5 * (data4[i] - data4.Min()) / (data4.Max() - data4.Min()));
                //}
            }

            //Net Setup
            vNetwork vNet = new vNetwork();
            int numTrainInputs = normData1.Count - 100;
            const int numTrainOutputs = 50;
            const int numValidateOutputs = 50;
            int layer1 = numTrainInputs;
            vNet.setNumNeuronsEachLayer(layer1.ToString() + ",200,1");
            vNet.resetNetwork(200, 3);
            vNet.learnRate = 1d;

            //Input Neurons
            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                vNet.setInputNeuron(i, new List<double>(new double[] { normData1[i], normData2[i], normData3[i], normData4[i] }));
            }

            //Hidden Neurons
            List<List<int>> inputNeurons = new List<List<int>>();
            List<List<List<double>>> weights = new List<List<List<double>>>();
            //List<List<double>> noneWeights = new List<List<double>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());

                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < vNet.numNeuronsEachLayer[1])
                {
                    vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 1d / (double)vNet.numNeuronsEachLayer[0], new List<double>(new double[] { 0d, 0d, 0d, 0d }));
                }
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();
            List<List<List<double>>> outNweights = new List<List<List<double>>>();

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < vNet.numNeuronsEachLayer[1])
                {
                    outInputNeurons.Add(new List<int>());
                    outInputNeurons[i].Add(i);
                    outInputNeurons[i].Add(1);
                }
            }

            for (int i = 0; i < vNet.NUM_DEPTH; i++)
            {
                if (i < 1)
                {
                    vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, outInputNeurons, 1d / (double)vNet.numNeuronsEachLayer[1], new List<double>(new double[] { 0d, 0d, 0d, 0d }));
                }
            }

            //Training
            //int numTrain = 1;
            for (int j = 0; j < numTrainOutputs; j++) {
                //Expected Values
                List<List<double>> expVals = new List<List<double>>();
                for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
                {
                    vNet.nNarray[i, 0].inputs[0][0] = normData1[i + j];
                    vNet.nNarray[i, 0].inputs[0][1] = normData2[i + j];
                    vNet.nNarray[i, 0].inputs[0][2] = normData3[i + j];
                    vNet.nNarray[i, 0].inputs[0][3] = normData4[i + j];

                    expVals.Add(new List<double>());
                    expVals[i].Add(normData1[numTrainInputs + j]);
                    expVals[i].Add(normData2[numTrainInputs + j]);
                    expVals[i].Add(normData3[numTrainInputs + j]);
                    expVals[i].Add(normData4[numTrainInputs + j]);
                }
                vNet.expOut = expVals;

                //Run
                vNet.train();

                Console.WriteLine("Output " + j + " trained");
            }

            //"Validation"
            for (int j = 0; j < numValidateOutputs; j++)
            {
                List<List<double>> newInputs = new List<List<double>>();
                for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
                {
                    newInputs.Add(new List<double>());
                    newInputs[i].Add(normData1[numTrainOutputs + i + j]);
                    newInputs[i].Add(normData2[numTrainOutputs + i + j]);
                    newInputs[i].Add(normData3[numTrainOutputs + i + j]);
                    newInputs[i].Add(normData4[numTrainOutputs + i + j]);
                }
                int index = 0;
                foreach (List<double> input in newInputs)
                {
                    vNet.nNarray[index, 0].inputs[0] = newInputs[index];
                    index++;
                }

                //Run
                vForwardPropNet(vNet);

                //Correct Output for Square Error Calculation
                List<List<double>> correctOut = new List<List<double>>();
                for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
                {
                    correctOut.Add(new List<double>());
                    correctOut[i].Add(normData1[i + numTrainInputs + numTrainOutputs]);
                    correctOut[i].Add(normData2[i + numTrainInputs + numTrainOutputs]);
                    correctOut[i].Add(normData3[i + numTrainInputs + numTrainOutputs]);
                    correctOut[i].Add(normData4[i + numTrainInputs + numTrainOutputs]);
                }

                vNet.expOut = correctOut;
                
                Console.WriteLine("sqError: " + vNet.calHalfSqError() * 2);
            }
            Console.WriteLine("");
        }

        //Vector NN Run with USD/JPY, Predict Day T+1 by using T and T-1
        static void setPaper_predictTomorrowByToday_case()
        {
            //Read Data
            string filePath = csvPath + "\\paperData.csv";

            List<double> data1 = DataReader.readDoubleColumn(filePath, 1);
            List<double> normData1 = new List<double>();

            //Extend Horizontally if needed
            //List<double> data2 = DataReader.readDoubleColumn(filePath, 2);
            //List<double> normData2 = new List<double>();
            //List<double> data3 = DataReader.readDoubleColumn(filePath, 3);
            //List<double> normData3 = new List<double>();
            //List<double> data4 = DataReader.readDoubleColumn(filePath, 4);
            //List<double> normData4 = new List<double>();

            for (int i = 0; i < data1.Count; i++)
            {
                if (data1[i] != 0d /*&& data2[i] != 0d && data3[i] != 0d && data4[i] != 0d*/)
                {
                    normData1.Add(0.5 * (data1[i] - data1.Min()) / (data1.Max() - data1.Min()));
                    //normData2.Add(0.5 * (data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
                    //normData3.Add(0.5 * (data3[i] - data3.Min()) / (data3.Max() - data3.Min()));
                    //normData4.Add(0.5 * (data4[i] - data4.Min()) / (data4.Max() - data4.Min()));
                }
            }

            //Net Setup
            vNetwork vNet = new vNetwork();

            int numValidEntries = normData1.Count();
            int numDaysInFutureToPredict = 1;
            int depth = numValidEntries - numDaysInFutureToPredict;
            //int numTrainInputs = normData1.Count - 100;
            //const int numTrainOutputs = 50;
            //const int numValidateOutputs = 50;

            vNet.setNumNeuronsEachLayer(depth + ",200," + depth);
            vNet.resetNetwork(depth, 3);
            vNet.learnRate = 1d;
            vNet.bLearnRate = 0.5d;

            //Input Neurons
            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                vNet.setInputNeuron(i, new List<double>(new double[] { normData1[i] }));

            }
            //vNet.setInputNeuron(1, new List<double>(new double[] { normData2[0] }));
            //vNet.setInputNeuron(2, new List<double>(new double[] { normData3[0] }));
            //vNet.setInputNeuron(3, new List<double>(new double[] { normData4[0] }));

            //Hidden Neurons L1
            List<List<int>> inputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 1d / (double)vNet.numNeuronsEachLayer[0], new List<double>(new double[] { 0d }));
            }

            /*//Hidden Neurons L2
            inputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(1);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, inputNeurons, 1d / (double)vNet.numNeuronsEachLayer[0], new List<double>(new double[] { 0d }));
            }*/

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(1);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, outInputNeurons, 1d / (double)vNet.numNeuronsEachLayer[1], new List<double>(new double[] { 0d }));
            }

            //Train Iteriations
            //double bestSqE = double.MaxValue;
            //List<List<List<List<List<double>>>>> bestWeights = new List<List<List<List<List<double>>>>>();

            //for (int run = 0; run < numTrainInputs + numTrainOutputs + numValidateOutputs - 1; run++) {
                //Input Neurons
                //vNet.nNarray[0, 0].inputs.Clear();
                //vNet.nNarray[0, 0].inputs.Add(new List<double>(new double[] { normData1[run] }));
                //vNet.nNarray[1, 0].inputs.Clear();
                //vNet.nNarray[1, 0].inputs.Add(new List<double>(new double[] { normData2[run] }));
                //vNet.nNarray[2, 0].inputs.Clear();
                //vNet.nNarray[2, 0].inputs.Add(new List<double>(new double[] { normData3[run] }));
                //vNet.nNarray[3, 0].inputs.Clear();
                //vNet.nNarray[3, 0].inputs.Add(new List<double>(new double[] { normData4[run] }));

            //Expected Values
            vNet.expOut.Clear();
            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                vNet.expOut.Add(new List<double>(new double[] { normData1[i + numDaysInFutureToPredict] }));
            }
                //vNet.expOut.Add(new List<double>(new double[] { normData2[run + 1] }));
                //vNet.expOut.Add(new List<double>(new double[] { normData3[run + 1] }));
                //vNet.expOut.Add(new List<double>(new double[] { normData4[run + 1] }));
            
            vNet.train();

            double sqE = (vNet.calHalfSqError() * 2);
            Console.WriteLine("Total Square Error: " + sqE);
            //if (double.IsNaN(sqE))
            //{
            //Console.Write("NaN result detected!!!");
            //}

            /*if(sqE < bestSqE)
            {
                bestSqE = sqE;
                bestWeights = vCopyWeights(vNet.nNarray, vNet.NUM_DEPTH, vNet.NUM_LAYERS);
            }*/
            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                vNet.nNarray[i, 0].inputs.Clear();
                vNet.nNarray[i, 0].inputs.Add(new List<double>(new double[] { normData1[i + numDaysInFutureToPredict] }));
            }

                //vNet.nNarray[1, 0].inputs.Clear();
                //vNet.nNarray[1, 0].inputs.Add(new List<double>(new double[] { normData2[run + 1] }));
                //vNet.nNarray[2, 0].inputs.Clear();
                //vNet.nNarray[2, 0].inputs.Add(new List<double>(new double[] { normData3[run + 1] }));
                //vNet.nNarray[3, 0].inputs.Clear();
                //vNet.nNarray[3, 0].inputs.Add(new List<double>(new double[] { normData4[run + 1] }));

            //Run #2
            vForwardPropNet(vNet);
            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                Console.WriteLine(vNet.nNarray[i, 2].postActOutput[0]/* + "," + vNet.nNarray[1, 2].postActOutput[0] + "," + vNet.nNarray[2, 2].postActOutput[0] + "," + vNet.nNarray[3, 2].postActOutput[0]*/);
            }
            //Console.WriteLine("Train day " + run + " completed; Sq Error = " + sqE + " Best Sq Error = " + bestSqE);
        //}

            //Test and Validate Iteriations
            //vApplyWeights(bestWeights, vNet.nNarray, vNet.NUM_DEPTH, vNet.NUM_LAYERS);

            //List<double> inputs = new List<double>();
            //inputs.Add(normData1.Last());
            //inputs.Add(normData2.Last());
            //inputs.Add(normData3.Last());
            //inputs.Add(normData4.Last());

            /*for (int run = 0; run < 50; run++) //numTrainOutputs + numValidateOutputs - 1
            {
                //Input Neurons
                vNet.nNarray[0, 0].inputs.Clear();
                vNet.nNarray[0, 0].inputs.Add(new List<double>(new double[] { inputs[0] }));
                vNet.nNarray[1, 0].inputs.Clear();
                vNet.nNarray[1, 0].inputs.Add(new List<double>(new double[] { inputs[1] }));
                vNet.nNarray[2, 0].inputs.Clear();
                vNet.nNarray[2, 0].inputs.Add(new List<double>(new double[] { inputs[2] }));
                vNet.nNarray[3, 0].inputs.Clear();
                vNet.nNarray[3, 0].inputs.Add(new List<double>(new double[] { inputs[3] }));

                //Expected Values
                vNet.expOut.Clear();
                vNet.expOut.Add(new List<double>(new double[] { normData1[run + 1 + numTrainInputs] }));
                vNet.expOut.Add(new List<double>(new double[] { normData2[run + 1 + numTrainInputs] }));
                vNet.expOut.Add(new List<double>(new double[] { normData3[run + 1 + numTrainInputs] }));
                vNet.expOut.Add(new List<double>(new double[] { normData4[run + 1 + numTrainInputs] }));

                vForwardPropNet(vNet);

                Console.WriteLine(run + "," + vNet.nNarray[0, 2].postActOutput[0]); // + "," + vNet.nNarray[1, 2].postActOutput[0] + "," + vNet.nNarray[2, 2].postActOutput[0] + "," + vNet.nNarray[3, 2].postActOutput[0]
            //double sqE = (vNet.calHalfSqError() * 2);
            //Console.WriteLine("Validate run " + run + " completed; Sq Error = " + sqE);
            //Console.WriteLine("Output: " + vNet.nNarray[0, 2].postActOutput[0]);

            inputs[0] = vNet.nNarray[0, 2].postActOutput[0];
                inputs[1] = vNet.nNarray[1, 2].postActOutput[0];
                inputs[2] = vNet.nNarray[2, 2].postActOutput[0];
                inputs[3] = vNet.nNarray[3, 2].postActOutput[0];
            }*/

            Console.WriteLine("");
        }

        //Vector NN Test Case (similar to setNNcase3())
        static void setCase3Vec()
        {
            //Net Setup
            vNetwork vnet = new vNetwork();
            vnet.setNumNeuronsEachLayer("2,2,2");
            vnet.resetNetwork(2, 3);
            vnet.learnRate = 0.5d;

            //Expected Values
            List<List<double>> expVals = new List<List<double>>();
            expVals.Add(new List<double>());
            expVals[0].Add(0.01 * 0.5);
            expVals.Add(new List<double>());
            expVals[1].Add(0.99 * 0.5);
            vnet.expOut = expVals;

            //Input Neurons
            vnet.setInputNeuron(0, new List<double>(new double[] { 0.05 }));
            vnet.setInputNeuron(1, new List<double>(new double[] { 0.1 }));

            //Hidden Neurons
            List<List<int>> inputNeurons = new List<List<int>>();
            inputNeurons.Add(new List<int>(new int[] { 0, 0 }));
            inputNeurons.Add(new List<int>(new int[] { 1, 0 }));

            List<List<List<double>>> weights = new List<List<List<double>>>();

            List<double> element = new List<double>(new double[] { 0.15 });
            List<List<double>> matrix = new List<List<double>>();
            matrix.Add(element);
            weights.Add(matrix);

            List<double> element2 = new List<double>(new double[] { 0.2 });
            List<List<double>> matrix2 = new List<List<double>>();
            matrix2.Add(element2);
            weights.Add(matrix2);

            vnet.setHidOutNeuron(0, 1, vNeuron.actFuncType.logistic, inputNeurons, weights, new List<double>(new double[] { 0.35 }));


            weights = new List<List<List<double>>>();

            element = new List<double>(new double[] { 0.25 });
            matrix = new List<List<double>>();
            matrix.Add(element);
            weights.Add(matrix);

            element2 = new List<double>(new double[] { 0.3 });
            matrix2 = new List<List<double>>();
            matrix2.Add(element2);
            weights.Add(matrix2);

            vnet.setHidOutNeuron(1, 1, vNeuron.actFuncType.logistic, inputNeurons, weights, new List<double>(new double[] { 0.35 }));

            //Output Neurons
            inputNeurons = new List<List<int>>();
            inputNeurons.Add(new List<int>(new int[] { 0, 1 }));
            inputNeurons.Add(new List<int>(new int[] { 1, 1 }));

            weights = new List<List<List<double>>>();

            element = new List<double>(new double[] { 0.4 });
            matrix = new List<List<double>>();
            matrix.Add(element);
            weights.Add(matrix);

            element2 = new List<double>(new double[] { 0.45 });
            matrix2 = new List<List<double>>();
            matrix2.Add(element2);
            weights.Add(matrix2);

            vnet.setHidOutNeuron(0, 2, vNeuron.actFuncType.logistic, inputNeurons, weights, new List<double>(new double[] { 0.6 }));


            weights = new List<List<List<double>>>();

            element = new List<double>(new double[] { 0.5 });
            matrix = new List<List<double>>();
            matrix.Add(element);
            weights.Add(matrix);

            element2 = new List<double>(new double[] { 0.55 });
            matrix2 = new List<List<double>>();
            matrix2.Add(element2);
            weights.Add(matrix2);

            vnet.setHidOutNeuron(1, 2, vNeuron.actFuncType.logistic, inputNeurons, weights, new List<double>(new double[] { 0.6 }));

            //vForwardPropNet(vnet);
            //vBackPropNet(vnet);
            vnet.train();
        }

        //Vector NN Test Case (similar to setCase3Vec() and setNNcase3())
        static void setCase3Vec2d()
        {
            //Net Setup
            vNetwork vnet = new vNetwork();
            vnet.setNumNeuronsEachLayer("2,2,2");
            vnet.resetNetwork(2, 3);
            vnet.learnRate = 0.5d;

            //Expected Values
            List<List<double>> expVals = new List<List<double>>();
            expVals.Add(new List<double>());
            expVals[0].Add(0.01 * 0.5);
            expVals[0].Add(0.01 * 0.5);
            expVals.Add(new List<double>());
            expVals[1].Add(0.99 * 0.5);
            expVals[1].Add(0.99 * 0.5);
            vnet.expOut = expVals;

            //Input Neurons
            vnet.setInputNeuron(0, new List<double>(new double[] { 0.05, 0.05 }));
            vnet.setInputNeuron(1, new List<double>(new double[] { 0.1, 0.1 }));

            //Hidden Neurons
            List<List<int>> inputNeurons = new List<List<int>>();
            inputNeurons.Add(new List<int>(new int[] { 0, 0 }));
            inputNeurons.Add(new List<int>(new int[] { 1, 0 }));


            List<List<List<double>>> weights = new List<List<List<double>>>();

            List<double> element = new List<double>(new double[] { 0.15, 0.15 });
            List<double> elementP = new List<double>(new double[] { 0.15, 0.15 });
            List<List<double>> matrix = new List<List<double>>();
            matrix.Add(element);
            matrix.Add(elementP);
            weights.Add(matrix);

            List<double> element2 = new List<double>(new double[] { 0.2, 0.2 });
            List<double> element2P = new List<double>(new double[] { 0.2, 0.2 });
            List<List<double>> matrix2 = new List<List<double>>();
            matrix2.Add(element2);
            matrix2.Add(element2P);
            weights.Add(matrix2);

            vnet.setHidOutNeuron(0, 1, vNeuron.actFuncType.logistic, inputNeurons, weights, new List<double>(new double[] { 0.35, 0.35 }));


            weights = new List<List<List<double>>>();

            element = new List<double>(new double[] { 0.25, 0.25 });
            elementP = new List<double>(new double[] { 0.25, 0.25 });
            matrix = new List<List<double>>();
            matrix.Add(element);
            matrix.Add(elementP);
            weights.Add(matrix);

            element2 = new List<double>(new double[] { 0.3, 0.3 });
            element2 = new List<double>(new double[] { 0.3, 0.3 });
            matrix2 = new List<List<double>>();
            matrix2.Add(element2);
            matrix2.Add(element2P);
            weights.Add(matrix2);

            vnet.setHidOutNeuron(1, 1, vNeuron.actFuncType.logistic, inputNeurons, weights, new List<double>(new double[] { 0.35, 0.35 }));

            //Output Neurons
            inputNeurons = new List<List<int>>();
            inputNeurons.Add(new List<int>(new int[] { 0, 1 }));
            inputNeurons.Add(new List<int>(new int[] { 1, 1 }));

            weights = new List<List<List<double>>>();

            element = new List<double>(new double[] { 0.4, 0.4 });
            elementP = new List<double>(new double[] { 0.4, 0.4 });
            matrix = new List<List<double>>();
            matrix.Add(element);
            matrix.Add(elementP);
            weights.Add(matrix);

            element2 = new List<double>(new double[] { 0.45, 0.45 });
            element2P = new List<double>(new double[] { 0.45, 0.45 });
            matrix2 = new List<List<double>>();
            matrix2.Add(element2);
            matrix2.Add(element2P);
            weights.Add(matrix2);

            vnet.setHidOutNeuron(0, 2, vNeuron.actFuncType.logistic, inputNeurons, weights, new List<double>(new double[] { 0.6, 0.6 }));


            weights = new List<List<List<double>>>();

            element = new List<double>(new double[] { 0.5, 0.5 });
            elementP = new List<double>(new double[] { 0.5, 0.5 });
            matrix = new List<List<double>>();
            matrix.Add(element);
            matrix.Add(elementP);
            weights.Add(matrix);

            element2 = new List<double>(new double[] { 0.55, 0.55 });
            element2 = new List<double>(new double[] { 0.55, 0.55 });
            matrix2 = new List<List<double>>();
            matrix2.Add(element2);
            matrix2.Add(element2P);
            weights.Add(matrix2);

            vnet.setHidOutNeuron(1, 2, vNeuron.actFuncType.logistic, inputNeurons, weights, new List<double>(new double[] { 0.6, 0.6 }));

            //vForwardPropNet(vnet);
            //vBackPropNet(vnet);
            vnet.train();
        }

        //Vector NN Run with USD/JPY, CNY/JPY, NASDAQ, NIKKEI, Predict T+1 by using T-19, T-18, T-17 ... T
        static void setPaper_predictTomorrowByPast19_case()
        {
            //Read Data
            string filePath = csvPath + "\\paperData.csv";

            List<double> data1 = DataReader.readDoubleColumn(filePath, 1);
            List<double> normData1 = new List<double>();
            List<double> data2 = DataReader.readDoubleColumn(filePath, 2);
            List<double> normData2 = new List<double>();
            List<double> data3 = DataReader.readDoubleColumn(filePath, 3);
            List<double> normData3 = new List<double>();
            List<double> data4 = DataReader.readDoubleColumn(filePath, 4);
            List<double> normData4 = new List<double>();

            for (int i = 0; i < data1.Count; i++)
            {
                if (data1[i] != 0d && data2[i] != 0d && data3[i] != 0d && data4[i] != 0d)
                {
                    normData1.Add(0.5 * (data1[i] - data1.Min()) / (data1.Max() - data1.Min()));
                    normData2.Add(0.5 * (data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
                    normData3.Add(0.5 * (data3[i] - data3.Min()) / (data3.Max() - data3.Min()));
                    normData4.Add(0.5 * (data4[i] - data4.Min()) / (data4.Max() - data4.Min()));
                }
            }

            //Net Setup
            vNetwork vNet = new vNetwork();

            int numTrainInputs = normData1.Count - 100; //not really the train inputs
            const int numTrainOutputs = 50; //not really the train outputs
            const int numValidateOutputs = 50; //not really the validation outputs

            int inputDays = 19;

            vNet.setNumNeuronsEachLayer(inputDays.ToString() + ",190,1");
            vNet.resetNetwork(inputDays * 10, 3);
            vNet.learnRate = 1d;

            Init:
            //Input Neurons
            for (int i = 0; i < inputDays; i++)
            {
                vNet.setInputNeuron(i, new List<double>(new double[] { normData1[i], normData2[i], normData3[i], normData4[i] }));
            }

            //Hidden Neurons
            List<List<int>> inputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 1d / (double)vNet.numNeuronsEachLayer[0], new List<double>(new double[] { 0d, 0d, 0d, 0d }));
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(1);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, outInputNeurons, 1d / (double)vNet.numNeuronsEachLayer[1], new List<double>(new double[] { 0d, 0d, 0d, 0d }));
            }

            //Train Iteriations
            double bestSqE = double.MaxValue;
            List<List<List<List<List<double>>>>> bestWeights = new List<List<List<List<List<double>>>>>();

            for (int run = 0; run < numTrainInputs + numTrainOutputs + numValidateOutputs - inputDays; run++)
            {
                //Input Neurons
                for (int i = 0; i < inputDays; i++)
                {
                    vNet.nNarray[i, 0].inputs.Clear();
                    vNet.nNarray[i, 0].inputs.Add(new List<double>(new double[] { normData1[i + run], normData2[i + run], normData3[i + run], normData4[i + run] }));
                }

                //Expected Values
                vNet.expOut.Clear();
                vNet.expOut.Add(new List<double>(new double[] { normData1[run + inputDays], normData2[run + inputDays], normData3[run + inputDays], normData4[run + inputDays] }));

                vNet.train();

                double sqE = (vNet.calHalfSqError() * 2);
                if (double.IsNaN(sqE))
                {
                    Console.WriteLine("NaN result detected! (Network gave up?)");

                    //goto Init;
                    //bestSqE = double.MaxValue;
                    //bestWeights = new List<List<List<List<List<double>>>>>();
                    //run = 0;
                }

                if (sqE < bestSqE)
                {
                    //bestSqE = sqE;
                    //bestWeights = vCopyWeights(vNet.nNarray, vNet.NUM_DEPTH, vNet.NUM_LAYERS);
                }

                //Input Neurons
                for (int i = 0; i < inputDays; i++)
                {
                    vNet.nNarray[i, 0].inputs.Clear();
                    vNet.nNarray[i, 0].inputs.Add(new List<double>(new double[] { normData1[i + run + 1], normData2[i + run + 1], normData3[i + run + 1], normData4[i + run + 1] }));
                }

                vForwardPropNet(vNet);
                Console.WriteLine(run + "," + vNet.nNarray[0, 2].postActOutput[0] + "," + vNet.nNarray[0, 2].postActOutput[1] + "," + vNet.nNarray[0, 2].postActOutput[2] + "," + vNet.nNarray[0, 2].postActOutput[3]);
                //Console.WriteLine("Train run " + run + " completed; Sq Error = " + sqE + " Best Sq Error = " + bestSqE);
            }

            /*//Test and Validate Iteriations
            vApplyWeights(bestWeights, vNet.nNarray, vNet.NUM_DEPTH, vNet.NUM_LAYERS);

            for (int run = numTrainInputs; run < numTrainInputs + numTrainOutputs + numValidateOutputs - inputDays; run++)
            {
                //Input Neurons
                for (int i = 0; i < inputDays; i++)
                {
                    vNet.nNarray[i, 0].inputs.Clear();
                    vNet.nNarray[i, 0].inputs.Add(new List<double>(new double[] { normData1[i + run], normData2[i + run], normData3[i + run], normData4[i + run] }));
                }

                //Expected Values
                vNet.expOut.Clear();
                vNet.expOut.Add(new List<double>(new double[] { normData1[run + inputDays], normData2[run + inputDays], normData3[run + inputDays], normData4[run + inputDays] }));

                //Input Neurons
                vNet.nNarray[0, 0].inputs.Clear();
                vNet.nNarray[0, 0].inputs.Add(new List<double>(new double[] { normData1[run + numTrainInputs] }));
                vNet.nNarray[1, 0].inputs.Clear();
                vNet.nNarray[1, 0].inputs.Add(new List<double>(new double[] { normData2[run + numTrainInputs] }));
                vNet.nNarray[2, 0].inputs.Clear();
                vNet.nNarray[2, 0].inputs.Add(new List<double>(new double[] { normData3[run + numTrainInputs] }));
                vNet.nNarray[3, 0].inputs.Clear();
                vNet.nNarray[3, 0].inputs.Add(new List<double>(new double[] { normData4[run + numTrainInputs] }));

                //Expected Values
                vNet.expOut.Clear();
                vNet.expOut.Add(new List<double>(new double[] { normData1[run + 1 + numTrainInputs] }));
                vNet.expOut.Add(new List<double>(new double[] { normData2[run + 1 + numTrainInputs] }));
                vNet.expOut.Add(new List<double>(new double[] { normData3[run + 1 + numTrainInputs] }));
                vNet.expOut.Add(new List<double>(new double[] { normData4[run + 1 + numTrainInputs] }));

                vForwardPropNet(vNet);

                double sqE = (vNet.calHalfSqError() * 2);
                Console.WriteLine("Validate run " + run + " completed; Sq Error = " + sqE);
                Console.WriteLine("Output: " + vNet.nNarray[0, 2].postActOutput[0]);
            }*/

            Console.WriteLine("");
        }

        //Late Version of Vector NN Replicating Paper (many variables)
        static void setPaper_predictWithCnyNasdaqNikkei_case()
        {
            //Read Data
            string filePath = csvPath + "\\usdjpycnyjpyndnk.csv";

            List<double> data1 = DataReader.readDoubleColumn(filePath, 2);
            List<double> normData1 = new List<double>();
            List<double> data2 = DataReader.readDoubleColumn(filePath, 3);
            List<double> normData2 = new List<double>();
            List<double> data3 = DataReader.readDoubleColumn(filePath, 4);
            List<double> normData3 = new List<double>();
            List<double> data4 = DataReader.readDoubleColumn(filePath, 5);
            List<double> normData4 = new List<double>();

            for (int i = 0; i < data1.Count; i++)
            {
                if (data1[i] != 0d && data2[i] != 0d && data3[i] != 0d && data4[i] != 0d)
                {
                    normData1.Add(0.5 * (data1[i] - data1.Min()) / (data1.Max() - data1.Min()));
                    normData2.Add(0.5 * (data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
                    normData3.Add(0.5 * (data3[i] - data3.Min()) / (data3.Max() - data3.Min()));
                    normData4.Add(0.5 * (data4[i] - data4.Min()) / (data4.Max() - data4.Min()));
                }
            }

            //Net Setup
            vNetwork vNet = new vNetwork();

            int validEntries = normData1.Count;
            int numDaysInFutureToPredict = 5;
            //int numTrainInputs = (int)(validEntries * 0.6); //not really the train inputs
            //int numTrainOutputs = (int)(validEntries * 0.2); //not really the train outputs
            //int numValidateOutputs = normData1.Count - numTrainInputs - numTrainOutputs; //not really the validation outputs

            int inputDays = validEntries - numDaysInFutureToPredict;

            vNet.setNumNeuronsEachLayer(inputDays + "," + 200 + "," + inputDays);
            vNet.resetNetwork(vNet.numNeuronsEachLayer.Max(), 3);
            vNet.learnRate = 1d;
            vNet.bLearnRate = 0.5d;

            //Init:
            //Input Neurons
            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                vNet.setInputNeuron(i, new List<double>(new double[] { normData1[i], normData2[i], normData3[i], normData4[i] }));
            }

            //Hidden Neurons
            List<List<int>> inputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 1d / (double)vNet.numNeuronsEachLayer[0], new List<double>(new double[] { 0d, 0d, 0d, 0d }));
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(1);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, outInputNeurons, 1d / (double)vNet.numNeuronsEachLayer[1], new List<double>(new double[] { 0d, 0d, 0d, 0d }));
            }

            //Train Iteriations
            //double bestSqE = double.MaxValue;
            //List<List<List<List<List<double>>>>> bestWeights = new List<List<List<List<List<double>>>>>();

            //for (int run = 0; run < numTrainInputs + numTrainOutputs + numValidateOutputs; run++)
            //{
                //Input Neurons
                //for (int i = 0; i < inputDays; i++)
                //{
                //    vNet.nNarray[i, 0].inputs.Clear();
                //    vNet.nNarray[i, 0].inputs.Add(new List<double>(new double[] { normData1[i + run], normData2[i + run], normData3[i + run], normData4[i + run] }));
                //}

                //Expected Values
                vNet.expOut.Clear();
            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                vNet.expOut.Add(new List<double>(new double[] { normData1[numDaysInFutureToPredict + i]/*, normData2[numDaysInFutureToPredict + i], normData3[numDaysInFutureToPredict + i], normData4[numDaysInFutureToPredict + i]*/ }));
            }

            //Run
            vNet.train();

                double sqE = (vNet.calHalfSqError() * 2);
            Console.WriteLine("Total Square Error: " + sqE);
            //if (double.IsNaN(sqE))
                //{
                    //Console.WriteLine("NaN result detected! (Network gave up?)");

                    //goto Init;
                    //bestSqE = double.MaxValue;
                    //bestWeights = new List<List<List<List<List<double>>>>>();
                    //run = 0;
                //}

                //if (sqE < bestSqE)
                //{
                    //bestSqE = sqE;
                    //bestWeights = vCopyWeights(vNet.nNarray, vNet.NUM_DEPTH, vNet.NUM_LAYERS);
                //}

                //Input Neurons
                for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
                {
                    vNet.nNarray[i, 0].inputs.Clear();
                    vNet.nNarray[i, 0].inputs.Add(new List<double>(new double[] { normData1[i + numDaysInFutureToPredict], normData2[i + numDaysInFutureToPredict], normData3[i + numDaysInFutureToPredict], normData4[i + numDaysInFutureToPredict] }));
                }

                //Run #2
                vForwardPropNet(vNet);

            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                Console.WriteLine(vNet.nNarray[i, 2].postActOutput[0]/* + "," + vNet.nNarray[i, 2].postActOutput[1] + "," + vNet.nNarray[0, 2].postActOutput[2] + "," + vNet.nNarray[0, 2].postActOutput[3]*/);
            }
            //Console.WriteLine("Train run " + run + " completed; Sq Error = " + sqE + " Best Sq Error = " + bestSqE);
        }

        //Multiple Hidden Layer Vector Test
        static void set4layersTestCase()
        {
            //Net Setup
            vNetwork vNet = new vNetwork();

            vNet.setNumNeuronsEachLayer("2,2,2,2");
            vNet.resetNetwork(2, 4);
            vNet.learnRate = 1d;
            vNet.bLearnRate = 0.5d;

            //Input Neurons
            //for (int i = 0; i < 2; i++)
            //{
            vNet.setInputNeuron(0, new List<double>(new double[] { 0.05d, 0.1d }));
            vNet.setInputNeuron(1, new List<double>(new double[] { 0.15d, 0.2d }));
            //}

            //Hidden Neurons Layer 1
            List<List<int>> inputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 1d / (double)vNet.numNeuronsEachLayer[0], new List<double>(new double[] { 0d, 0d }));
            }

            //Hidden Neurons Layer 2
            inputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(1);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, inputNeurons, 1d / (double)vNet.numNeuronsEachLayer[1], new List<double>(new double[] { 0d, 0d }));
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[3]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(2);
            }

            //List<List<List<double>>> outWeights = new List<List<List<double>>>();
            //outWeights.Add(MyMath.makeIdentityMatrix(2));
            //outWeights.Add(MyMath.makeIdentityMatrix(2));

            for (int i = 0; i < vNet.numNeuronsEachLayer[3]; i++)
            {
                vNet.setHidOutNeuron(i, 3, vNeuron.actFuncType.logistic, outInputNeurons, 1d / (double)vNet.numNeuronsEachLayer[2], new List<double>(new double[] { 0d, 0d }));
            }

            //Train Iteriations
            //double bestSqE = double.MaxValue;
            //List<List<List<List<List<double>>>>> bestWeights = new List<List<List<List<List<double>>>>>();

            //for (int run = 0; run < numTrainInputs; run++)
            //{
            //Input Neurons
            //for (int i = 0; i < inputDays; i++)
            //{
            //    vNet.nNarray[i, 0].inputs.Clear();
            //    vNet.nNarray[i, 0].inputs.Add(new List<double>(new double[] { normData1[i + run], normData2[i + run], normData3[i + run], normData4[i + run] }));
            //}

            //Expected Values
            vNet.expOut.Clear();
            vNet.expOut.Add(new List<double>(new double[] { 0.2d , 0.3d }));
            vNet.expOut.Add(new List<double>(new double[] { 0.25d, 0.4d }));

            vNet.train();

            double sqE = (vNet.calHalfSqError() * 2);
            if (double.IsNaN(sqE))
            {
                Console.WriteLine("NaN result detected! (Network gave up?)");

                //goto Init;
                //bestSqE = double.MaxValue;
                //bestWeights = new List<List<List<List<List<double>>>>>();
                //run = 0;
            }

            //if (sqE < bestSqE)
            //{
            //    bestSqE = sqE;
            //    bestWeights = vCopyWeights(vNet.nNarray, vNet.NUM_DEPTH, vNet.NUM_LAYERS);
            //}

            Console.WriteLine("Train run " + " completed; Sq Error = " + sqE + " Best Sq Error = ");
            //}

            //Test and Validate Iteriations
            //vApplyWeights(bestWeights, vNet.nNarray, vNet.NUM_DEPTH, vNet.NUM_LAYERS);

            //for (int run = numTrainInputs; run < numTrainInputs + numTrainOutputs + numValidateOutputs - inputDays; run++)
            //{
            //Input Neurons
            //for (int i = 0; i < inputDays; i++)
            //{
            //    vNet.nNarray[i, 0].inputs.Clear();
            //    vNet.nNarray[i, 0].inputs.Add(new List<double>(new double[] { normData1[i + run], normData2[i + run], normData3[i + run], normData4[i + run] }));
            //}

            //Expected Values
            //vNet.expOut.Clear();
            //vNet.expOut.Add(new List<double>(new double[] { normData1[run + inputDays], normData2[run + inputDays], normData3[run + inputDays], normData4[run + inputDays] }));

            //vForwardPropNet(vNet);

            //double sqE = (vNet.calHalfSqError() * 2);
            //Console.WriteLine("Validate run " + run + " completed; Sq Error = " + sqE);
            //Console.WriteLine("Output: " + vNet.nNarray[0, 2].postActOutput[0]);
            //}

            Console.WriteLine("");
        }

        //Vector NN Predicting USD/JPY with 1, 2, 3... as Input
        static void setPaper_arbitaryInputs()
        {
            //Read Data
            string filePath = csvPath + "\\paperData.csv";

            List<double> data1 = DataReader.readDoubleColumn(filePath, 1);
            List<double> normData1 = new List<double>();
            //<double> data2 = DataReader.readDoubleColumn(filePath, 2);
            //List<double> normData2 = new List<double>();
            //List<double> data3 = DataReader.readDoubleColumn(filePath, 3);
            //List<double> normData3 = new List<double>();
            //List<double> data4 = DataReader.readDoubleColumn(filePath, 4);
            //List<double> normData4 = new List<double>();

            for (int i = 0; i < data1.Count; i++)
            {
                if (data1[i] != 0d /*&& data2[i] != 0d && data3[i] != 0d && data4[i] != 0d*/)
                {
                    normData1.Add(0.5 * (data1[i] - data1.Min()) / (data1.Max() - data1.Min()));
                    //normData2.Add(0.5 * (data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
                    //normData3.Add(0.5 * (data3[i] - data3.Min()) / (data3.Max() - data3.Min()));
                    //normData4.Add(0.5 * (data4[i] - data4.Min()) / (data4.Max() - data4.Min()));
                }
            }

            //Net Setup
            vNetwork vNet = new vNetwork();

            //int numTrainInputs = normData1.Count - 100;
            //const int numTrainOutputs = 50;
            //const int numValidateOutputs = 50;

            vNet.setNumNeuronsEachLayer(normData1.Count + ",200," + normData1.Count);
            vNet.resetNetwork(normData1.Count, 3);
            vNet.learnRate = 1d;
            vNet.bLearnRate = 0.5d;

            //Input Neurons
            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                vNet.setInputNeuron(i, new List<double>(new double[] { i + 1 }));
            }

            //Hidden Neurons L1
            List<List<int>> inputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                vNet.setHidOutNeuron(i, 1, vNeuron.actFuncType.logistic, inputNeurons, 1d / (double)vNet.numNeuronsEachLayer[0], new List<double>(new double[] { 0d }));
            }

            /*//Hidden Neurons L2
            inputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(1);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, inputNeurons, 1d / (double)vNet.numNeuronsEachLayer[0], new List<double>(new double[] { 0d }));
            }*/

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < vNet.numNeuronsEachLayer[1]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(1);
            }

            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                vNet.setHidOutNeuron(i, 2, vNeuron.actFuncType.logistic, outInputNeurons, 1d / (double)vNet.numNeuronsEachLayer[1], new List<double>(new double[] { 0d }));
            }

            //Train Iteriations
            double bestSqE = double.MaxValue;
            List<List<List<List<List<double>>>>> bestWeights = new List<List<List<List<List<double>>>>>();

            //for (int run = 0; run < numTrainInputs + numTrainOutputs + numValidateOutputs - 1; run++)
            //{
            //Input Neurons
            //vNet.nNarray[0, 0].inputs.Clear();
            //vNet.nNarray[0, 0].inputs.Add(new List<double>(new double[] { normData1[run] }));
            //vNet.nNarray[1, 0].inputs.Clear();
            //vNet.nNarray[1, 0].inputs.Add(new List<double>(new double[] { normData2[run] }));
            //vNet.nNarray[2, 0].inputs.Clear();
            //vNet.nNarray[2, 0].inputs.Add(new List<double>(new double[] { normData3[run] }));
            //vNet.nNarray[3, 0].inputs.Clear();
            //vNet.nNarray[3, 0].inputs.Add(new List<double>(new double[] { normData4[run] }));

            //Expected Values
            vNet.expOut.Clear();
            for (int i = 0; i < normData1.Count; i++)
            {
                vNet.expOut.Add(new List<double>(new double[] { normData1[i] }));
            }
            //vNet.expOut.Add(new List<double>(new double[] { normData2[run + 1] }));
            //vNet.expOut.Add(new List<double>(new double[] { normData3[run + 1] }));
            //vNet.expOut.Add(new List<double>(new double[] { normData4[run + 1] }));

            vNet.train();
            Console.WriteLine("Total Square Error: " + vNet.calHalfSqError() * 2);

            //double sqE = (vNet.calHalfSqError() * 2);
            //if (double.IsNaN(sqE))
            //{
            //Console.Write("NaN result detected!!!");
            //
            //}

            /*if(sqE < bestSqE)
            {
                bestSqE = sqE;
                bestWeights = vCopyWeights(vNet.nNarray, vNet.NUM_DEPTH, vNet.NUM_LAYERS);
            }*/

            for (int i = 0; i < vNet.numNeuronsEachLayer[0]; i++){
                vNet.nNarray[i, 0].inputs.Clear();
                vNet.nNarray[i, 0].inputs.Add(new List<double>(new double[] { normData1.Count + i + 1 }));
            }


            //vNet.nNarray[1, 0].inputs.Clear();
            //vNet.nNarray[1, 0].inputs.Add(new List<double>(new double[] { normData2[run + 1] }));
            //vNet.nNarray[2, 0].inputs.Clear();
            //vNet.nNarray[2, 0].inputs.Add(new List<double>(new double[] { normData3[run + 1] }));
            //vNet.nNarray[3, 0].inputs.Clear();
            //vNet.nNarray[3, 0].inputs.Add(new List<double>(new double[] { normData4[run + 1] }));

            vForwardPropNet(vNet);

            for (int i = 0; i < vNet.numNeuronsEachLayer[2]; i++)
            {
                Console.WriteLine(vNet.nNarray[i, 2].postActOutput[0]/* + "," + vNet.nNarray[1, 2].postActOutput[0] + "," + vNet.nNarray[2, 2].postActOutput[0] + "," + vNet.nNarray[3, 2].postActOutput[0]*/);
            }
            //Console.WriteLine("Train day " + run + " completed; Sq Error = " + sqE + " Best Sq Error = " + bestSqE);
            //}

            //Test and Validate Iteriations
            //vApplyWeights(bestWeights, vNet.nNarray, vNet.NUM_DEPTH, vNet.NUM_LAYERS);

            //List<double> inputs = new List<double>();
            //inputs.Add(normData1.Last());
            //inputs.Add(normData2.Last());
            //inputs.Add(normData3.Last());
            //inputs.Add(normData4.Last());

            /*for (int run = 0; run < 50; run++) //numTrainOutputs + numValidateOutputs - 1
            {
                //Input Neurons
                vNet.nNarray[0, 0].inputs.Clear();
                vNet.nNarray[0, 0].inputs.Add(new List<double>(new double[] { inputs[0] }));
                vNet.nNarray[1, 0].inputs.Clear();
                vNet.nNarray[1, 0].inputs.Add(new List<double>(new double[] { inputs[1] }));
                vNet.nNarray[2, 0].inputs.Clear();
                vNet.nNarray[2, 0].inputs.Add(new List<double>(new double[] { inputs[2] }));
                vNet.nNarray[3, 0].inputs.Clear();
                vNet.nNarray[3, 0].inputs.Add(new List<double>(new double[] { inputs[3] }));

                //Expected Values
                vNet.expOut.Clear();
                vNet.expOut.Add(new List<double>(new double[] { normData1[run + 1 + numTrainInputs] }));
                vNet.expOut.Add(new List<double>(new double[] { normData2[run + 1 + numTrainInputs] }));
                vNet.expOut.Add(new List<double>(new double[] { normData3[run + 1 + numTrainInputs] }));
                vNet.expOut.Add(new List<double>(new double[] { normData4[run + 1 + numTrainInputs] }));

                vForwardPropNet(vNet);

                Console.WriteLine(run + "," + vNet.nNarray[0, 2].postActOutput[0]); // + "," + vNet.nNarray[1, 2].postActOutput[0] + "," + vNet.nNarray[2, 2].postActOutput[0] + "," + vNet.nNarray[3, 2].postActOutput[0]
            //double sqE = (vNet.calHalfSqError() * 2);
            //Console.WriteLine("Validate run " + run + " completed; Sq Error = " + sqE);
            //Console.WriteLine("Output: " + vNet.nNarray[0, 2].postActOutput[0]);

            inputs[0] = vNet.nNarray[0, 2].postActOutput[0];
                inputs[1] = vNet.nNarray[1, 2].postActOutput[0];
                inputs[2] = vNet.nNarray[2, 2].postActOutput[0];
                inputs[3] = vNet.nNarray[3, 2].postActOutput[0];
            }*/

            Console.WriteLine("");
        }

        //Matrix NN Test (Can be Used as Template)
        static void setMatrixNetworkTestCase()
        {
            mNetwork mNet = new mNetwork();
            mNet.setNumNeuronsEachLayer("1,2,2,2,1");
            mNet.resetNetwork(2, 5);
            mNet.learnRate = 1d;
            mNet.bLearnRate = 0.5d;

            //Input
            List<List<double>> input = new List<List<double>>();
            input.Add(new List<double>(new double[] { 0.05d, 0.1d }));
            input.Add(new List<double>(new double[] { 0.15d, 0.2d }));
            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                mNet.setInputNeuron(i, input);
            }

            //Hidden L1
            List<List<int>> inputNeurons = new List<List<int>>();
            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                mNet.setHidOutNeuron(i, 1, mNeuron.actFuncType.logistic, inputNeurons, 1d / (double)mNet.numNeuronsEachLayer[0], 0d, input.Count, input[0].Count);
            }

            //Hidden Neurons L2
            inputNeurons = new List<List<int>>();

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(1);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[2]; i++)
            {
                mNet.setHidOutNeuron(i, 2, mNeuron.actFuncType.logistic, inputNeurons, 1d / (double)mNet.numNeuronsEachLayer[1], 0d, input.Count, input[0].Count);
            }

            //Hidden Neurons L3
            inputNeurons = new List<List<int>>();

            for (int i = 0; i < mNet.numNeuronsEachLayer[2]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(2);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[3]; i++)
            {
                mNet.setHidOutNeuron(i, 3, mNeuron.actFuncType.logistic, inputNeurons, 1d / (double)mNet.numNeuronsEachLayer[2], 0d, input.Count, input[0].Count);
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < mNet.numNeuronsEachLayer[3]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(3);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[3]; i++)
            {
                mNet.setHidOutNeuron(i, 4, mNeuron.actFuncType.logistic, outInputNeurons, 1d / (double)mNet.numNeuronsEachLayer[3], 0d, input.Count, input[0].Count);
            }

            //Expected Values
            List<List<double>> expV = new List<List<double>>();
            expV.Add(new List<double>(new double[] { 0.2d, 0.3d }));
            expV.Add(new List<double>(new double[] { 0.25d, 0.4d }));

            mNet.expOut.Clear();
            for(int i = 0; i < mNet.numNeuronsEachLayer[mNet.NUM_LAYERS - 1]; i++)
            {
                mNet.expOut.Add(expV);
            }

            //Run
            mNet.train();
            Console.WriteLine("Total Square Error: " + mNet.calHalfSqError() * 2);

            //Change Inputs and FP
        }

        //Matrix NN for Optimization
        static List<double> setPaperMatCnyNasdaqNikkeiCase(int daysToPredict, int inputNum, int numLayers, int numNeurons)
        {
            //Read Data
            string filePath = csvPath + "\\usdjpycnyjpyndnk.csv";

            List<double> data1 = DataReader.readDoubleColumn(filePath, 2);
            List<double> normData1 = new List<double>();
            List<double> data2 = DataReader.readDoubleColumn(filePath, 3);
            List<double> normData2 = new List<double>();
            List<double> data3 = DataReader.readDoubleColumn(filePath, 4);
            List<double> normData3 = new List<double>();
            List<double> data4 = DataReader.readDoubleColumn(filePath, 5);
            List<double> normData4 = new List<double>();

            for (int i = 0; i < inputNum/*data1.Count*/; i++)
            {
                if (data1[i] != 0d && data2[i] != 0d && data3[i] != 0d && data4[i] != 0d)
                {
                    normData1.Add(0.5 * (data1[i] - data1.Min()) / (data1.Max() - data1.Min()));
                    normData2.Add(0.5 * (data2[i] - data2.Min()) / (data2.Max() - data2.Min()));
                    normData3.Add(0.5 * (data3[i] - data3.Min()) / (data3.Max() - data3.Min()));
                    normData4.Add(0.5 * (data4[i] - data4.Min()) / (data4.Max() - data4.Min()));
                }
            }

            int numDaysToPredict = daysToPredict; //5
            int validInputs = inputNum/*normData1.Count*/ - numDaysToPredict;

            //Net Setup
            mNetwork mNet = new mNetwork();
            string mid = "";
            for(int i = 0; i < numLayers; i++)
            {
                mid += "200,";
            }
            mNet.setNumNeuronsEachLayer("1," + mid + "1");
            mNet.resetNetwork(200, numLayers + 2);
            mNet.learnRate = 1d;
            mNet.bLearnRate = 0.5d;

            //Input
            List<List<double>> input = new List<List<double>>();

            //Vertical Feed
            //for (int i = 0; i < validInputs; i++)
            //{
            //    input.Add(new List<double>(new double[] { normData1[i], normData2[i], normData3[i], normData4[i] }));
            //}

            //Horizontal Feed
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[0].Add(normData1[i]);
            }
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[1].Add(normData2[i]);
            }
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[2].Add(normData3[i]);
            }
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[3].Add(normData4[i]);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                mNet.setInputNeuron(i, input);
            }

            for (int l = 0; l < numLayers; l++)
            {
                //Hidden L1
                List<List<int>> inputNeurons = new List<List<int>>();
                for (int i = 0; i < mNet.numNeuronsEachLayer[0+l]; i++)
                {
                    inputNeurons.Add(new List<int>());
                    inputNeurons[i].Add(i);
                    inputNeurons[i].Add(0+l);
                }

                for (int i = 0; i < mNet.numNeuronsEachLayer[1+l]; i++)
                {
                    mNet.setHidOutNeuron(i, 1+l, mNeuron.actFuncType.logistic, inputNeurons, 1d / validInputs /*(double)mNet.numNeuronsEachLayer[0]*/, 0d, input.Count, input[0].Count);
                }
            }
            //Hidden Neurons L2
            //inputNeurons = new List<List<int>>();

            //for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            //{
            //    inputNeurons.Add(new List<int>());
            //    inputNeurons[i].Add(i);
            //    inputNeurons[i].Add(1);
            //}

            //for (int i = 0; i < mNet.numNeuronsEachLayer[2]; i++)
            //{
            //    mNet.setHidOutNeuron(i, 2, mNeuron.actFuncType.logistic, inputNeurons, 1d / (double)mNet.numNeuronsEachLayer[1], 0d, input[0].Count);
            //}

            //Hidden Neurons L3
            //inputNeurons = new List<List<int>>();

            //for (int i = 0; i < mNet.numNeuronsEachLayer[2]; i++)
            //{
            //    inputNeurons.Add(new List<int>());
            //    inputNeurons[i].Add(i);
            //    inputNeurons[i].Add(2);
            //}

            //for (int i = 0; i < mNet.numNeuronsEachLayer[3]; i++)
            //{
            //    mNet.setHidOutNeuron(i, 3, mNeuron.actFuncType.logistic, inputNeurons, 1d / (double)mNet.numNeuronsEachLayer[2], 0d, input[0].Count);
            //}

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < mNet.numNeuronsEachLayer[numLayers]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(numLayers);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[numLayers+1]; i++)
            {
                mNet.setHidOutNeuron(i, numLayers + 1, mNeuron.actFuncType.logistic, outInputNeurons, 1d / validInputs /*(double)mNet.numNeuronsEachLayer[1]*/, 0d, input.Count, input[0].Count);
            }

            //Expected Values
            List<List<double>> expV = new List<List<double>>();

            //for (int i = 0; i < validInputs; i++)
            //{
            //    expV.Add(new List<double>(new double[] { normData1[i + numDaysToPredict] }));
            //}

            expV.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                expV[0].Add(normData1[i + numDaysToPredict]);
            }

            mNet.expOut.Clear();
            for (int i = 0; i < mNet.numNeuronsEachLayer[mNet.NUM_LAYERS - 1]; i++)
            {
                mNet.expOut.Add(expV);
            }

            //Run
            mNet.train();
            Console.WriteLine("Total Square Error: " + mNet.calHalfSqError() * 2);

            //Change Inputs and FP
            List<List<double>> newInputs = new List<List<double>>();

            //for (int i = 0; i < validInputs; i++)
            //{
            //    newInputs.Add(new List<double>(new double[] { normData1[i + numDaysToPredict], normData2[i + numDaysToPredict], normData3[i + numDaysToPredict], normData4[i + numDaysToPredict] }));
            //}

            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[0].Add(normData1[i + numDaysToPredict]);
            }
            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[1].Add(normData2[i + numDaysToPredict]);
            }
            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[2].Add(normData3[i + numDaysToPredict]);
            }
            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[3].Add(normData4[i + numDaysToPredict]);
            }

            mNet.nNarray[0, 0].inputs[0] = newInputs;

            mNet.forwardPropag();

            List<double> output = new List<double>();
            //Print Results
            for(int i = mNet.nNarray[0, mNet.NUM_LAYERS-1].postActOutput[0].Count - numDaysToPredict; i < mNet.nNarray[0, mNet.NUM_LAYERS - 1].postActOutput[0].Count; i++)
            {
                //Console.WriteLine(mNet.nNarray[0, mNet.NUM_LAYERS - 1].postActOutput[0][i]);
                output.Add(mNet.nNarray[0, mNet.NUM_LAYERS - 1].postActOutput[0][i]);
            }
            return output;
        }

        //Base Case
        static void setFinal3_Basecase()
        {
            //Read Data
            string filePath = csvPath + "\\final3_base.csv";

            List<double> data = DataReader.readDoubleColumn(filePath, 0);
            List<double> normData = new List<double>();

            double Max = data.Max();
            double Min = data.Min();
            for (int i = 0; i < data.Count; i++)
            {
                normData.Add(0.5 * (data[i] - Min) / (Max - Min));
            }

            int numDaysToPredict = 6;
            int validInputs = normData.Count - numDaysToPredict;

            //Net Setup
            mNetwork mNet = new mNetwork();
            mNet.setNumNeuronsEachLayer("1,240,1");
            mNet.resetNetwork(240, 3);
            mNet.learnRate = 1d;
            mNet.bLearnRate = 0.5d;

            //Input
            List<List<double>> input = new List<List<double>>();

            //Horizontal Feed
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[0].Add(normData[i]);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                mNet.setInputNeuron(i, input);
            }

            //Hidden L1
            List<List<int>> inputNeurons = new List<List<int>>();
            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                mNet.setHidOutNeuron(i, 1, mNeuron.actFuncType.logistic, inputNeurons, 1d / validInputs, 0d, input.Count, input[0].Count);
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(1);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[2]; i++)
            {
                mNet.setHidOutNeuron(i, 2, mNeuron.actFuncType.logistic, outInputNeurons, 1d / validInputs, 0d, input.Count, input[0].Count);
            }

            //Expected Values
            List<List<double>> expV = new List<List<double>>();

            expV.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                expV[0].Add(normData[i + numDaysToPredict]);
            }

            mNet.expOut.Clear();
            for (int i = 0; i < mNet.numNeuronsEachLayer[mNet.NUM_LAYERS - 1]; i++)
            {
                mNet.expOut.Add(expV);
            }

            //Run
            mNet.train();
            Console.WriteLine("Total Square Error: " + mNet.calHalfSqError() * 2);

            //Change Inputs and FP
            List<List<double>> newInputs = new List<List<double>>();

            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[0].Add(normData[i + numDaysToPredict]);
            }

            mNet.nNarray[0, 0].inputs[0] = newInputs;

            mNet.forwardPropag();

            //Print Results
            for (int i = 0; i < mNet.nNarray[0, 2].postActOutput[0].Count; i++)
            {
                Console.WriteLine(mNet.nNarray[0, 2].postActOutput[0][i]);
            }
        }

        //Vertically Extended by 3 Years (change file to extend by 1)
        static void setFinal3_Vcase()
        {
            //Read Data
            string filePath = csvPath + "\\final3_vertiEx.csv";

            List<double> data1 = DataReader.readDoubleColumn(filePath, 0);
            List<double> normData1 = new List<double>();

            double Max = data1.Max();
            double Min = data1.Min();

            for (int i = 0; i < data1.Count; i++)
            {
                normData1.Add(0.5 * (data1[i] - Min) / (Max - Min));
            }

            int numDaysToPredict = 6;
            int validInputs = normData1.Count - numDaysToPredict;

            //Net Setup
            mNetwork mNet = new mNetwork();
            mNet.setNumNeuronsEachLayer("1,240,1");
            mNet.resetNetwork(240, 3);
            mNet.learnRate = 1d;
            mNet.bLearnRate = 0.5d;

            //Input
            List<List<double>> input = new List<List<double>>();

            //Horizontal Feed
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[0].Add(normData1[i]);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                mNet.setInputNeuron(i, input);
            }

            //Hidden L1
            List<List<int>> inputNeurons = new List<List<int>>();
            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                mNet.setHidOutNeuron(i, 1, mNeuron.actFuncType.logistic, inputNeurons, 1d / validInputs, 0d, input.Count, input[0].Count);
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(1);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[2]; i++)
            {
                mNet.setHidOutNeuron(i, 2, mNeuron.actFuncType.logistic, outInputNeurons, 1d / validInputs, 0d, input.Count, input[0].Count);
            }

            //Expected Values
            List<List<double>> expV = new List<List<double>>();

            expV.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                expV[0].Add(normData1[i + numDaysToPredict]);
            }

            mNet.expOut.Clear();
            for (int i = 0; i < mNet.numNeuronsEachLayer[mNet.NUM_LAYERS - 1]; i++)
            {
                mNet.expOut.Add(expV);
            }

            //Run
            mNet.train();
            Console.WriteLine("Total Square Error: " + mNet.calHalfSqError() * 2);

            //Change Inputs and FP
            List<List<double>> newInputs = new List<List<double>>();

            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[0].Add(normData1[i + numDaysToPredict]);
            }

            mNet.nNarray[0, 0].inputs[0] = newInputs;

            mNet.forwardPropag();

            //Print Results
            for (int i = 0; i < mNet.nNarray[0, 2].postActOutput[0].Count; i++)
            {
                Console.WriteLine(mNet.nNarray[0, 2].postActOutput[0][i]);
            }
        }

        //Flawed Attempt at Changing the Cost Function
        static void setFinal3_baseModExpCase()
        {
            //Read Data
            string filePath = csvPath + "\\final3_base.csv";

            List<double> data1 = DataReader.readDoubleColumn(filePath, 0);
            List<double> normData1 = new List<double>();

            double Max = data1.Max();
            double Min = data1.Min();

            for (int i = 0; i < data1.Count; i++)
            {
                normData1.Add(0.5 * (data1[i] - Min) / (Max - Min));
            }

            int numDaysToPredict = 6;
            int validInputs = normData1.Count - numDaysToPredict;

            //Net Setup
            mNetwork mNet = new mNetwork();
            mNet.setNumNeuronsEachLayer("1,240,1");
            mNet.resetNetwork(240, 3);
            mNet.learnRate = 1d;
            mNet.bLearnRate = 0.5d;

            //Input
            List<List<double>> input = new List<List<double>>();

            //Horizontal Feed
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[0].Add(normData1[i]);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                mNet.setInputNeuron(i, input);
            }

            //Hidden L1
            List<List<int>> inputNeurons = new List<List<int>>();
            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                mNet.setHidOutNeuron(i, 1, mNeuron.actFuncType.logistic, inputNeurons, 1d / validInputs, 0d, input.Count, input[0].Count);
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(1);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[2]; i++)
            {
                mNet.setHidOutNeuron(i, 2, mNeuron.actFuncType.logistic, outInputNeurons, 1d / validInputs, 0d, input.Count, input[0].Count);
            }

            //Expected Values
            List<List<double>> expV = new List<List<double>>();

            expV.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                if (i >= validInputs - numDaysToPredict + 1)
                {
                    expV[0].Add(normData1[i + numDaysToPredict]);
                }
                else
                {
                    expV[0].Add(-2);
                }
            }

            mNet.expOut.Clear();
            for (int i = 0; i < mNet.numNeuronsEachLayer[mNet.NUM_LAYERS - 1]; i++)
            {
                mNet.expOut.Add(expV);
            }

            //Run
            mNet.train();
            Console.WriteLine("Total Square Error: " + mNet.calHalfSqError() * 2);

            //Change Inputs and FP
            List<List<double>> newInputs = new List<List<double>>();

            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[0].Add(normData1[i + numDaysToPredict]);
            }

            mNet.nNarray[0, 0].inputs[0] = newInputs;

            mNet.forwardPropag();

            //Print Results
            for (int i = 0; i < mNet.nNarray[0, 2].postActOutput[0].Count; i++)
            {
                Console.WriteLine(mNet.nNarray[0, 2].postActOutput[0][i]);
            }
        }

        //Horizontally Extend by 3 Input Types
        static void setFinal3_Hcase()
        {
            //Read Data
            string filePath = csvPath + "\\final3_hori.csv";

            List<double> data1 = DataReader.readDoubleColumn(filePath, 0);
            List<double> normData1 = new List<double>();
            List<double> data2 = DataReader.readDoubleColumn(filePath, 1);
            List<double> normData2 = new List<double>();
            List<double> data3 = DataReader.readDoubleColumn(filePath, 2);
            List<double> normData3 = new List<double>();
            List<double> data4 = DataReader.readDoubleColumn(filePath, 3);
            List<double> normData4 = new List<double>();

            double Max1 = data1.Max();
            double Min1 = data1.Min();

            double Max2 = data2.Max();
            double Min2 = data2.Min();

            double Max3 = data3.Max();
            double Min3 = data3.Min();

            double Max4 = data4.Max();
            double Min4 = data4.Min();

            for (int i = 0; i < data1.Count; i++)
            {
                if (data1[i] != 0d && data2[i] != 0d && data3[i] != 0d && data4[i] != 0d)
                {
                    normData1.Add(0.5d * (data1[i] - Min1) / (Max1 - Min1));
                    normData2.Add(0.5d * (data2[i] - Min2) / (Max2 - Min2));
                    normData3.Add(0.5d * (data3[i] - Min3) / (Max3 - Min3));
                    normData4.Add(0.5d * (data4[i] - Min4) / (Max4 - Min4));
                }
            }

            int numDaysToPredict = 6;
            int validInputs = normData1.Count - numDaysToPredict;

            //Net Setup
            mNetwork mNet = new mNetwork();
            mNet.setNumNeuronsEachLayer("1,240,1");
            mNet.resetNetwork(240, 3);
            mNet.learnRate = 1d;
            mNet.bLearnRate = 0.5d;

            //Input
            List<List<double>> input = new List<List<double>>();

            //Horizontal Feed
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[0].Add(normData1[i]);
            }
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[1].Add(normData2[i]);
            }
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[2].Add(normData3[i]);
            }
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[3].Add(normData4[i]);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                mNet.setInputNeuron(i, input);
            }

            //Hidden L1
            List<List<int>> inputNeurons = new List<List<int>>();
            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                mNet.setHidOutNeuron(i, 1, mNeuron.actFuncType.logistic, inputNeurons, 1d / validInputs, 0d, input.Count, input[0].Count);
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(1);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[2]; i++)
            {
                mNet.setHidOutNeuron(i, 2, mNeuron.actFuncType.logistic, outInputNeurons, 1d / validInputs, 0d, input.Count, input[0].Count);
            }

            //Expected Values
            List<List<double>> expV = new List<List<double>>();

            expV.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                expV[0].Add(normData1[i + numDaysToPredict]);
            }

            mNet.expOut.Clear();
            for (int i = 0; i < mNet.numNeuronsEachLayer[mNet.NUM_LAYERS - 1]; i++)
            {
                mNet.expOut.Add(expV);
            }

            //Run
            mNet.train();
            Console.WriteLine("Total Square Error: " + mNet.calHalfSqError() * 2);

            //Change Inputs and FP
            List<List<double>> newInputs = new List<List<double>>();

            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[0].Add(normData1[i + numDaysToPredict]);
            }
            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[1].Add(normData2[i + numDaysToPredict]);
            }
            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[2].Add(normData3[i + numDaysToPredict]);
            }
            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[3].Add(normData4[i + numDaysToPredict]);
            }

            mNet.nNarray[0, 0].inputs[0] = newInputs;

            mNet.forwardPropag();

            //Print Results
            for (int i = 0; i < mNet.nNarray[0, 2].postActOutput[0].Count; i++)
            {
                Console.WriteLine(mNet.nNarray[0, 2].postActOutput[0][i]);
            }
        }

        //Horizontally Extend by 1 Input Types
        static void setFinal3_H2case()
        {
            //Read Data
            string filePath = csvPath + "\\final3_hori2.csv";

            List<double> data1 = DataReader.readDoubleColumn(filePath, 0);
            List<double> normData1 = new List<double>();
            List<double> data2 = DataReader.readDoubleColumn(filePath, 1);
            List<double> normData2 = new List<double>();

            double Max1 = data1.Max();
            double Min1 = data1.Min();

            double Max2 = data2.Max();
            double Min2 = data2.Min();

            for (int i = 0; i < data1.Count; i++)
            {
                if (data1[i] != 0d && data2[i] != 0d)
                {
                    normData1.Add(0.5d * (data1[i] - Min1) / (Max1 - Min1));
                    normData2.Add(0.5d * (data2[i] - Min2) / (Max2 - Min2));
                }
            }

            int numDaysToPredict = 6;
            int validInputs = normData1.Count - numDaysToPredict;

            //Net Setup
            mNetwork mNet = new mNetwork();
            mNet.setNumNeuronsEachLayer("1,240,1");
            mNet.resetNetwork(240, 3);
            mNet.learnRate = 1d;
            mNet.bLearnRate = 0.5d;

            //Input
            List<List<double>> input = new List<List<double>>();

            //Horizontal Feed
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[0].Add(normData1[i]);
            }
            input.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                input[1].Add(normData2[i]);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                mNet.setInputNeuron(i, input);
            }

            //Hidden L1
            List<List<int>> inputNeurons = new List<List<int>>();
            for (int i = 0; i < mNet.numNeuronsEachLayer[0]; i++)
            {
                inputNeurons.Add(new List<int>());
                inputNeurons[i].Add(i);
                inputNeurons[i].Add(0);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                mNet.setHidOutNeuron(i, 1, mNeuron.actFuncType.logistic, inputNeurons, 1d / validInputs, 0d, input.Count, input[0].Count);
            }

            //Output Neurons
            List<List<int>> outInputNeurons = new List<List<int>>();

            for (int i = 0; i < mNet.numNeuronsEachLayer[1]; i++)
            {
                outInputNeurons.Add(new List<int>());
                outInputNeurons[i].Add(i);
                outInputNeurons[i].Add(1);
            }

            for (int i = 0; i < mNet.numNeuronsEachLayer[2]; i++)
            {
                mNet.setHidOutNeuron(i, 2, mNeuron.actFuncType.logistic, outInputNeurons, 1d / validInputs, 0d, input.Count, input[0].Count);
            }

            //Expected Values
            List<List<double>> expV = new List<List<double>>();

            expV.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                expV[0].Add(normData1[i + numDaysToPredict]);
            }

            mNet.expOut.Clear();
            for (int i = 0; i < mNet.numNeuronsEachLayer[mNet.NUM_LAYERS - 1]; i++)
            {
                mNet.expOut.Add(expV);
            }

            //Run
            mNet.train();
            Console.WriteLine("Total Square Error: " + mNet.calHalfSqError() * 2);

            //Change Inputs and FP
            List<List<double>> newInputs = new List<List<double>>();

            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[0].Add(normData1[i + numDaysToPredict]);
            }
            newInputs.Add(new List<double>());
            for (int i = 0; i < validInputs; i++)
            {
                newInputs[1].Add(normData2[i + numDaysToPredict]);
            }

            mNet.nNarray[0, 0].inputs[0] = newInputs;

            mNet.forwardPropag();

            //Print Results
            for (int i = 0; i < mNet.nNarray[0, 2].postActOutput[0].Count; i++)
            {
                Console.WriteLine(mNet.nNarray[0, 2].postActOutput[0][i]);
            }
        }

        //Vector Network
        public class vNetwork
        {
            //Class Field Variables
            public int NUM_DEPTH;
            public int NUM_LAYERS;
            public vNeuron[,] nNarray = null;
            public double learnRate = 0f;
            public double bLearnRate = 0.5d;
            public List<List<double>> expOut = null;
            public List<int> numNeuronsEachLayer = new List<int>();

            //Empty Initializing Constructor
            public vNetwork()
            {
                this.NUM_DEPTH = -1;
                this.NUM_LAYERS = -1;
                this.nNarray = null;
            }

            //Standard Constructor
            public vNetwork(int inDepth, int inLayers)
            {
                this.NUM_DEPTH = inDepth;
                this.NUM_LAYERS = inLayers;
                this.nNarray = new vNeuron[inDepth, inLayers];
                this.expOut = new List<List<double>>();

                for (int i = 0; i < inLayers; i++)
                {
                    for (int j = 0; j < inDepth; j++)
                    {
                        this.nNarray[j, i] = new vNeuron();
                    }
                }
            }

            //Record # Neurons in Each Layer (for counting purposes)
            public void setNumNeuronsEachLayer(string x)
            {
                string[] split = x.Split(',');
                foreach (string i in split)
                {
                    numNeuronsEachLayer.Add(int.Parse(i));
                }
            }

            //Set 1 Input Neuron
            public void setInputNeuron(int m, List<double> finalOutput)
            {
                this.nNarray[m, 0] = new vNeuron(finalOutput);
            }

            //Set 1 Hidden or Output Neuron with Weights Input as Decimal (randomly generated)
            public void setHidOutNeuron(int m, int n, vNeuron.actFuncType func, List<List<int>> inputNeu, Double weightVar, List<double> beta)
            {
                this.nNarray[m, n] = new vNeuron(n, func, inputNeu, weightVar, beta);
            }

            //Set 1 Hidden or Output Neuron with Weights Input as Matrix
            public void setHidOutNeuron(int m, int n, vNeuron.actFuncType func, List<List<int>> inputNeu, List<List<List<double>>> weights, List<double> beta)
            {
                this.nNarray[m, n] = new vNeuron(n, func, inputNeu, weights, beta);
            }

            //Change Network Shape and Clear Neurons
            public int resetNetwork(int inDepth, int inLayers)
            {
                this.NUM_DEPTH = inDepth;
                this.NUM_LAYERS = inLayers;
                this.nNarray = new vNeuron[inDepth, inLayers];
                this.expOut = new List<List<double>>();

                for (int i = 0; i < inLayers; i++)
                {
                    for (int j = 0; j < inDepth; j++)
                    {
                        this.nNarray[j, i] = new vNeuron();
                    }
                }

                return 0;
            }

            //Calculate Half Total Square Error of All Output Neurons
            public double calHalfSqError()
            {
                double result = 0f;
                    for (int i = 0; i < this.numNeuronsEachLayer[this.NUM_LAYERS - 1]; i++)
                    {
                        for (int j = 0; j < this.expOut[0].Count; j++)
                        {
                            try
                            {
                                result += 0.5 * Math.Pow(this.expOut[i][j] - this.nNarray[i, this.NUM_LAYERS - 1].postActOutput[j], 2);
                            }
                            catch (Exception e)
                            {
                                return 0d;
                            }
                        }
                    }
                    return result;
            }

            //Change Weights to New Weights (which the gradient is applied to) and Clear New Weights (this method should be used after each satisfactory backward propagation run)
            public void switchToNewWeights()
            {
                for (int i = 0; i < this.NUM_LAYERS; i++)
                {
                    for (int j = 0; j < this.NUM_DEPTH; j++)
                    {
                        if (nNarray[j, i].weights.Count == nNarray[j, i].newWeights.Count)
                        {
                            nNarray[j, i].switchToNewWeights();
                        }
                    }
                }
            }

            //Forward Propagates and Backward Propagates, Continue Loop If there is a Decrease in Half Square Error Large Enough (>0.000001)
            public void train()
            {
                double lastRunHalfSqError = 0f;
                int runNum = 1;
                double halfSqError = double.MaxValue;

                do
                {
                    lastRunHalfSqError = halfSqError;

                    vForwardPropNet(this);
                    Console.WriteLine("Forward Propagation Complete, Run Number " + runNum);

                    vBackPropNet(this);
                    Console.WriteLine("Back Propagation Complete, Run Number " + runNum);

                    runNum++;

                    halfSqError = this.calHalfSqError();

                    for (int i = 0; i < this.numNeuronsEachLayer[this.NUM_LAYERS - 1]; i++)
                    {
                    //int i = 0;
                            Console.Write("Output Neuron #" + i + " Outputs Value of \n");
                            for (int j = 0; j < this.nNarray[i, this.NUM_LAYERS - 1].postActOutput.Count; j++)
                            {
                            try
                            {
                                Console.Write(j + "d: " + this.nNarray[i, this.NUM_LAYERS - 1].postActOutput[j] + "\n");
                            }
                            catch
                            {

                            }
                            try
                            {
                                Console.Write("Expected: (" + this.expOut[i][j] + ")\n");
                            }
                            catch
                            {

                            }
                            }
                    }
                    Console.WriteLine("Half Square Error is " + halfSqError);
                }
                while (lastRunHalfSqError - halfSqError > 0.000001);
            }
        }

        //Matrix Network
        public class mNetwork
        {
            //Class Field Variables
            public int NUM_DEPTH;
            public int NUM_LAYERS;
            public mNeuron[,] nNarray = null;
            public double learnRate = 1d;
            public double bLearnRate = 0.5d;
            public List<List<List<double>>> expOut = null;
            public List<int> numNeuronsEachLayer = new List<int>();

            //Empty Initializing Constructor
            public mNetwork()
            {
                this.NUM_DEPTH = -1;
                this.NUM_LAYERS = -1;
                this.nNarray = null;
            }

            //Record # Neurons in Each Layer (for counting purposes)
            public void setNumNeuronsEachLayer(string x)
            {
                string[] split = x.Split(',');
                foreach (string i in split)
                {
                    numNeuronsEachLayer.Add(int.Parse(i));
                }
            }

            //Set 1 Input Neuron
            public void setInputNeuron(int m, List<List<double>> finalOutput)
            {
                this.nNarray[m, 0] = new mNeuron(finalOutput);
            }

            //Set 1 Hidden or Output Neuron with Weights and Bias (beta) Inputs as Decimal (randomly generated)
            public void setHidOutNeuron(int m, int n, mNeuron.actFuncType func, List<List<int>> inputNeu, double weightVar, double betaVar, int mCount, int nCount)
            {
                this.nNarray[m, n] = new mNeuron(n, m, func, inputNeu, weightVar, betaVar, mCount, nCount);
            }

            //Change Network Shape and Clear Neurons
            public void resetNetwork(int inDepth, int inLayers)
            {
                this.NUM_DEPTH = inDepth;
                this.NUM_LAYERS = inLayers;
                this.nNarray = new mNeuron[inDepth, inLayers];
                this.expOut = new List<List<List<double>>>();

                for (int i = 0; i < inLayers; i++)
                {
                    for (int j = 0; j < inDepth; j++)
                    {
                        this.nNarray[j, i] = new mNeuron();
                    }
                }
            }

            //Calculate Half Total Square Error of All Output Neurons
            public double calHalfSqError()
            {
                double result = 0d;
                for (int i = 0; i < this.numNeuronsEachLayer[this.NUM_LAYERS - 1]; i++)
                {
                    for (int j = 0; j < this.expOut[0].Count; j++)
                    {
                        for(int k = 0; k < this.expOut[0][0].Count; k++)
                        {
                            try
                            {
                                result += 0.5 * Math.Pow(this.expOut[i][j][k] - this.nNarray[i, this.NUM_LAYERS - 1].postActOutput[j][k], 2);
                            }
                            catch (Exception e)
                            {
                                return 0d;
                            }
                        }
                        
                    }
                }
                return result;
            }

            //Change Weights to New Weights (which the gradient is applied to) and Clear New Weights (this method should be used after each satisfactory backward propagation run)
            public void switchToNewWeights()
            {
                for (int i = 0; i < this.NUM_LAYERS; i++)
                {
                    for (int j = 0; j < this.NUM_DEPTH; j++)
                    {
                        if (nNarray[j, i].weights.Count == nNarray[j, i].newWeights.Count)
                        {
                            nNarray[j, i].switchToNewWeights();
                        }
                    }
                }
            }

            //Forward Propagates and Backward Propagates, Continue Loop If there is a Decrease in Half Square Error Large Enough (>0.000001)
            public void train()
            {
                double lastRunHalfSqError = 0d;
                int runNum = 1;
                double halfSqError = double.MaxValue;

                do
                {
                    lastRunHalfSqError = halfSqError;

                    for(int i = 0; i < this.NUM_LAYERS; i++)
                    {
                        for(int j = 0; j < this.numNeuronsEachLayer[i]; j++)
                        {
                            this.nNarray[j, i].ForwardPropag(this.nNarray);
                        }
                    }
                    Console.WriteLine("Forward Propagation Complete, Run Number " + runNum);

                    for(int i = 0; i < this.numNeuronsEachLayer[this.NUM_LAYERS - 1]; i++)
                    {
                        this.nNarray[i, this.NUM_LAYERS - 1].BackPropag(this.nNarray, this.expOut[i], this.learnRate, this.bLearnRate);
                    }

                    for (int i = (this.NUM_LAYERS - 2); i >= 0; i--)
                    {
                        for (int j = 0; j < this.numNeuronsEachLayer[i]; j++)
                        {
                            this.nNarray[j, i].BackPropag(this.nNarray, this.learnRate, this.bLearnRate);
                        }
                    }
                    Console.WriteLine("Back Propagation Complete, Run Number " + runNum);

                    runNum++;

                    halfSqError = this.calHalfSqError();

                    for (int i = 0; i < this.numNeuronsEachLayer[this.NUM_LAYERS - 1]; i++)
                    {
                        //int i = 0;
                        //Console.WriteLine("Output Neuron #" + i + " Outputs Value of ");
                        for (int j = 0; j < this.nNarray[i, this.NUM_LAYERS - 1].postActOutput.Count; j++)
                        {
                            for(int k = 0; k < this.nNarray[i, this.NUM_LAYERS - 1].postActOutput[j].Count; k++)
                            {
                                try
                                {
                                    //Console.WriteLine(this.nNarray[i, this.NUM_LAYERS - 1].postActOutput[j][k] + "(" + this.expOut[i][j][k] + ")");
                                }
                                catch
                                {

                                }
                            }
                        }
                    }
                    //Console.WriteLine("Half Square Error is " + halfSqError);
                }
                while (lastRunHalfSqError - halfSqError > 0.000001);
            }

            //Forward Propagate Once
            public void forwardPropag()
            {
                for (int i = 0; i < this.NUM_LAYERS; i++)
                {
                    for (int j = 0; j < this.numNeuronsEachLayer[i]; j++)
                    {
                        this.nNarray[j, i].ForwardPropag(this.nNarray);
                    }
                }
                //Console.WriteLine("Forward Propagation Complete");
            }
        }
    }
}