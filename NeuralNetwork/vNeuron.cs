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
    class vNeuron
    {
        //Class Field Variables
        const double DEFAULT_WEIGHT = 0.5d;

        //This Neuron's Layer #
        int layer = -1;

        //This Neuron's Vectors' # of Dimensions
        int dimension = -1;

        //This Neuron's Input Neurons' Coordinates
        List<List<int>> inputNeurons = new List<List<int>>();

        //This Neuron's Input Neurons' Values
        public List<List<double>> inputs = new List<List<double>>();

        //Collection of Weights for All Input Neurons
        public List<List<List<double>>> weights = new List<List<List<double>>>();

        //This Neuron's Bias Value
        List<double> beta = new List<double>();

        //Activation Function Type
        public enum actFuncType {logistic, identity};
        actFuncType actFunc = actFuncType.logistic;

        //Pre-activation Output Save
        List<double> output = new List<double>();

        //This Neuron's Final, Post-activation Output
        public List<double> postActOutput = new List<double>();

        //Delta Calculated from this Neuron's Error and f Prime
        List<double> delta = new List<double>();

        //For Hidden Neurons, the Delta Sum Passed Down from the Layer Above
        public List<List<double>> handedDownDelta = new List<List<double>>();

        //New Weights with the Gradient Applied (stored for switching after iteration)
        public List<List<List<double>>> newWeights = new List<List<List<double>>>();

        //Empty Constructor
        public vNeuron()
        {

        }

        /*public vNeuron(main.Network inNet, List<List<int>> inInputNs, int inLayer, int inDim)
        {
            this.inputNeurons = inInputNs;
            this.layer = inLayer;
            this.dimension = inDim;

            this.weights = new List<List<double>>();
            for(int i = 0; i < inNet.NUM_DEPTH; i++)
            {
                this.weights.Add(Enumerable.Repeat(DEFAULT_WEIGHT, inDim).ToList());
            }
        }*/

        //Input Neuron Constructor
        public vNeuron(List<double> output)
        {
            this.layer = 0;
            this.actFunc = actFuncType.identity;
            this.dimension = output.Count;

            this.inputs.Add(new List<double>());
            for (int i = 0; i < this.dimension; i++)
            {
                this.inputs[0].Add(output[i]);
                this.beta.Add(0d);
            }

            this.weights.Add(MyMath.makeIdentityMatrix(this.dimension));
        }

        //Hidden and Output Neuron Constructor
        public vNeuron(int layer, actFuncType func, List<List<int>> inputNeu, double weightVar, /*List<List<double>> weights, */List<double> beta)
        {
            this.layer = layer;
            this.actFunc = func;
            this.dimension = beta.Count;

            foreach(List<int> vec in inputNeu)
            {
                List<int> curVec = new List<int>();
                foreach(int i in vec)
                {
                    curVec.Add(i);
                }
                this.inputNeurons.Add(curVec);
            }

            Random rng = main.rng;

            for(int k = 0; k < inputNeu.Count; k++)
            {
                this.weights.Add(new List<List<double>>());
                for (int i = 0; i < this.dimension; i++)
                {
                    this.weights[k].Add(new List<double>());
                    for (int j = 0; j < this.dimension; j++)
                    {
                        this.weights[k][i].Add(weightVar * rng.NextDouble());
                    }
                }
            }

            foreach(double x in beta)
            {
                this.beta.Add(x);
            }

            for(int j = 0; j < inputNeu.Count; j++)
            {
                //this.handedDownDelta.Add(new List<double>());
                for (int i = 0; i < beta.Count; i++)
                {
                    //this.handedDownDelta[j].Add(0d);
                }
            }
        }

        //Custom Hidden and Output Neurons
        public vNeuron(int layer, actFuncType func, List<List<int>> inputNeu, List<List<List<double>>> weights, List<double> beta)
        {
            this.layer = layer;
            this.actFunc = func;
            this.dimension = beta.Count;

            foreach (List<int> vec in inputNeu)
            {
                List<int> curVec = new List<int>();
                foreach (int i in vec)
                {
                    curVec.Add(i);
                }
                this.inputNeurons.Add(curVec);
            }

            //Random rng = new Random();

            for(int k = 0; k < weights.Count; k++)
            {
                this.weights.Add(new List<List<double>>());
                for (int i = 0; i < weights[0].Count; i++)
                {
                    this.weights[k].Add(new List<double>());
                    for (int j = 0; j < weights[0][0].Count; j++)
                    {
                        this.weights[k][i].Add(weights[k][i][j]);
                    }
                }
            }

            foreach (double x in beta)
            {
                this.beta.Add(x);
            }

            for (int j = 0; j < inputNeu.Count; j++)
            {
                //this.handedDownDelta.Add(new List<double>());
                for (int i = 0; i < beta.Count; i++)
                {
                    //this.handedDownDelta[j].Add(0d);
                }
            }
        }

        //Get Values from all Input Neurons' Coordinates
        public void GetInputs(vNeuron[,] nn)
        {
            this.inputs.Clear();

            for(int i = 0; i < this.inputNeurons.Count(); i++)
            {
                this.inputs.Add(new List<double>());
                for(int j = 0; j < this.dimension; j++)
                {
                    int curM = this.inputNeurons[i][0];
                    int curN = this.inputNeurons[i][1];
                    vNeuron curNeu = nn[curM, curN];
                    this.inputs[i].Add(curNeu.postActOutput[j]);
                }
            }
        }

        //Forward Propagate this Neuron
        public void ForwardPropag(vNeuron[,] nn)
        {
            if (this.layer != 0) {
                this.GetInputs(nn);
            }

            this.output = Enumerable.Repeat(0d, this.inputs[0].Count).ToList();
            this.postActOutput = Enumerable.Repeat(0d, this.inputs[0].Count).ToList();
            this.handedDownDelta = main.createNewMatrix(inputs[0].Count, inputs[0].Count);

            for (int i = 0; i < this.inputs.Count; i++)
            {
                List<double> curInput = this.inputs[i];
                List<List<double>> result = MyMath.matMulti(this.weights[i], MyMath.makeVertiColMat(curInput));
                this.output = MyMath.transposeMat(MyMath.matAdd(MyMath.makeVertiColMat(this.output), result))[0];
            }

            if (this.actFunc == actFuncType.logistic)
            {
                for(int j = 0; j < this.output.Count; j++)
                {
                    this.postActOutput[j] = MyMath.logisticFunc(this.output[j], beta[j]);
                }
            }
            else if (this.actFunc == actFuncType.identity)
            {
                for (int j = 0; j < this.output.Count; j++)
                {
                    this.postActOutput[j] = MyMath.identityFunc(this.output[j], beta[j]);
                }
            }
        }

        //Back Propagate for Output Neuron
        public void BackPropag(vNeuron[,] net, List<double> expOut, double learnRate, double bLearnRate)
        {
            List<double> error = Enumerable.Repeat(0d, this.postActOutput.Count).ToList();
            for (int i = 0; i < expOut.Count; i++)
            {
                error[i] = this.postActOutput[i] - expOut[i];
            }
            //List<double> error = MyMath.colSub(this.postActOutput, expOut);

            List<double> fPrime = new List<double>();
            if (this.actFunc == actFuncType.logistic)
            {
                for (int i = 0; i < this.output.Count; i++)
                {
                    fPrime.Add(MyMath.logisticPrimeFunc(this.output[i]));
                }
            }
            else if (this.actFunc == actFuncType.identity)
            {
                for (int i = 0; i < this.output.Count; i++)
                {
                    fPrime.Add(1d);
                }
            }

            List<List<double>> result = MyMath.pointwiseMatMulti(MyMath.makeHorizonColMat(error), MyMath.makeHorizonColMat(fPrime));
            this.delta = result[0];
            this.beta = MyMath.matSub(MyMath.makeHorizonColMat(this.beta), MyMath.scalarProd(bLearnRate, MyMath.makeHorizonColMat(this.delta)))[0];

            for (int i = 0; i < this.inputNeurons.Count; i++)
            {
                int m = inputNeurons[i][0];
                int n = inputNeurons[i][1];

                if (net[m, n].handedDownDelta.Count > 0)
                {
                    net[m, n].handedDownDelta[0]/*.Add(this.delta);*/ = MyMath.colAdd(net[m, n].handedDownDelta[0], this.delta);
                }
                else
                {
                    net[m, n].handedDownDelta.Add(Enumerable.Repeat(0d, delta.Count).ToList());
                    net[m, n].handedDownDelta[0]/*.Add(this.delta);*/ = MyMath.colAdd(net[m, n].handedDownDelta[0], this.delta);
                }

                List<List<double>> matrix = this.weights[i];
                this.newWeights.Add(MyMath.matSub(matrix, MyMath.scalarProd(learnRate, MyMath.matMulti(MyMath.makeVertiColMat(this.delta), MyMath.makeHorizonColMat(this.inputs[i])))));
            }
        }

        //Back Propagate for Hidden Neuron
        public void BackPropag(vNeuron[,] net, int mPos, double learnRate, double bLearnRate)
        {   
            for (int i = 0; i < this.inputNeurons.Count; i++)
            {
                List<List<double>> error = MyMath.matMulti(MyMath.transposeMat(this.weights[i]), MyMath.makeVertiColMat(this.handedDownDelta[0]));

                List<double> fPrime = new List<double>();
                if (this.actFunc == actFuncType.logistic)
                {
                    for (int j = 0; j < this.output.Count; j++)
                    {
                        fPrime.Add(MyMath.logisticPrimeFunc(this.output[j]));
                    }
                }
                else if (this.actFunc == actFuncType.identity)
                {
                    for (int j = 0; j < this.output.Count; j++)
                    {
                        fPrime.Add(1d);
                    }
                }

                List<List<double>> result = MyMath.transposeMat(MyMath.pointwiseMatMulti(error, MyMath.makeVertiColMat(fPrime)));
                this.delta = result[0];
                this.beta = MyMath.matSub(MyMath.makeHorizonColMat(this.beta), MyMath.scalarProd(bLearnRate, MyMath.makeHorizonColMat(this.delta)))[0];

                int m = inputNeurons[i][0];
                int n = inputNeurons[i][1];
                
                if (net[m, n].handedDownDelta.Count > 0)
                    net[m, n].handedDownDelta[0]/*.Add(this.delta);*/ = MyMath.colAdd(net[m, n].handedDownDelta[0], this.delta);
                else
                {
                    net[m, n].handedDownDelta.Add(Enumerable.Repeat(0d, delta.Count).ToList());
                    net[m, n].handedDownDelta[0]/*.Add(this.delta);*/ = MyMath.colAdd(net[m, n].handedDownDelta[0], this.delta);
                }

                List<List<double>> matrix = this.weights[i];
                this.newWeights.Add(MyMath.matSub(matrix, MyMath.scalarProd(learnRate, MyMath.matMulti(MyMath.makeVertiColMat(this.delta), MyMath.makeHorizonColMat(this.inputs[i])))));
            }
        }

        //Switch to New Weights, Clear New Weights Variable for Next Iteration
        public void switchToNewWeights()
        {
            for (int i = 0; i < this.weights.Count; i++)
            {
                for (int j = 0; j < this.weights[0].Count; j++)
                {
                    for(int k = 0; k < this.weights[0][0].Count; k++)
                    {
                        this.weights[i][j][k] = this.newWeights[i][j][k];
                    }
                }
            }
            this.newWeights.Clear();
        }
    }
}
