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
    class mNeuron
    {
        //Class Field Variables

        //This Neuron's Layer #
        int layer = -1;

        //This Neuron's Depth #
        int depth = -1; //debug purposes only

        //This Neuron's Input Neurons' Coordinates
        List<List<int>> inputNeurons = new List<List<int>>();

        //This Neuron's Input Neurons' Values
        public List<List<List<double>>> inputs = new List<List<List<double>>>();

        //Collection of Weights for All Input Neurons
        public List<List<List<double>>> weights = new List<List<List<double>>>();

        //This Neuron's Bias Value
        List<List<double>> beta = new List<List<double>>();

        //Activation Function Type
        public enum actFuncType { logistic, identity };
        actFuncType actFunc = actFuncType.logistic;

        //Pre-activation Output Save
        List<List<double>> output = new List<List<double>>();

        //This Neuron's Final, Post-activation Output
        public List<List<double>> postActOutput = new List<List<double>>();

        //Delta Calculated from this Neuron's Error and f Prime
        List<List<double>> delta = new List<List<double>>();

        //For Hidden Neurons, the Delta Sum Passed Down from the Layer Above
        public List<List<double>> handedDownDelta = new List<List<double>>();

        //New Weights with the Gradient Applied (stored for switching after iteration)
        public List<List<List<double>>> newWeights = new List<List<List<double>>>();

        //Empty Constructor
        public mNeuron()
        {

        }

        //Input Neuron Constructor
        public mNeuron (List<List<double>> output)
        {
            this.layer = 0;
            this.actFunc = actFuncType.identity;
            int d = 0;

            this.inputs.Add(new List<List<double>>());
            for(int i = 0; i < output.Count; i++)
            {
                d++;
                this.inputs[0].Add(new List<double>());
                this.beta.Add(new List<double>());
                for (int j = 0; j < output[0].Count; j++)
                {
                    this.inputs[0][i].Add(output[i][j]);
                    this.beta[i].Add(0d);
                }
            }

            this.weights.Add(MyMath.makeIdentityMatrix(d));
        }

        //Hidden and Output Neuron Constructor
        public mNeuron (int layer, int depth, actFuncType func, List<List<int>> inputNeu, double weightVar, double betaVar, int mCount, int nCount)
        {
            this.layer = layer;
            this.depth = depth;
            this.actFunc = func;

            foreach(List<int> vec in inputNeu)
            {
                List<int> curVec = new List<int>();
                foreach (int i in vec)
                {
                    curVec.Add(i);
                }
                this.inputNeurons.Add(curVec);
            }

            Random rng = main.rng;

            for (int k = 0; k < inputNeu.Count; k++)
            {
                this.weights.Add(new List<List<double>>());
                for (int i = 0; i < mCount; i++)
                {
                    this.weights[k].Add(new List<double>());
                    for (int j = 0; j < mCount;  j++)
                    {
                        this.weights[k][i].Add(weightVar * rng.NextDouble());
                    }
                }
            }

            for (int i = 0; i < mCount; i++)
            {
                this.beta.Add(new List<double>());
                for (int j = 0; j < nCount; j++)
                {
                    this.beta[i].Add(betaVar);
                }
            }
        }

        //Get Values from all Input Neurons' Coordinates
        public void GetInputs(mNeuron[,] nn)
        {
            this.inputs.Clear();

            for (int i = 0; i < this.inputNeurons.Count(); i++)
            {
                List<List<double>> target = nn[this.inputNeurons[i][0], this.inputNeurons[i][1]].postActOutput;

                this.inputs.Add(new List<List<double>>());

                for (int j = 0; j < target.Count; j++)
                {
                    this.inputs[i].Add(new List<double>());

                    for(int k = 0; k < target[0].Count; k++)
                    {
                        this.inputs[i][j].Add(target[j][k]);
                    }
                }
            }
        }

        //Forward Propagate this Neuron
        public void ForwardPropag(mNeuron[,] nn)
        {
            if(this.layer != 0)
            {
                this.GetInputs(nn);
            }

            this.output = main.createNewMatrix(this.inputs[0].Count, this.inputs[0][0].Count);
            this.postActOutput = main.createNewMatrix(this.inputs[0].Count, this.inputs[0][0].Count);
            this.handedDownDelta = main.createNewMatrix(this.inputs[0].Count, this.inputs[0][0].Count);

            for (int i = 0; i < this.inputs.Count; i++)
            {
                List<List<double>> curInput = this.inputs[i];
                List<List<double>> result = MyMath.matMulti(this.weights[i], curInput);
                this.output = MyMath.matAdd(this.output, result);
            }

            if (this.actFunc == actFuncType.logistic)
            {
                for (int m = 0; m < output.Count; m++)
                {
                    for(int n = 0; n < output[0].Count; n++)
                    {
                        this.postActOutput[m][n] = MyMath.logisticFunc(this.output[m][n], this.beta[m][n]);
                    }
                }
            }
            else if (this.actFunc == actFuncType.identity)
            {
                for (int m = 0; m < output.Count; m++)
                {
                    for (int n = 0; n < output[0].Count; n++)
                    {
                        this.postActOutput[m][n] = MyMath.identityFunc(this.output[m][n], this.beta[m][n]);
                    }
                }
            }
        }

        //Back Propagate for Output Neuron
        public void BackPropag(mNeuron[,] net, List<List<double>> expOut, double learnRate, double bLearnRate)
        {
            List<List<double>> error = main.createNewMatrix(this.postActOutput.Count, this.postActOutput[0].Count);
            for (int i = 0; i < expOut.Count; i++)
            {
                for(int j = 0; j < expOut[0].Count; j++)
                {
                    if (expOut[i][j] != -2)
                    {
                        error[i][j] = this.postActOutput[i][j] - expOut[i][j];
                    }
                    else
                    {
                        error[i][j] = 0;
                    }
                }
            }

            List<List<double>> fPrime = new List<List<double>>();
            if(this.actFunc == actFuncType.logistic)
            {
                for(int i = 0; i < this.output.Count; i++)
                {
                    fPrime.Add(new List<double>());
                    for (int j = 0; j < this.output[0].Count; j++)
                    {
                        fPrime[i].Add(MyMath.logisticPrimeFunc(this.output[i][j]));
                    }
                }
            }
            else if(this.actFunc == actFuncType.identity){
                for (int i = 0; i < this.output.Count; i++)
                {
                    fPrime.Add(new List<double>());
                    for (int j = 0; j < this.output[0].Count; j++)
                    {
                        fPrime[i].Add(1d);
                    }
                }
            }

            List<List<double>> result = MyMath.pointwiseMatMulti(error, fPrime);
            this.delta = result;
            this.beta = MyMath.matSub(this.beta, MyMath.scalarProd(bLearnRate, this.delta));

            for(int i = 0; i < this.inputNeurons.Count; i++)
            {
                int m = inputNeurons[i][0];
                int n = inputNeurons[i][1];

                net[m, n].handedDownDelta = MyMath.matAdd(net[m, n].handedDownDelta, this.delta);

                List<List<double>> matrix = this.weights[i];
                List<List<double>> gradient = MyMath.matMulti(this.delta, MyMath.transposeMat(this.inputs[i]));
                this.newWeights.Add(MyMath.matSub(matrix, MyMath.scalarProd(learnRate, gradient)));
            }
        }

        //Back Propagate for Hidden Neuron
        public void BackPropag(mNeuron[,] net, double learnRate, double bLearnRate)
        {
            List<List<double>> fPrime = new List<List<double>>();
            if (this.actFunc == actFuncType.logistic)
            {
                for (int i = 0; i < this.output.Count; i++)
                {
                    fPrime.Add(new List<double>());
                    for (int j = 0; j < this.output[0].Count; j++)
                    {
                        fPrime[i].Add(MyMath.logisticPrimeFunc(this.output[i][j]));
                    }
                }
            }
            else if (this.actFunc == actFuncType.identity)
            {
                for (int i = 0; i < this.output.Count; i++)
                {
                    fPrime.Add(new List<double>());
                    for (int j = 0; j < this.output[0].Count; j++)
                    {
                        fPrime[i].Add(1d);
                    }
                }
            }

            for (int i = 0; i < this.inputNeurons.Count; i++)
            {
                List<List<double>> error = MyMath.matMulti(MyMath.transposeMat(this.weights[i]), this.handedDownDelta);
            
                List<List<double>> result = MyMath.pointwiseMatMulti(error, fPrime);
                this.delta = result;
                this.beta = MyMath.matSub(this.beta, MyMath.scalarProd(bLearnRate, this.delta));

                int m = inputNeurons[i][0];
                int n = inputNeurons[i][1];

                net[m, n].handedDownDelta = MyMath.matAdd(net[m, n].handedDownDelta, this.delta);

                List<List<double>> matrix = this.weights[i];
                List<List<double>> gradient = MyMath.matMulti(this.delta, MyMath.transposeMat(this.inputs[i]));
                this.newWeights.Add(MyMath.matSub(matrix, MyMath.scalarProd(learnRate, gradient)));
            }
        }

        //Switch to New Weights, Clear New Weights Variable for Next Iteration
        public void switchToNewWeights()
        {
            for (int i = 0; i < this.weights.Count; i++)
            {
                for (int j = 0; j < this.weights[0].Count; j++)
                {
                    for (int k = 0; k < this.weights[0][0].Count; k++)
                    {
                        this.weights[i][j][k] = this.newWeights[i][j][k];
                    }
                }
            }
            this.newWeights.Clear();
        }
    }
}
