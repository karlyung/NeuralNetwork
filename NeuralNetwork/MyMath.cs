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
    public class MyMath
    {
        //Make Identity Matrix
        public static List<List<double>> makeIdentityMatrix(int d)
        {
            if(d <= 0)
            {
                throw new Exception("Cannot make identity matrix with dimension less than or equal to 0!");
            }

            List<List<double>> result = new List<List<double>>();
            for(int i = 0; i < d; i++)
            {
                result.Add(new List<double>());
                for(int j = 0; j < d; j++)
                {
                    if(i == j)
                    {
                        result[i].Add(1d);
                    }
                    else
                    {
                        result[i].Add(0d);
                    }
                }
            }
            return result;
        }

        //Turn Vector into Horizontal Matrix
        public static List<List<double>> makeHorizonColMat(List<double> x)
        {
            if (x.Count <= 0)
            {
                throw new Exception("Cannot turn vector with dimension less than or equal to 0 into horizontal matrix!");
            }

            List<List<double>> result = new List<List<double>>();
            result.Add(new List<double>());
            for(int i = 0; i < x.Count; i++)
            {
                result[0].Add(x[i]);
            }
            return result;
        }

        //Turn Vector into Vertical Matrix
        public static List<List<double>> makeVertiColMat(List<double> x)
        {
            if (x.Count <= 0)
            {
                throw new Exception("Cannot turn vector with dimension less than or equal to 0 into vertical matrix!");
            }

            List<List<double>> result = new List<List<double>>();
            for (int i = 0; i < x.Count; i++)
            {
                result.Add(new List<double>());
                result[i].Add(x[i]);
            }
            return result;
        }

        //Transpose Matrix
        public static List<List<double>> transposeMat(List<List<double>> x)
        {
            if (x.Count <= 0)
            {
                throw new Exception("Cannot transpose matrix with m dimension less than or equal to 0!");
            }

            for (int i = 0; i < x.Count; i++)
            {
                if (x[i].Count <= 0)
                {
                    throw new Exception("Cannot transpose matrix with n dimension less than or equal to 0!");
                }
                if (x[i].Count != x[0].Count)
                {
                    throw new Exception("Cannot transpose matrix with different n dimensions across rows!");
                }
            }

            List<List<double>> result = Enumerable.Repeat(Enumerable.Repeat(0d, x.Count).ToList(), x[0].Count).ToList();

            for (int i = 0; i < x.Count; i++)
            {
                for (int j = 0; j < x[0].Count; j++)
                {
                    result[j][i] = x[i][j];
                }
            }
            return result;
        }

        //Multiply Matrix with Scalar
        public static List<List<double>> scalarProd(double s, List<List<double>> x)
        {
            if (x.Count <= 0)
            {
                throw new Exception("Cannot multiply scalar with matrix with m dimension less than or equal to 0!");
            }

            for (int i = 0; i < x.Count; i++)
            {
                if (x[i].Count <= 0)
                {
                    throw new Exception("Cannot multiply scalar with matrix with n dimension less than or equal to 0!");
                }
                if (x[i].Count != x[0].Count)
                {
                    throw new Exception("Cannot multiply scalar with matrix with different n dimensions across rows!");
                }
            }

            List<List<double>> result = new List<List<double>>();
            for(int i = 0; i < x.Count; i++)
            {
                result.Add(new List<double>());
                for(int j = 0; j < x[0].Count; j++)
                {
                    result[i].Add(x[i][j] * s);
                }
            }
            return result;
        }

        //Multiply Vector with Matrix
        public static double colMatMulti(List<double> x, List<double> y)
        {
            if (x.Count <= 0 || y.Count <= 0)
            {
                throw new Exception("Cannot multiply column vectors that include one or two with length less than or equal to 0!");
            }

            if(x.Count != y.Count)
            {
                throw new Exception("Cannot multiply column vectors of different lengths!");
            }

            double result = 0.0;

            for(int i = 0; i < x.Count && i < y.Count; i++)
            {
                result += x[i] * y[i];
            }

            return result;
        }

        //Multiply Matrix with Matrix
        public static List<List<double>> matMulti(List<List<double>> x, List<List<double>> y)
        {
            if (x[0].Count != y.Count)
            {
                throw new Exception("Cannot multiply matrices where matrix-1's n doesnt not equal to matrix-2's m!");
            }

            if (x.Count <= 0)
            {
                throw new Exception("Cannot multiply matrices with matrix-1's m dimension less than or equal to 0!");
            }

            for (int i = 0; i < x.Count; i++)
            {
                if (x[i].Count <= 0)
                {
                    throw new Exception("Cannot multiply matrices with matrix-1's n dimension less than or equal to 0!");
                }
                if (x[i].Count != x[0].Count)
                {
                    throw new Exception("Cannot multiply matrices with matrix-1 having different n dimensions across rows!");
                }
            }

            if (y.Count <= 0)
            {
                throw new Exception("Cannot multiply matrices with matrix-2's m dimension less than or equal to 0!");
            }

            for (int i = 0; i < y.Count; i++)
            {
                if (y[i].Count <= 0)
                {
                    throw new Exception("Cannot multiply matrices with matrix-2's n dimension less than or equal to 0!");
                }
                if (y[i].Count != y[0].Count)
                {
                    throw new Exception("Cannot multiply matrices with matrix-2 having different n dimensions across rows!");
                }
            }

            List<List<double>> result = new List<List<double>>();

            for (int i = 0; i < x.Count; i++)
            {
                result.Add(new List<double>());
                for (int j = 0; j < y[0].Count; j++)
                {
                    result[i].Add(0d);
                }
            }

            for (int i = 0; i < x.Count; i++)
            {
                for (int j = 0; j < y[0].Count; j++)
                {
                    for (int k = 0; k < x[0].Count; k++)
                    {
                        result[i][j] += x[i][k] * y[k][j];
                    }
                }
            }
            
            return result;
        }

        //Pointwise Multiply Matrix with Matrix
        public static List<List<double>> pointwiseMatMulti(List<List<double>> x, List<List<double>> y)
        {
            if (x.Count != y.Count)
            {
                throw new Exception("Cannot pointwise multiply matrices where matrix-1's m doesnt not equal to matrix-2's m!");
            }

            if (x[0].Count != y[0].Count)
            {
                throw new Exception("Cannot pointwise multiply matrices where matrix-1's n doesnt not equal to matrix-2's n!");
            }

            if (x.Count <= 0)
            {
                throw new Exception("Cannot pointwise multiply matrices with matrix-1's m dimension less than or equal to 0!");
            }

            for (int i = 0; i < x.Count; i++)
            {
                if (x[i].Count <= 0)
                {
                    throw new Exception("Cannot pointwise multiply matrices with matrix-1's n dimension less than or equal to 0!");
                }
                if (x[i].Count != x[0].Count)
                {
                    throw new Exception("Cannot pointwise multiply matrices with matrix-1 having different n dimensions across rows!");
                }
            }

            if (y.Count <= 0)
            {
                throw new Exception("Cannot pointwise multiply matrices with matrix-2's m dimension less than or equal to 0!");
            }

            for (int i = 0; i < y.Count; i++)
            {
                if (y[i].Count <= 0)
                {
                    throw new Exception("Cannot pointwise multiply matrices with matrix-2's n dimension less than or equal to 0!");
                }
                if (y[i].Count != y[0].Count)
                {
                    throw new Exception("Cannot pointwise multiply matrices with matrix-2 having different n dimensions across rows!");
                }
            }

            List<List<double>> result = new List<List<double>>();

            for(int i = 0; i < x.Count; i++)
            {
                result.Add(new List<double>());
                for(int j = 0; j < x[0].Count; j++)
                {
                    result[i].Add(x[i][j] * y[i][j]);
                }
            }
            return result;
        }

        //Multiply Column Vector with Matrix
        public static List<double> colTimesMat(List<double> x, List<List<double>> y)
        {
            if (x.Count != y[0].Count)
            {
                throw new Exception("Cannot multiply vector and matrix when vector's m does not equal to matrix's n!");
            }

            if (x.Count <= 0)
            {
                throw new Exception("Cannot multiply vector and matrix when vector's m dimension is less than or equal to 0!");
            }

            if (y.Count <= 0)
            {
                throw new Exception("Cannot multiply vector and matrix with matrix's m dimension less than or equal to 0!");
            }

            for (int i = 0; i < y.Count; i++)
            {
                if (y[i].Count <= 0)
                {
                    throw new Exception("Cannot multiply vector and matrix with matrix's n dimension less than or equal to 0!");
                }
                if (y[i].Count != y[0].Count)
                {
                    throw new Exception("Cannot multiply vector and matrix with matrix having different n dimensions across rows!");
                }
            }

            List<double> result = new List<double>();

            for(int i = 0; i < y[0].Count; i++)
            {
                result.Add(0d);
            }

            for (int j = 0; j < y[0].Count; j++)
            {
                for (int k = 0; k < x.Count; k++)
                {
                    result[j] += x[k] * y[k][j];
                }
            }

            return result;
        }

        //Add Vectors
        public static List<double> colAdd(List<double> x, List<double> y)
        {
            if (x.Count != y.Count)
            {
                throw new Exception("Cannot add vectors with different dimensions!");
            }

            if (x.Count <= 0)
            {
                throw new Exception("Cannot add vectors when vector-1's m is less than or equal to 0!");
            }

            if (y.Count <= 0)
            {
                throw new Exception("Cannot add vectors when vector-2's m is less than or equal to 0!");
            }

            List<double> result = new List<double>();
            for(int i = 0; i < x.Count; i++)
            {
                result.Add(x[i] + y[i]);
            }
            return result;
        }

        //Subtract Vectors
        public static List<double> colSub(List<double> x, List<double> y)
        {
            if (x.Count != y.Count)
            {
                throw new Exception("Cannot subtract vectors with different dimensions!");
            }

            if (x.Count <= 0)
            {
                throw new Exception("Cannot subtract vectors when vector-1's m is less than or equal to 0!");
            }

            if (y.Count <= 0)
            {
                throw new Exception("Cannot subtract vectors when vector-2's m is less than or equal to 0!");
            }

            List<double> result = new List<double>();
            for(int i = 0; i < x.Count; i++)
            {
                result.Add(x[i] - y[i]);
            }
            return result;
        }

        //Subtract Matrices
        public static List<List<double>> matSub(List<List<double>> x, List<List<double>> y)
        {
            if (x.Count != y.Count)
            {
                throw new Exception("Cannot subtract matrices where matrix-1's m doesnt not equal to matrix-2's m!");
            }

            if (x[0].Count != y[0].Count)
            {
                throw new Exception("Cannot subtract matrices where matrix-1's n doesnt not equal to matrix-2's n!");
            }

            if (x.Count <= 0)
            {
                throw new Exception("Cannot subtract matrices with matrix-1's m dimension less than or equal to 0!");
            }

            for (int i = 0; i < x.Count; i++)
            {
                if (x[i].Count <= 0)
                {
                    throw new Exception("Cannot subtract matrices with matrix-1's n dimension less than or equal to 0!");
                }
                if (x[i].Count != x[0].Count)
                {
                    throw new Exception("Cannot subtract matrices with matrix-1 having different n dimensions across rows!");
                }
            }

            if (y.Count <= 0)
            {
                throw new Exception("Cannot subtract matrices with matrix-2's m dimension less than or equal to 0!");
            }

            for (int i = 0; i < y.Count; i++)
            {
                if (y[i].Count <= 0)
                {
                    throw new Exception("Cannot subtract matrices with matrix-2's n dimension less than or equal to 0!");
                }
                if (y[i].Count != y[0].Count)
                {
                    throw new Exception("Cannot subtract matrices with matrix-2 having different n dimensions across rows!");
                }
            }

            List<List<double>> result = new List<List<double>>();
            for(int i = 0; i < x.Count; i++)
            {
                result.Add(new List<double>());
                for(int j = 0; j < x[0].Count; j++)
                {
                    result[i].Add(x[i][j] - y[i][j]);
                }
            }
            return result;
        }

        //Add Matrices
        public static List<List<double>> matAdd(List<List<double>> x, List<List<double>> y)
        {
            if (x.Count != y.Count)
            {
                throw new Exception("Cannot add matrices where matrix-1's m doesnt not equal to matrix-2's m!");
            }

            if (x[0].Count != y[0].Count)
            {
                throw new Exception("Cannot add matrices where matrix-1's n doesnt not equal to matrix-2's n!");
            }

            if (x.Count <= 0)
            {
                throw new Exception("Cannot add matrices with matrix-1's m dimension less than or equal to 0!");
            }

            for (int i = 0; i < x.Count; i++)
            {
                if (x[i].Count <= 0)
                {
                    throw new Exception("Cannot add matrices with matrix-1's n dimension less than or equal to 0!");
                }
                if (x[i].Count != x[0].Count)
                {
                    throw new Exception("Cannot add matrices with matrix-1 having different n dimensions across rows!");
                }
            }

            if (y.Count <= 0)
            {
                throw new Exception("Cannot add matrices with matrix-2's m dimension less than or equal to 0!");
            }

            for (int i = 0; i < y.Count; i++)
            {
                if (y[i].Count <= 0)
                {
                    throw new Exception("Cannot add matrices with matrix-2's n dimension less than or equal to 0!");
                }
                if (y[i].Count != y[0].Count)
                {
                    throw new Exception("Cannot add matrices with matrix-2 having different n dimensions across rows!");
                }
            }

            List<List<double>> result = new List<List<double>>();
            for (int i = 0; i < x.Count; i++)
            {
                result.Add(new List<double>());
                for (int j = 0; j < x[0].Count; j++)
                {
                    result[i].Add(x[i][j] + y[i][j]);
                }
            }
            return result;
        }

        //Outputs f_logisticSigmoid(x + b)
        public static double logisticFunc(double x, double b)
        {
            double result = double.MinValue; //done for error identification

            try
            {
                //must have -0.5 otherwise negative inputs become positive, big no no
                result = (1d / (1d + Math.Pow(Math.E, -(x + b)))) - 0.5d;
            }
            catch (Exception)
            {
                throw new Exception("Sigmoid function error!");
            }
            return result;
        }

        //The Derivative of logisticFunc()
        public static double logisticPrimeFunc(double x)
        {
            return Math.Exp(x) / Math.Pow((Math.Exp(x) + 1), 2);
        }

        public static double identityFunc(double x, double b)
        {
            double result = double.MinValue; //done for error identification

            try
            {
                result = (x + b);
            }
            catch (Exception)
            {
                throw new Exception("Identity function error!");
            }
            return result;
        }

        //Outputs f_tanhSigmoid(x + b)
        public static double tanhFunc(double x, double b)
        {
            double result = double.MinValue; //done for error identification

            try
            {
                result = Math.Tanh(x + b);
            }
            catch (Exception)
            {
                throw new Exception("Tanh function error!");
            }
            return result;
        }

        //Outputs f_ReLU(x + b)
        public static double reluFunc(double x, double b)
        {
            double result = double.MinValue; //done for error identification

            try
            {
                result = Math.Max(0, x + b);
            }
            catch (Exception){
                throw new Exception("ReLU function error!");
            }
            return result;
        }
    }
}
