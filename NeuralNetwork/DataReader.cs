//Import System Components
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

//File Name
namespace NeuralNetwork
{
    //Class Name
    class DataReader
    {
        //Read 1 Specific Column (column #colNum) from File
        public static List<double> readDoubleColumn(string filePath, int colNum)
        {
            StreamReader sr = null;

            List<double> result = new List<double>();

            sr = new StreamReader(filePath);

            string currentLine = " ";
            while (currentLine != null)
            {
                currentLine = sr.ReadLine();

                if (currentLine != null) //Could be simplified!
                {
                    string[] curLineSplit = currentLine.Split(',');
                    double curElement = 0d;
                    try
                    {
                        curElement = double.Parse(curLineSplit[colNum].Trim('"'));
                    }
                    catch
                    {
                        curElement = 0d;
                    }
                    result.Add(curElement);
                }
            }

            return result;
        }
    }
}
