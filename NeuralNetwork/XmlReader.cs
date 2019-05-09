using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace NeuralNetwork
{
    class XmlReader //Not Used (may be further developed in the future)
    {
        static string filePath = "C:\\Users\\Karl\\Desktop\\nnConfig.xml";
        XmlTextReader reader = new XmlTextReader(filePath);

        public int readConfig()
        {
            try
            {
                while (reader.Read())
                {
                    
                }
                return 0;
            }
            catch (Exception)
            {
                return 1;
            }
        }
    }
}
