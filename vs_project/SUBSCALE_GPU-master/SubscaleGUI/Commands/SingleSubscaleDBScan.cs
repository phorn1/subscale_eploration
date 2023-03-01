using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SubscaleGUI.Commands
{
    internal class SingleSubscaleDBScan : ICommand
    {
        private string sourceFilename;
        private string targetFilename;

        private int minPoints;
        private double epsilon;

        public SingleSubscaleDBScan(string sourceFilename, string targetFilename, int minPoints, int epsilon)
        {
            this.sourceFilename = sourceFilename;
            this.targetFilename = targetFilename;
            this.minPoints = minPoints;
            this.epsilon = epsilon;
        }

        public void execute()
        {
            throw new NotImplementedException();
        }
    }
}
