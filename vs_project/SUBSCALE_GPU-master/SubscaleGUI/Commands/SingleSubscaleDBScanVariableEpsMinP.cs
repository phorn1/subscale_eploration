using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SubscaleGUI.Commands
{
    internal class SingleSubscaleDBScanVariableEpsMinP : ICommand
    {
        private string sourceFilename;
        private string targetFilename;

        private double subscaleEps;
        private int subscaleMinPoints;

        private List<double> dbscanEps;
        private List<int> dbscanMinPoints;

        public SingleSubscaleDBScanVariableEpsMinP(string sourceFilename, string targetFilename, double subscaleEps, int subscaleMinPoints, List<double> dbscanEps, List<int> dbscanMinPoints)
        {
            this.sourceFilename = sourceFilename;
            this.targetFilename = targetFilename;
            this.subscaleEps = subscaleEps;
            this.subscaleMinPoints = subscaleMinPoints;
            this.dbscanEps = dbscanEps;
            this.dbscanMinPoints = dbscanMinPoints;
        }

        public void execute()
        {
            throw new NotImplementedException();
        }
    }
}
