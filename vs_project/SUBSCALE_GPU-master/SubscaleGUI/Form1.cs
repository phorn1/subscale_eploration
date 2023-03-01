using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SubscaleGUI
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void buttonTest_Click(object sender, EventArgs e)
        {
            DataPointBinding[] points = null;
            Subspace[] subspaces = null;
            Subspace[] clusterTableRet = null;

            uint nSubspaces = 0;
            uint nClusters = 0;

            IntPtr candidates = new IntPtr();

            SubscaleBindings.readData(ref points, "E:\\INFM Projekt\\repos\\subscale_latex\\vs_project\\SUBSCALE_GPU-master\\SubscaleGPU\\data\\sample5.csv", ',');

            SubscaleBindings.executeSubscale(points, ref candidates, (uint)points.Length, ref subspaces, ref nSubspaces, 0.02, 4);

            SubscaleBindings.executeDBScan(points, candidates, ref clusterTableRet, ref nClusters, (uint)points.Length, 0.02, 4);
        }
    }
}
