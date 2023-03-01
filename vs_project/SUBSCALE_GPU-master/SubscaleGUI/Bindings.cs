using System.Runtime.InteropServices;
using System;

namespace SubscaleGUI
{

    [StructLayout(LayoutKind.Sequential)]
    struct DataPointBinding
    {
        uint id;
        uint nValues;
        double[] values;
    };

    [StructLayout(LayoutKind.Sequential)]
    struct Subspace
    {
        uint[] dimensions;
        uint[] ids;
        uint nDimensions;
        uint nIds;
    };

    class SubscaleBindings
    {
        [DllImport("SubscaleGPU.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void readData(ref DataPointBinding[] points, string filepath, char delimiter);

        [DllImport("SubscaleGPU.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void executeSubscale(DataPointBinding[] points, ref IntPtr pClusterCandidates, uint nPoints, ref Subspace[] subspaces, ref uint nSubspaces, double eps, int minP);

        [DllImport("SubscaleGPU.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern void executeDBScan(DataPointBinding[] points, IntPtr clusterCandidates, ref Subspace[] clusterTableRet, ref uint nClusters, uint nPoints, double eps, int minP);
    }
}
