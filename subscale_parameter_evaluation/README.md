# Parameter evaluation for Subscale/DBSCAN
This module provides a convenient way to test different Subscale and 
DBSCAN parameters using synthetically generated data. The module includes four files, each with its own functionality:
## generator.py
This Python script generates synthetic data that can be used for testing different parameter configurations.
## SubScaleExtended.jar
This Java executable provides a command-line interface for finding the maximal subspaces. 
The DBSCAN functionality is disabled in this version.
## subscale_explorer.py
This Python file contains the **SubscaleExplorer** class, which can be used to explore different parameter configurations. 
It uses **generator.py** to generate data, **SubScaleExtended.jar** for the subscale algorithm, and Scikit-learn for DBSCAN. 
Three evaluation methods are implemented: Cluster Error (CE-Score), F1-Score, and RNIA-Score.
```python
# Example usage
from subscale_explorer import SubscaleExplorer
sub = SubscaleExplorer()
sub.generate_database(n=1000, d=200, c=10, sub_n=20, sub_d=10, std=0.1)
sub.subscale(epsilon=0.3, minpts=5)
sub.dbscan(eps=0.3, min_samples=10, adjust_epsilon=True)
sub.score("ce")
```

## subscale_analysis.ipynb
This Jupyter Notebook contains different test runs and evaluations of different datasets and parameter configurations based on **subscale_explorer.px**. 
It is used to visualize the results of the parameter evaluations.
