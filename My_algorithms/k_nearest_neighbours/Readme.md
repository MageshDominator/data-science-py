This is an implementation of K-Nearest Neighbors(KNN) Classification with Iris dataset(availabe in "iris.data.txt" file).

Algorithm:
	KNN is implemented in "k_nearest_neighbors.py" file, wich can be imported to work on any kind of Classification problems.
	Note:
	1. Specify number of neighbors to be considered(k)

Dataset:
	Dataset is available as a text file "iris.data.txt" which is then converted to .csv file using my script in my repo "data-science-py\PythonTools\Text2CSV\txtToCsv.py"

Data Preprocessing:
	Data is extracted from .csv file and preprocessed in "DataProcessing.py" which also has an implementation of KNN(our own algorithm) and evaluation of the model as well.

KNN using scikit-learn:
	The above procedures being directly implement in "k_nearest_neighbors_scikit.py" using scikit-learn's KNN-Classifier model. 
