# Machine Learning Algorithms
[![Build Status](https://travis-ci.com/vivekkr12/machine-learning-algorithms.svg?branch=master)](https://travis-ci.com/vivekkr12/machine-learning-algorithms) [![Coverage Status](https://coveralls.io/repos/github/vivekkr12/machine-learning-algorithms/badge.svg?branch=master)](https://coveralls.io/github/vivekkr12/machine-learning-algorithms?branch=master)

Implementation of common machine learning algorithms from scratch in Python

|Algorithm|Math|Implementation|Demo|
|---------|------|--------------|----|
|Ridge Regression|[Theory and Derivation](theory/ridge_regression.ipynb)|[Python Implementation](pymlalgo/regression/ridge_regression.py)|[Demo](demo/ridge_regression_demo.ipynb)|
|K Means|[Theory and Algorithm](theory/k_means.ipynb)|[Python Implementation](pymlalgo/clustering/k_means.py)|[Demo](demo/k_means_demo.ipynb)|
|Principal Component Analysis|[Theory and Algorithm](theory/pca.ipynb)|[Python Implementation](pymlalgo/reduction/pca.py)|[Demo](demo/pca_demo.ipynb)|

## Dependencies
The package depends only on `numpy`. Running the demos will require additional packages such as `jupyter`, `pandas`
and `sklearn`. 

## Running the Demos
The demos are in ipynb notebooks. Make sure you have the dependencies - `pandas` and `sklearn` installed in your
environment. Then from the root of the project start the ipynb server by running `$ jupyter notebook` and navigate to
the directory `demo`.

## Installation 
To install the package locally, run the following from the root of the project
```bash
$ python setup.py install
```

To make a pip installable tar archive, run
```bash
$ python setup.py sdist
```
The tar file would be generated inside `dist` folder. The package can be installed using the tar archive by running
```bash
$ pip install pymlalgo-0.0.1.tar.gz
```

For details on how to install packages from tar archive, refer to [this link on StackOverflow](https://stackoverflow.com/questions/36014334/)

## Usage
Once you have the package installed, import the module, initialize the class with data and hyper parameters and then
train the model. For example to use `KMeans`:

```python
from pymlalgo.clustering.k_means import KMeans

model = KMeans(x, k, max_iter=200)
model.train()

# get assigned cluster
model.labels
```
