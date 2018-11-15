# 373 project

## data

`filter.py` :

    input: `datatest.txt`, `datatest2.txt`, `datatraining.txt`
    All data are mix and cv will be reimplement
    Use to shuffle data
    split feature X and y
    change label 0 to -1
    split as 3:7 for testing : training
    k-fold with k = 5, for training data
    output: `testset.p`,`kf.p`

## algorithms

`svm_kf.py` :

    input: `testset.p`,`kf.p`
    try svm in k-fold with all hyperpara constant
    k = 5
    output stdout: mean and variance
    ```
    0.9889506601806811
    1.130041178893725e-06
    ```
## graph:

`accuracy.py` :

    input: `datatest.txt`, `datatest2.txt`, `datatraining.txt`

    output: `*.png`

## draft

`svm.py` :

    understand sklearn




