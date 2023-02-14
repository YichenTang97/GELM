# GELM
 A python implementation of the discriminative graph regularized Extreme Learning Machine (GELM) proposed by Yong Peng et al. [1], with sklearn compatibility

## Introduction

I coded this GELM model based on papers by Yong Peng et al. [1] and Wei-Long Zheng et al. [2]. Please cite [1] (and [2]) when using this model. This model has been coded to include the sklearn estimator API, and could be used as a standard sklearn classifier.

I used this model to perform my electroencephalogram (EEG) analyses, hence the default hyperparameters were tuned toward my specific usage. Please always do some hyper parameter tunings before using the model on your dataset.

## Requirements
This model was coded and tested on Python 3.9 with the following libraries and versions (minor differences in versions should not affect the model outcomes):

```Python
numpy >= 1.21.6
scikit-learn >= 1.1.3
```

## Examples

```Python
TODO
```

## References
[1] Y. Peng, S. Wang, X. Long, and B. L. Lu, “Discriminative graph regularized extreme learning machine and its application to face recognition,” Neurocomputing, vol. 149, no. Part A, pp. 340–353, Feb. 2015, doi: 10.1016/J.NEUCOM.2013.12.065.
[2] W. L. Zheng, J. Y. Zhu, and B. L. Lu, “Identifying stable patterns over time for emotion recognition from eeg,” IEEE Trans. Affect. Comput., vol. 10, no. 3, pp. 417–429, 2019, doi: 10.1109/TAFFC.2017.2712143.