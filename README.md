# NeurIPS - 2020: Non Euclidean Universal Approximation
### Numerical Example: *DNN Layer Constrution using our Theory vs. DNN Layer Construction Failing Assumptions(s).*

---

## Requirements

To install requirements:
*  Install [Anaconda](https://www.anaconda.com/products/individual)  version 4.8.2.
* Create Conda Environment
``` pyhton
# cd into the same directory as this README file.

conda create python=3.8 --name architopes \
conda activate architopes \
pip install -r requirements.txt
```


## Preprocessing, Training, and Evaluating
1. Specify the parameters related to each set and the space of hyper parameters in: Hyperparameter_Grid.py 

2. Preprocessing data, train models and obtaining predictions can all be done by executing the following command:
python3.8 Example.py


## Results

Our models and their benchmarks achieves the following performance on:

### [California Housing Price Dataset](https://github.com/ageron/handson-ml/tree/master/datasets/housing)

The house prices were multiplied by $10^{-5}$ to avoid exploding gradient issues.

1. For Train:

|    |  Good I(*) | Good II(†) |     Bad  | Vanilla   |
|--- |------------| ---------- |--------- |---------- |
|MAE  |  0.252428  |  0.295724   | 0.887295  |  0.283949  |
|MSE  |  0.173975  |  0.233633   | 1.408521  |  0.209303  |
|MAPE | 12.920774  | 15.667824   | 8.698249  | 14.878316  |

1. For Test:


|    |  Good I(*) | Good II(†) |     Bad  | Vanilla   |
|--- |------------| ---------- |--------- |---------- |
| MAE    | 0.317888   | 0.319974   | 0.875507   | 0.320133 |
| MSE    | 0.247136   | 0.258657   | 1.354547   | 0.257305 |
| MAPE  | 16.714252  | 17.626384  | 48.051326  | 17.427648 |



(*) Notation: In the paper "Good" is used to denote "Good I".
(†) Notation: In the paper "Rand" is used to denote "Good II".
