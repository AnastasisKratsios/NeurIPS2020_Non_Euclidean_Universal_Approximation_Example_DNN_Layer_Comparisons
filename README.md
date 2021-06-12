# NeurIPS - 2020: [Non Euclidean Universal Approximation](https://arxiv.org/abs/2006.02341)
Coauthored by:
- [Anastasis Kratsios](https://people.math.ethz.ch/~kratsioa/)
- [Ievgen Bilokopytov](https://apps.ualberta.ca/directory/person/bilokopy)

# Cite As:

    @inproceedings{NEURIPS2020_786ab8c4,
     author = {Kratsios, Anastasis and Bilokopytov, Ievgen},
     booktitle = {Advances in Neural Information Processing Systems},
     editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
     pages = {10635--10646},
     publisher = {Curran Associates, Inc.},
     title = {Non-Euclidean Universal Approximation},
     url = {https://proceedings.neurips.cc/paper/2020/file/786ab8c4d7ee758f80d57e65582e609d-Paper.pdf},
     volume = {33},
     year = {2020}
    }



### Numerical Example: *DNN Layer Constrution using our Theory vs. DNN Layer Construction Failing Assumptions(s).*

---

## Requirements

To install requirements:
*  Install [Anaconda](https://www.anaconda.com/products/individual)  version 4.8.2.
* Create Conda Environment
``` pyhton
# cd into the same directory as this README file.

pip install -r requirements.txt
```
---

## Organization of directory:
 - Data in the "inputs" sub-directory,
 - All model outputs go to the "outputs" subdirectory,
 - Jupyternotebook versions of the python files are found in the "Jupyternote book" subdirectory.  

---

## Preprocessing, Training, and Evaluating
1. Specify the parameters related to each set and the space of hyper parameters in: Hyperparameter_Grid.py 

2. Preprocessing data, train models and obtaining predictions can all be done by executing the following command:
python3.8 Example.py

---

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

---
Hyperparameter Grid Used in Training for the paper ["Non-Euclidean Universal Approximation"](https://arxiv.org/abs/2006.02341)

| Batch size | Epochs | Learning Rate | Height (Middle Layers) | Depth - Input Layers | Depth - Middle Layers | Depth - Output Layers |
|------------|--------|---------------|------------------------|----------------------|-----------------------|-----------------------|
|     16     |  200   |    0.0001     |         200            |          2           |           1           |            2          |
|      32    |  400   |    0.0005     |         250            |          3           |           2           |            3          |
|     -      |  800   |    0.005      |         400            |          4           |           -           |            4          |
|     -      |  1000  |      -        |         600            |          5           |           -           |            5          |
|     -      |  1200  |      -        |         800            |          -           |           -           |            -          |
|     -      |  -     |      -        |        1000            |          -           |           -           |            -          |


### Meta-parameters Used:
- n_jobs = 70 (Number of cores used in training).
- n_iter = 10 (The number of cross-validation iterations used/per model when performing the grid-search).
- CV_folds = 4 (The number of cross-validation folds used).
