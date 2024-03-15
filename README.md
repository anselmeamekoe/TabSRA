# Exploring Accuracy and Interpretability trade-off in Tabular Learning with Novel Attention-Based Models
 Apart from the predictive performance, interpretability is essential for :
 - uncovering hidden patterns in the data
 - providing meaningful justification of decisions made by machine learning mode
 - ...
   
 In this concern, an important question arises: should one use *inherently interpretable* models or explain full-complexity models such as XGBoost, Random Forest with post hoc tools?

In this repository, we provide some concrete numerical results that can guide practitioners (or researchers) in their choice between using inherently interpretable
solutions and explaining full-complexity models. 

This study includes, *TabSRAs*, an attention based inherently interpretable model which is proving to be a viable option for (i) generating stable or robust explanations, and (ii) incorporating
human knowledge during the training phase.

## What is the actual performance gap between the full-complexity state-of-the-art models and their inherently interpretable counterpartsin terms of accuracy?
| Model            | Rank (min)|   Rank (max)      |   Rank (mean)    | Rank (median)|Test score (mean)|Test score (median)|Test score (std)|Runing Time (mean)|Runing Time (meadian)|
|------------------|-----|-----|--------|--------|-------|--------|-------|---------|----------|
| DT               | 2   | 11  | 9.480  | 10     | 0.870 | 0.916  | 0.163 | 0.294   | 0.032  |
| EBM              | 1   | 9   | 4.578  | 4      | 0.961 | 0.985  | 0.067 | 97.837  | 19.737|
| EBM_S            | 1   | 10  | 6.751  | 7      | 0.933 | 0.958  | 0.086 | 23.997  | 5.144 |
| LR               | 6   | 11  | 10.703 | 11     | 0.761 | 0.840  | 0.232 | 21.124  | 19.716 |
| TabSRALinear     | 1   | 11  | 7.289  | 8      | 0.903 | 0.973  | 0.197 | 47.576  |  38.073|
|                  |                                                                                 |
| MLP              | 1   | 11  | 6.119  | 7      | 0.926 | 0.978  | 0.159 | 24.165  | 19.256 |
| ResNet           | 1   | 11  | 6.251  | 7      | 0.911 | 0.976  | 0.195 | 95.123  | 53.212 |
| SAINT            | 1   | 11  | 4.776  | 5      | 0.949 | 0.985  | 0.093 | 216.053 | 126.841 |
| FT Transformer   | 1   | 10  | 4.382  | 4      | 0.946 | 0.988  | 0.109 | 126.589 | 77.465  |
| RandomForest     | 1   | 9   | 3.501  | 3      | 0.987 | 0.994  | 0.019 | 39.030  | 8.252   |
| XGBoost          | 1   | 10  | 2.171  | 1      | 0.990 | 1.000  | 0.028 | 18.254  | 12.561  |


Predictive performance of models across a benchmark of 45 datasets (59 tasks) introduced in the paper "Why do tree-based models still outperform deep learning on typical tabular data?". We report the rank over all tasks, the relative test score (Accuracy/$R^2$) and running time (training+inference) in seconds.
The considered inherently interpretable models are:
- Decision Trees [(DT)](https://scikit-learn.org/stable/modules/tree.html)

TabSRA is a class of accurate tabular learning models with inherent intelligibility published at the 5th International Workshop on eXplainable Knowledge Discovery in Data Mining **XKDD 2023** and The 31th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning **ESANN 2023**.
In short, TabSRA contains a Self-Reinforcement Attention (SRA) block that is used to learn a *Reinforced* representation of the raw input through element-wise multiplication with the produced attention vector. The learned representation is aggregated by a highly transparent function (e.g., linear) that produces the final output. In this repository we propose the implementation of TabSRA with a linear aggregator namely **TabSRALinear**.

<img src="https://github.com/anselmeamekoe/TabSRA/blob/main/ESANN_XKDD/ressources/images/TabSRAs.svg" width=600px>

## Usage
### Prerequisites
First, clone the TabSRA repository, and install the requirements:
```bash
$ https://github.com/anselmeamekoe/TabSRA
$ cd TabSRA
```
  The main dependencies are:
  - python>=3.9.12
  - torch>=1.12.1
  - einops>=0.4.1
  - numpy>=1.21.5
  - pandas>=1.4.2
  - matplotlib>=3.5.1
  - seaborn>=0.11.2
### Experiments 
Reproduce experiments on benchmark datasets:
```bash
$ python main.py -h
```
For example to train the TabSRALinear on the first fold of the Churn Modeling datase:
```bash
$ python main.py --dataset Churn --fold 0 --mode train --lr 0.006835471440855879 --dropout_rate 0.2 --epochs 900 --batch_size 512 --eval_metric AUCROC --seed 42 --weight_decay 0.00038918784304332334 --device cuda:0 --verbose 0
```
This command line will:
  - print the training and validation AUCROC
  - save the trained model at TabSRA/Churn_0.pth

### Details on parameters and training 

```python
import torch
import torch.nn as nn
from SRAModels import TabSRALinear
from DataProcessing import load_benchmark
from utils import TrainModel,reset_seed_,Predict, load, save,LinearScaling

reset_seed_(42)
Model = TabSRALinear(dim_input,
                     dim_output,
                     dim_head = 8,
                     get_attention = True,
                     dropout_rate = 0.0,
                     activation = nn.ReLU(),
                     bias = True,
                     s =  nn.Identity(),
                     for_classif=True
)
Model = TrainModel(Model, 
                   train_set,
                   test_set= None,
                   save_path= None,
                   device= 'cpu',
                   epochs= 150,
                   batch_size= 256,
                   lr= 1e-2 ,
                   eval_every= 5,
                   weight_decay= 1e-5,
                   verbose= 1,
                   load_best_eval= True,
                   eval_metric= 'AUCROC'
)
```
Model parameters
 - ```dim_input```: int
   
   The input dimension or the number of features
 - ```dim_output```: int
   
   The output dimension. 1 for binary classification and regression problem
 - ```dim_head```: int (default=8)

   The attention head dimension , d_k in the paper.
 - ```get_attention```: bool (default=True)

   Whether to give attention weights or not
 - ```dropout_rate```: float (default=0.0) 

   The neuron dropout rate used  in the Key/Query encorder during the training.
 - ```activation```: (default=nn.ReLU()) ,
   
   The activation  function used in the Key/Query encorder. Any 1-Lipschitz activation is accepted (ex. ReLu, Sigmoid)
 - ```bias```: bool (default=True)

  Whether to use bias term in linear transformations
 - ```s```: bool (default=nn.Identity())
  
  The scaling for the attention weights ```s = LinearScaling(scale = dim_head**-0.5)``` result in scale free attention weights in [0,1] used in the paper.
 - ```for_classif```: bool (default=True)

 If it is binary classification model ```for_classif=True``` and ```for_classif=False``` for regression problems.
### Useful links
  - [Churn modeling classification example](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/BankChurn_Classification_Example.ipynb)
  - [Regression example](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/Synthetic3_Regression_Example.ipynb)
  - [2D visualization example](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/2D_Visualization_Reinforced_Vectors.ipynb)
  - [Stability study example](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/HelocFico_Classification_Example_LipschitzEstimate.ipynb)
### Todo 
TabSRA package with sklearn interface
### Acknowledgments
This work has been done in collaboration between BPCE Group, Laboratoire d'Informatique de Paris Nord (LIPN UMR 7030),  DAVID Lab UVSQ-Université Paris Saclay and was supported by the program Convention
Industrielle de Formation par la Recherche (CIFRE) of the Association Nationale de la Recherche et de la Technologie (ANRT).
### Citations
  If you find the code useful, please cite it by using the following BibTeX entry:
  ```
@inproceedings{kodjoEs23,
 author       = {Kodjo Mawuena Amekoe and
                  Mohamed Djallel Dilmi and
                  Hanene Azzag and
                  Zaineb Chelly Dagdia and
                    Mustapha Lebbah and
                  Gregoire Jaffre},
  title = {TabSRA: An Attention based Self-Explainable Model for Tabular Learning},
 booktitle = {The31th European Symposium on Artificial Neural Networks, Computational  Intelligence and Machine Learning (ESANN)},
  year         = {2023}
 }
  ```
  ```
@inproceedings{XKDD23,
 author       = {Kodjo Mawuena Amekoe and
                  Hanene Azzag and
                    Mustapha Lebbah and
                  Zaineb Chelly Dagdia and
                  Gregoire Jaffre},
  title = {A New Class of Intelligible Models for Tabular Learning},
 booktitle = {In The 5th International Workshop on eXplainable Knowledge Discovery in Data Mining (PKDD)-ECML-PKDD},
  year         = {2023}
 }
  ```
  ```
  https://arxiv.org/abs/2305.11684
  ```
