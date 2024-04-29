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
|DT | 2 | 12 | 10.476 | 11|                      0.868 |  0.907 | 0.163 |     0.294 |   0.032 |
| EBM\_S | 1 | 11 |  7.692 |  8 |                      0.931 |  0.955 | 0.087 |    23.997 |   5.144 |
| EBM | 1 | 10 | 5.477 |  5 |                      0.959 |  0.982 | 0.067 |    97.837 |  19.737 |
| LR | 7 | 12 | 11.701 | 12 |                      0.760 |  0.839 | 0.232 |    21.124 |  19.716 |
|TabSRALinear | 1 | 12 |  8.225 |  9 |                      0.901 |  0.971 | 0.197 |   47.576 |  38.073 |
|                  |                                                                                 |
| MLP | 1 | 12 |  6.992 |  8 |                      0.924 |  0.973 | 0.159 |    24.165 |  19.256 |
| ResNet | 1 | 12 |  7.120 |  8 |                     0.909 |  0.975 | 0.195 |    95.123 |  53.212 |
|SAINT | 1 | 12 |  5.625 |  6 |                      0.946 |  0.982 | 0.093 |   216.053 | 126.841 |
| FT-Transformer | 1 | 11 |  5.203 | 5 |                      0.944 |  0.984 | 0.109 |   126.589 |  77.465 |
  Random Forest | 1 | 10 |  4.214 |  4 |                      0.985 |  0.992 | 0.021 |   39.030 |  8.252 |
| XGBoost | 1 | 11 |  2.728 | 2 |                      0.988 |  0.998 | 0.029 |    18.254 |  12.561 |
| CatBoost |1 | 10 |  2.545 |  2 |                      0.991 |  0.999 | 0.021 |    12.176 |   4.025 |


Predictive performance of models across a benchmark of 45 datasets (59 tasks) introduced in the paper "Why do tree-based models still outperform deep learning on typical tabular data?". We report the rank over all tasks, the relative test score (Accuracy/ $R^2$ ) and running time (training+inference) in seconds.

The considered inherently interpretable models are:
- Decision Trees [(DT)](https://scikit-learn.org/stable/modules/tree.html)
- Explainable Boosting Machine [EBMs](https://github.com/interpretml/interpret)
  - EBM: EBMs with **pairwise interaction** terms
  - EBM_S: EBMs **without pairwise interaction** terms
- Linear/Logistic Regression (LR): pytorch is used for the implementation
- TabSRALinear: an instantiation of TabSRAs, which imitates the formulation of classical Linear models. More details or in the papers [ESANN](https://www.esann.org/sites/default/files/proceedings/2023/ES2023-37.pdf), [ECML@XKDD](http://xkdd2023.isti.cnr.it/papers/426.pdf)
  
Among full-complexty models, we considered:
- MultiLayer Perceptron (MLP): pytorch is used for the implementation
- [ResNet](https://arxiv.org/pdf/2106.11959.pdf)
- [SAINT](https://arxiv.org/pdf/2106.01342.pdf)
- [FT Transformer](https://arxiv.org/pdf/2106.11959.pdf)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)

### What about the robustness of explanations, are the produced feature attributions similar for similar inputs?
<img src="https://github.com/anselmeamekoe/TabSRA/blob/main/ressources/Stability_CardFraud_Split0_hideen1_1E_3.png" width=600px>

Changes in feature attributions (the lower the better) using the [CreditCardFraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset. 

LR = Logistic Regression, SRA=TabSRALinear, XGB_SHAP=XGBoost+TreeSHAP


## Usage
### Prerequisites
Create a new python environment, install the [requirements](https://github.com/anselmeamekoe/TabSRA/blob/main/requirements.txt)
### Replicating the analyses/results on the predictive performance
1. Clone this repository of your machine
2. Dowanload the random search results using the links:
   - [full-complexity models](https://figshare.com/ndownloader/files/40081681)
   - [inherently interpretable models](https://drive.google.com/file/d/1sNmzjmzQg4ym7g62ZOi699QjVqE20W6H/view?usp=sharing)
3.  Copy and paste the downloaded files to ```TabSRA/tabular-benchmark/analyses/```
4.  Run the [Notebook](https://github.com/anselmeamekoe/TabSRA/blob/main/tabular-benchmark/analyses/results.ipynb) for reproducing results
   
NB: To use the notebook, you will need to install it in the python environment you have created using pip for example

### Replicating the results on the robustness of explanations 
1. Use the [Notebook](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/Robustness_Study_CreditCardFraud.ipynb) for the example on the Credit Card Fraud dataset
2. Use the [Notebook](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/Robustness_Study_HelocFico.ipynb) for the example on the Heloc Fico dataset

### Replicating the results on the study of the faithfulness of explanations
1. [Linear functions](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/Synthetic1_Regression_Example.ipynb)
2. [Parabolic functions](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/Synthetic2_Regression_Example.ipynb) 
3. [Linear functions with interactions](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/Synthetic3_Regression_Example.ipynb) 

### Real-world examples of TabSRALinear's unique capabilities
1. [Churn modeling with bias correction](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/Application1_BankChurnModeling.ipynb)
2. [Credit Default modeling with a group of correlated features](https://github.com/anselmeamekoe/TabSRA/blob/main/notebooks/Application2_TaiwanCreditDefault.ipynb)
   
### Benchmarking your own or another  algorithm
Please follow the instructions [here](https://github.com/LeoGrin/tabular-benchmark/tree/main) to benchmark a new model depending on your budget.
### Usage of TabSRALinear
We use the [skorch](https://skorch.readthedocs.io/en/stable/) framework to make our implementation more scikit-learn friendly.
[Here](https://github.com/anselmeamekoe/TabSRA/tree/main/ESANN_XKDD) is the old version.

```python
import torch
import torch.nn as nn
from skorch.callbacks import EarlyStopping,LRScheduler,Checkpoint, TrainEndCheckpoint, EpochScoring, InputShapeSetterTabSRA
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from sramodels.SRAModels import TabSRALinearClassifier
from sklearn.metrics import roc_auc_score

configs = {
         "module__n_head":1,
         "module__dim_head":8,
         "module__n_hidden_encoder":1,
         "module__dropout_rate":0.3,
         "optimizer__lr":0.001,
         "random_state":42,
         "criterion": nn.BCEWithLogitsLoss,
         "max_epochs":100,
         "batch_size":256,
         "device":'cpu'
}
scoring = EpochScoring(scoring='roc_auc',lower_is_better=False)#the scoring function
setter = InputShapeSetterTabSRA(regression=False)#used for setting the input and output dimension automatically
early_stop = EarlyStopping(monitor=scoring.scoring, patience=10,load_best=True,lower_is_better=False, threshold=0.0001,threshold_mode='abs')
callbacks = [scoring, setter, early_stop, lr_scheduler]

valid_dataset = Dataset(X_val.values.astype(np.float32),Y_val.astype(np.float32))# custom validation dataset
TabClassifier = TabSRALinearClassifier(**configs,train_split = predefined_split(valid_dataset),callbacks = callbacks)
_ = TabClassifier.fit(X_train_.values.astype(np.float32),Y_train_.astype(np.float32))

# prediction
Y_val_pred = TabClassifier.predict_proba(X_val.values.astype(np.float32))
best_aucroc = roc_auc_score(Y_val.astype(np.float32), Y_val_pred[:,1])

# feature attribution
attributions_val = TabClassifier.get_feature_attribution(X_val.values.astype(np.float32))

# attention weights
attentions_val = TabClassifier.get_attention(X_val.values.astype(np.float32))

```
Key parameters
the model parameters are preceded by ```module```.
 - ```module_n_head```: int (default=2)
   Number of SRA head/ensemble. Bigger values gives capacity to the model to produce less stable/robust explanations.
   Typical values are 1 or 2.
  
 - ```module__dim_head```: int (default=8)
   The attention head dimension , $d_k$ in the paper.
   Typical values are {4,8,12}.
   
  - ```module__n_hidden_encoder```: int (default=1)
   The number of hidden layers in  in the Key/Query encoder.
   Typical values are {1,2}.

 - ```module__dropout_rate```: float (default=0.0) 
   The neuron dropout rate used  in the Key/Query encorder during the training.
   
 - ```module__classifier_bias```: bool (default=True)
   Whether to use bias term in the downstream linear classifier.
   
 - ```optimizer```: (default=torch.optim.Adam)
   
 - ```optimizer__lr```: float (default=0.05)
   learning rate used for the training.
   
 - ```max_epochs```: int (default=100)
   Maximal number of training iterations.
  
 - ```batch_size```: int (default=256)

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
