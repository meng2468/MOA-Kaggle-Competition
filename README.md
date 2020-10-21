# MOA Exploration + Resources
## Problem Statement
**Predict**: Drug sample and dose as one or more MoA classes
...
**Given**: Gene Expression & Cell Viability

**Evaluation**: Average value of log-loss for each drug-MoA pair
## Problem Resources
- https://www.kaggle.com/c/lish-moa/discussion/184005 (General Insights)
- https://www.kaggle.com/c/lish-moa/discussion/181040 (Ideas for Exploration)
- https://www.kaggle.com/gunesevitan/mechanisms-of-action-moa-prediction-eda (Completed EDA)

## Papers
- https://arxiv.org/pdf/1908.07442.pdf (TabNet)
- https://arxiv.org/abs/1906.02629 (Label Smoothing)

## General Resources
- https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python (EDA Basics on Housing Price Comp.)
- https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python (Feature Cleaning ref. Titanic)
- https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969 (Pseudo-labeling)

## Baselines
- https://www.kaggle.com/optimo/tabnetregressor-2-0-train-infer (TabNetRegressor 0.01864)
- https://www.kaggle.com/nicohrubec/pytorch-multilabel-neural-network (Multi-label NN 0.01875)
- https://www.kaggle.com/kailex/moa-transfer-recipe-with-smoothing (NN + Transfer Learning 0.01858)
- https://www.kaggle.com/ksouriazer/the-power-of-blend (Blending: 2 NNs + XGBoost 0.01866)

# Features: 
**sig_id** is the unique sample id

### g- prefix
772 **Gene Expression** features (from g-0 to g-771)

https://www.yourgenome.org/facts/what-is-gene-expression

Gene expression is the process by which the instructions in our DNA are converted into a functional product, such as a protein.

### c- prefix
100 **Cell Viability** features (from c-0 to c-99)

https://www.cellsignal.de/contents/_/synopsis-of-cell-proliferation-metabolic-status-and-cell-death/cell-viability-and-survival

Cell viability is a measure of the proportion of live, healthy cells within a population. Cell viability assays are used to determine the overall health of cells, optimize culture or experimental conditions, and to measure cell survival following treatment with compounds, such as during a drug screen.

### cp_type
Binary categorical feature which indicates the samples are treated with a compound or with a control perturbation (trt_cp or ctl_vehicle)

https://www.kaggle.com/c/lish-moa/discussion/180304 (Discussion on wtf it means)

"Yes, the controls have no MoA in both testing and training. So, setting the value to zero is what you want to do. However, in real applications, the null controls are one way to assess the efficacy of a drug and can be useful in modeling the other MoAs.

At the same time, I see that our decision of keeping the controls in both datasets may be confusing. Here he idea was essentially to avoid splitting the dataset into too many files." - Host

**cp_time** is a categorical feature which indicates the treatment duration (24, 48 or 72 hours)

**cp_dose** is a binary categorical feature which indicates the dose is low or high (D1 or D2)
