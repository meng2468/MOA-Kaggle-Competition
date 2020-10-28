# MoA Experimentation Pipeline v2
## Reasoning
- Current code structure is getting worse with additions
- Need normalised naming and modularisation conventions
- Still don't have
    - Logging Conventions
    - Experiment Reproducibility
    - Automatic Hyperparameter Tuning
    - Experiment -> Submission Guideline

## Plan
Seperate the three experimental processes
1. Data processing
    - Feature Augmentation
    - Feature Engineering
    - CSV Saving for Experiment
2. Model Engineering
    - NN Architecture
4. Experimentation
    - Run a combinationg of:
        - Feature Engineering
        - Architecture
        - Hyperparameters 
    - Save Results
Don't know where to put
- Submission Template
- Hyperparameter tuning
- Crossvalidation

Submissions
- Copy feature eng. function
- Copy model architecture