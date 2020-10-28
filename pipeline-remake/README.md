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
3. Validation
    - k-fold CV
    - multi-seed with k-fold CV
4. Experimentation
    - Run a combinationg of:
        - Feature Engineering
        - Architecture
        - Validation 
    - Perform Hyperparameter Tuning
    - Save Results
Submissions
- Copy feature eng. function
- Copy model architecture

