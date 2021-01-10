# MoA Competition Folder

## Final Submission Notes
### Blend of three m. four models
- TabNet: https://www.kaggle.com/malteadrianmeng/tabnet
- Deep NN w. Transfer Learning: https://www.kaggle.com/thehemen/pytorch-transfer-learning-with-k-folds-by-drug-ids
- Reg NN w. Clustering and Meta-Features: https://www.kaggle.com/kushal1506/moa-pretrained-non-scored-targets-as-meta-features
- Shallow NN w. 2 do layers: https://www.kaggle.com/malteadrianmeng/pca-var-cv-simple-nn

### Tweaking
- Test Swish or Mish activation function
- Final layer bias
- Feature selection
- Removal of clustering and meta-features
- TabNet can still be optimised to .01834!!! https://www.kaggle.com/c/lish-moa/discussion/198850

### Implementation
- Re-write all final models
- Add weight loading so final submission can just be plugged
