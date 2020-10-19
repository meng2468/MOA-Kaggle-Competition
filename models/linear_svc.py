import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

df_train_x = pd.read_csv('../processed-input/prep_train_features.csv')
df_train_y = pd.read_csv('../processed-input/train_target_p.csv')

clf = MultiOutputClassifier(LogisticRegression(n_jobs=-1, verbose=1)).fit(df_train_x,df_train_y)

pred = clf.predict(df_train_x[3000:])
df_pred = pd.DataFrame(pred)
fpr, tpr, thresholds = metrics.roc_curve(df_train_y[3000:], df_pred, pos_label=2)
print(metrics.auc(fpr, tpr))