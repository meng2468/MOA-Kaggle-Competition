import processing.preprocessing as pp
import processing.crossvalidation as cv

df_train_x, df_train_y = pp.get_training_data()
cv.df_label_folds(df_train_x, df_train_y, 2)