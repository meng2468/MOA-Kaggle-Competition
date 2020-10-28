import pandas as pd
import numpy as np

def misc_engineering(df_x):
    return df_x, df_y

def get_eng_df(df_x):
    df_x = misc_engineering(df_x)

    return df_x

def to_csv(df_x):
    df_x = misc_engineering(df_x)

    df_x.to_csv('featire_eng_temp.csv', index=False)

