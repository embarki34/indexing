import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def remove_empty_rows(df):
    """Remove all empty rows from the DataFrame."""
    return df.dropna(how='all')


df = pd.read_csv('train.csv')
df = remove_empty_rows(df)
df.to_csv('train2.csv', index=False)





