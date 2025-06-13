"""view df"""

import pandas as pd
from display_df import display_df

df = pd.read_csv("Datasets/synth_i5_r0-9_n-1000.csv")
display_df(df, "df")
