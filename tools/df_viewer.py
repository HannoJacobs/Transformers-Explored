"""view df"""

import pandas as pd
from display_df import display_df

df = pd.read_csv("Datasets/eng_afr/eng_afr_parallel_1000_rows.csv")
display_df(df, "df")
