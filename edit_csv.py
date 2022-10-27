import pandas as pd
from utils import PATH_MODIF_CSV, PATH_ORIGINAL_CSV

# Keep only the composers with more than 20 different pieces in the dataset


df = pd.read_csv(PATH_ORIGINAL_CSV)


df_unique = df.drop_duplicates(
    subset=["canonical_composer", "canonical_title"], keep=False
)

dic = dict(df_unique["canonical_composer"].value_counts())

dic2 = {}
L = []
for i in dic:
    if dic[i] > 20:
        dic2[i] = dic[i]
        L.append(i)
df_reduced = df_unique[df_unique["canonical_composer"].isin(L)]


df_reduced.to_csv(PATH_MODIF_CSV)
