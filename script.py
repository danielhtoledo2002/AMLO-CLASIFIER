import pandas as pd
df = pd.read_csv("concat_2023.csv")
df = df[df["Participante"] == "PRESIDENTE ANDRES MANUEL LOPEZ OBRADOR"]
print(df.head())
df.to_csv("amlo.csv", index=False)
