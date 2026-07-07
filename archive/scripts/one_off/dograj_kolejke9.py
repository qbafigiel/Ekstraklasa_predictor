import pandas as pd

df = pd.read_csv("data/mecze_2025_26.csv")

# Znajdź brakujący wiersz
idx = df[df["match_id"] == 2424].index[0]
print(f"Uzupełniam mecz 2424: {df.loc[idx, 'gospodarz']} vs {df.loc[idx, 'gosc']}")

df.loc[idx, "posiadanie_gosp"] = 41
df.loc[idx, "posiadanie_gosc"] = 59
df.loc[idx, "strzaly_gosp"] = 17
df.loc[idx, "strzaly_gosc"] = 13
df.loc[idx, "celne_gosp"] = 4
df.loc[idx, "celne_gosc"] = 9
df.loc[idx, "strzaly_zablokowane_gosp"] = 7
df.loc[idx, "strzaly_zablokowane_gosc"] = 0
df.loc[idx, "strzaly_niecelne_gosp"] = 6
df.loc[idx, "strzaly_niecelne_gosc"] = 4
df.loc[idx, "rozne_gosp"] = 5
df.loc[idx, "rozne_gosc"] = 5
df.loc[idx, "faule_gosp"] = 13
df.loc[idx, "faule_gosc"] = 8
df.loc[idx, "spalone_gosp"] = 3
df.loc[idx, "spalone_gosc"] = 3
df.loc[idx, "zk_gosp"] = 0
df.loc[idx, "zk_gosc"] = 2
df.loc[idx, "czk_gosp"] = 0
df.loc[idx, "czk_gosc"] = 0
df.loc[idx, "druga_zk_gosp"] = 0
df.loc[idx, "druga_zk_gosc"] = 0
df.loc[idx, "dosrodkowania_gosp"] = 20
df.loc[idx, "dosrodkowania_gosc"] = 17
df.loc[idx, "dosrodkowania_celne_gosp"] = 7
df.loc[idx, "dosrodkowania_celne_gosc"] = 7
df.loc[idx, "odbiory_gosp"] = 6
df.loc[idx, "odbiory_gosc"] = 6
df.loc[idx, "podania_gosp"] = 292
df.loc[idx, "podania_gosc"] = 443
df.loc[idx, "podania_celne_gosp"] = 228
df.loc[idx, "podania_celne_gosc"] = 384

df.to_csv("data/mecze_2025_26.csv", index=False, encoding="utf-8-sig")
print("Zapisano. Brakujące wartości:")
print(df.isnull().sum()[df.isnull().sum() > 0])