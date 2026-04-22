import pandas as pd
import numpy as np

# Carregar dataset original sem modificá-lo
PATH = "../carne-brasileira-exportada.csv"
df = pd.read_csv(PATH)

df["date"] = pd.to_datetime(df["ano"].astype(str) + "Q" + df["trimestre"].astype(str))
df = df.sort_values(["tipo_carne", "date"]).reset_index(drop=True)

# Criar cópia com features derivadas

def add_derived_features(group):
    group = group.copy()
    group = group.sort_values("date")
    group["time_index"] = np.arange(len(group))
    group["quarter_sin"] = np.sin(2 * np.pi * (group["trimestre"] - 1) / 4)
    group["quarter_cos"] = np.cos(2 * np.pi * (group["trimestre"] - 1) / 4)
    group["halfyear_sin"] = np.sin(2 * np.pi * (group["trimestre"] - 1) / 2)
    group["halfyear_cos"] = np.cos(2 * np.pi * (group["trimestre"] - 1) / 2)

    for col in ["exportacao_usd", "preco_medio_ton_usd"]:
        for lag in [1, 2, 3, 4]:
            group[f"{col}_lag{lag}"] = group[col].shift(lag)

        group[f"{col}_rolling_mean_4"] = group[col].shift(1).rolling(window=4, min_periods=1).mean()
        group[f"{col}_ewm_4"] = group[col].shift(1).ewm(span=4, adjust=False).mean()
        group[f"{col}_diff_1"] = group[col].diff(1)
        group[f"{col}_pct_change_a"] = group[col].pct_change(periods=4)

    return group

# Aplicar por tipo de carne para preservar dependências temporais dentro de cada série

df_derived = df.groupby("tipo_carne", group_keys=False).apply(add_derived_features).reset_index(drop=True)

# Salvar a cópia derivada
OUTPUT_PATH = "../carne-brasileira-derivada.csv"
df_derived.to_csv(OUTPUT_PATH, index=False)

# Visualizar as primeiras linhas
print(f"Derived dataset saved to: {OUTPUT_PATH}")
df_derived.head()
