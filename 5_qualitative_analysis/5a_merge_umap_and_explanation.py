import pandas as pd

# ----------------------------
# 1. CARICAMENTO DEI DATASET
# ----------------------------

# salvo il dataset UMAP in un df
df_umap = pd.read_csv("../4_quantitative_analysis/dataset/dataset_umap.csv")

# salvo il dataset con le spiegazioni in un df (lo userò come "base")
df_base = pd.read_csv("../2_embeddings_gen/dataset/Minerva-1B-ALL-EXPLANATIONS-to-embedding.csv")

# ----------------------------
# 2. MERGE DEI DUE DATAFRAME
# ----------------------------

# effettuo un left join: mantengo tutte le righe di df_base e aggiungo le colonne di df_umap dove il Latent ID coincide
df_finale = pd.merge(df_base, df_umap, on="Latent ID", how="left")

# stabilisco un ordine delle colonne
colonne_ordinate = ["Latent ID", "explanation", "UMAP 1", "UMAP 2"]
df_finale = df_finale[colonne_ordinate]

# ----------------------------
# 3. SALVATAGGIO DEL FILE
# ----------------------------

# salvo il df in formato .csv
df_finale.to_csv("dataset/dataset_explanation_and_umap.csv", index=False)

# stampo feedback che conferma l'avvenuta creazione del dataset
print(f"File creato con successo! Dimensioni del nuovo dataset: {df_finale.shape}")