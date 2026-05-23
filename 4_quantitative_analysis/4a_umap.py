import pandas as pd
import umap
import torch
import numpy as np
from safetensors.torch import load_file
import sys
import os
import json

# --------------------------------------------
# 1. CONFIGURAZIONE E CARICAMENTO PESI SAE
# ---------------------------------------------

# percorso del file dei pesi SAE

# ISTRUZIONI PER L'UTENTE GITHUB:
# 1. Scarica il modello SAE da: https://huggingface.co/alessandrobondielli/sae-Minerva-1B-32x
# 2. Inserisci qui sotto il TUO percorso locale assoluto fino alla cartella 'layers.14'
# Esempio: TENSOFILE_PATH = "/home/user/.cache/huggingface/hub/..."
TENSOFILE_PATH = ""

# controllo di sicurezza per gli utenti github
if not TENSOFILE_PATH or not os.path.exists(TENSOFILE_PATH):
    print("\n" + "="*70)
    print("ERRORE: PERCORSO MODELLO SAE NON CONFIGURATO O NON VALIDO")
    print("="*70)
    print("Per eseguire questo script è necessario possedere i pesi del modello SAE.")
    print("1. Scarica il modello da: https://huggingface.co/alessandrobondielli/sae-Minerva-1B-32x")
    print("2. Apri questo script e modifica la variabile 'TENSOFILE_PATH' inserendo")
    print("   il percorso assoluto del file scaricato sul tuo computer.")
    print(f"Percorso attualmente letto: '{TENSOFILE_PATH}'")
    print("="*70 + "\n")
    sys.exit(1) # il codice 1 indica al sistema operativo un'uscita per errore

try:
    # tento di caricare il file dei pesi SAE dal percorso specificato
    state_dict = load_file(f"{TENSOFILE_PATH}/sae.safetensors")
    # estraggo la matrice dei pesi dell'encoder: [n_latents, d_model] -> [65536, 2048]
    W_enc = state_dict["encoder.weight"]
    # stampo la dimensione finale della matrice dei latenti caricati per verifica
    print(f"Caricati {W_enc.shape[0]} latenti (vettori di dimensione {W_enc.shape[1]}).")
except FileNotFoundError:
    # catturo l'errore se il file non esiste nel percorso specificato
    print(f"ERRORE: File non trovato. Controlla il percorso: {TENSOFILE_PATH}")
    sys.exit(1) # termino lo script con codice di errore 1


# -------------------------------------------
# 2. APPLICAZIONE UMAP E SALVATAGGIO IN DF
# -------------------------------------------

# sposto il tensore PyTorch dalla GPU (se c'è) alla CPU e lo converto in array NumPy (formato richiesto da UMAP)
latent_vectors_matrix = W_enc.cpu().numpy()

# stampo feedback di inizio processo UMAP
print(f"\nInizio la riduzione di dimensionalità con UMAP su {W_enc.shape[0]} latenti...")
reducer = umap.UMAP(
    n_components=2, # riduco i vettori da 2048 dimensioni a 2 dimensioni
    n_neighbors=30, # quanti vicini considerare per modellare la struttura locale/globale
    min_dist=0.1, # forzo i punti a non essere troppo compressi (mantenendo la separazione tra i cluster)
    metric="cosine", # specifico la metrica usata per calcolare le distanze nello spazio 2048D (di default è 'euclidean')
    n_jobs=-1 # abiito il parallelismo su tutti i core della CPU per un calcolo veloce
)
# eseguo l'algoritmo UMAP e salvo le nuove coordinate 2D
embedding = reducer.fit_transform(latent_vectors_matrix)
print("Riduzione completata.")

# creo un DataFrame pandas con le coordinate 2D risultanti
umap_df = pd.DataFrame(embedding, columns=["UMAP 1", "UMAP 2"])
# aggiungo l'ID unico (es. layers.14_latent0) per ogni riga (che contiene le due coordinate di un punto)
umap_df["Latent ID"] = [f"layers.14_latent{i}" for i in range(len(umap_df))]
# (posso farlo perché la matrice era ordinata da latent 0 a n e UMAP è strettamente order-preserving)

# -------------------------------------------------------
# 3. FILTRAGGIO DEL CLUSTER ISOLATO E SALVATAGGIO RUMORE
# -------------------------------------------------------
# a seguito di una prima analisi dei risultati, mi sono reso conto della presenza di un cluster di rumore.
# implemento ora questo filtro aggiuntivo prima di salvare le coordinate UMAP eliminando quel cluster (di cui ho individuato visivamente le coordinate)

print(f"Dimensioni del df prima del filtro: {umap_df.shape[0]} righe.")

# 1. creo una maschera booleana (una serie di True/False).
# è True SOLO per le righe che si trovano nella zona che voglio eliminare
cluster_da_eliminare = (umap_df["UMAP 1"] < 2.5) & (umap_df["UMAP 2"] < 7)

# 2. estrazione dei latent che fanno rumore:
# uso la maschera per selezionare le righe del cluster di rumore, estraggo la colonna "Latent ID" e converto la series pandas in una lista python standard
latent_eliminati = umap_df[cluster_da_eliminare]["Latent ID"].tolist()

# 3. inverto la condizione con l'operatore '~' (NOT) in modo che i punti da tenere diventino True, mentre quelli da eliminare False
# Pandas filtra il dataframe mantenendo solo le righe in cui la maschera finale è True
umap_df = umap_df[~cluster_da_eliminare]

print(f"Dimensioni del DataFrame dopo il filtro: {umap_df.shape[0]} righe.")

filter_data_dir = "../2_embeddings_gen/dataset"
os.makedirs(filter_data_dir, exist_ok=True) # mi assicuro che la cartella esista

# creo un percorso completo per il file
json_output_path = os.path.join(filter_data_dir, "latent_removed.json")

# creo un dizionario con metadati e latent eliminati
data_to_save = {
    "descrizione": "Latent scartati perché appartenenti al cluster di rumore UMAP1 < 2.5 e UMAP2 < 7",
    "numero_elementi": len(latent_eliminati),
    "latent_ids": latent_eliminati # la lista dei latent eliminati
}

# salvo il file in formato JSON
with open(json_output_path, "w", encoding="utf-8") as f:
    # indent=4 rende il file leggibile anche per noi umani
    json.dump(data_to_save, f, indent=4)

print(f"Elenco latent eliminati salvato in: {json_output_path}")

# ----------------------------------------------
# 4. ESPORTAZIONE DATI PER ANALISI SUCCESSIVE
# ----------------------------------------------
# definisco il nome del file CSV in cui salvare i dati
csv_filename = "dataset/dataset_umap.csv"

# salvo il dataframe 'umap_df' in formato CSV; 'index=False' evita di salvare la colonna dei numeri di riga
umap_df.to_csv(csv_filename, index=False)

# stampo un messaggio per confermare l'avvenuto salvataggio
print(f"Dati UMAP esportati con successo in: {csv_filename}")
print(f"   Il file contiene {len(umap_df)} righe e le colonne: {list(umap_df.columns)}")