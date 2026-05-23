import pandas as pd
import os
import json

# definisco il percorso del file di input e quello di output
input_file = "dataset/Minerva-1B-ALL-EXPLANATIONS-CLEANED.csv"
output_file = "dataset/Minerva-1B-ALL-EXPLANATIONS-to-embedding.csv"

# percorso del file .json contenente la lista dei latent di rumore da ignorare
FILTER_FILE = "dataset/latent_removed.json"

# PULIZIA DEL DATASET
# in try per interrompere l'esecuzione in caso di errore
try:
    # carico il dataset originale e lo salvo come df pandas
    df = pd.read_csv(input_file)
    # stampo un feedback per assicurarmi che sia stato letto e sapere quante righe ci sono prima della pulizia
    print(f"Dataset caricato. Righe iniziali: {len(df)}")

    # STEP 1: devo assicurarmi che le spiegazioni non siano vuote o composte solo da spazi
    df_clean = df.dropna(subset=["explanation"])
    df_clean = df_clean[df_clean["explanation"].str.strip() != ""]

    # STEP 2: filtraggio dell'isola di rumore (latent rumorosi individuati con una analisi manuale - vedi 4a_umap)
    if os.path.exists(FILTER_FILE):
        print("Caricamento lista dei latent di rumore...")
        # apro il file .json
        with open(FILTER_FILE, "r", encoding="utf-8") as f:
            scarti_data = json.load(f)

        # converto la lista in un set per la massima efficienza
        set_scarti = set(scarti_data["latent_ids"])
        print(f"Individuati i latent di rumore. Rimozione in corso...")

        # mantengo in df_clean solo le righe il cui "Latent ID" NON si trova nel set degli scarti
        df_clean = df_clean[~df_clean["Latent ID"].isin(set_scarti)]
        print(f"Rimozione rumore completata.")
    else:
        print(f"Avviso: File {FILTER_FILE} non trovato. Nessun filtro applicato.")

    # stampo un feedback per avere contezza della pulizia fatta
    print(f"Righe valide dopo il filtraggio: {len(df_clean)}")

    # salvo il nuovo file al percorso indicato e stampo un feedback di conferma
    df_clean.to_csv(output_file, index=False)
    print(f"File pulito salvato con successo in: {output_file}")

# GESTIONE DEGLI ERRORI
# se l'errore è che il file non è stato trovato, stampo un messaggio specifico per questo errore
except FileNotFoundError:
    print(f"Errore: Il file '{input_file}' non è stato trovato.")

# per qualsiasi altro errore scrivo un messaggio generico di errore, riportando l'errore
except Exception as e:
    print(f"Si è verificato un errore imprevisto: {e}")

# OUTPUT
# Dataset caricato. Righe iniziali: 53540
# Caricamento lista dei latent di rumore...
# Individuati i latent di rumore. Rimozione in corso...
# Rimozione rumore completata.
# Righe valide dopo il filtraggio: 53119
# File pulito salvato con successo in: dataset/Minerva-1B-ALL-EXPLANATIONS-to-embedding.csv