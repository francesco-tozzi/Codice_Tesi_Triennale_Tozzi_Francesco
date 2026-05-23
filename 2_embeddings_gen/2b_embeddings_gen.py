import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import pyarrow

# -----------------------
# 1. CONFIGURAZIONE
# -----------------------

# indico il file di input (quello pulito generato nello step precedente, output del codice 2a)
INPUT_FILE = "dataset/Minerva-1B-ALL-EXPLANATIONS-to-embedding.csv"

# indico il file di output (dove salverò gli embeddings in formato .parquet)
OUTPUT_FILE = "results/Minerva-1B-ALL-EMBEDDINGS.parquet"

# indico il modello per la generazione degli embeddings (scelto mediante benchmark, step 1)
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# imposto la grandezza del chunk da analizzare
CHUNK_SIZE = 10000

# ----------------------------
# 2. GENERAZIONE EMBEDDINGS
# ----------------------------
# uso una funzione piuttosto che codice inline per lavorare con variabili locali e non globali

# definisco la funzione per la generazione di embeddings che contiene un ciclo per processare chunk del dataset originale
def generate_all_embeddings():
    # carico il dataset
    print("Caricamento dataset in memoria...")
    try:
        # salvo il .csv in un dataframe pandas
        df_input = pd.read_csv(INPUT_FILE)
    # gestisco l'eventuale errore di file non trovato
    except FileNotFoundError:
        print(f"Errore: File '{INPUT_FILE}' non trovato. Esegui prima lo script di pulizia 2a_dataset_filter.py")
        return

    # caricamento del modello
    print(f"Caricamento modello {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # righe totali del df
    total_rows = len(df_input)

    # eseguo i controlli per capire dove siamo arrivati
    # se esiste già il file di output
    if os.path.exists(OUTPUT_FILE):
        # leggo il file .parquet già esistente come df pandas, considerando solo una colonna per risparmiare memoria
        df_esistente = pd.read_parquet(OUTPUT_FILE, columns=["Latent ID"])
        start_idx = len(df_esistente) # riga inclusa nel nuovo ciclo di analisi
        print(f"Trovato file esistente. Riprendo dall'indice: {start_idx}")
    # mentre se il file non esiste già
    else:
        start_idx = 0
        print("Nessun file precedente trovato. Inizio dall'indice 0.")

    # ciclo per iterare su tutto il dataset ma a chunk
    for current_start in range(start_idx, total_rows, CHUNK_SIZE):
        # calcolo la fine del chunk corrente. Se è maggiore delle righe effettive, lo adatto all'ultima riga disponibile
        current_end = min(current_start + CHUNK_SIZE, total_rows)
        print(f"\n--- Elaborazione range: {current_start} a {current_end} ---")

        # seleziono le righe del range corrente, creo una copia di questa sezione del dataframe originale
        df_chunk = df_input.iloc[current_start:current_end].copy()

        # estraggo la lista delle spiegazioni
        sentences = df_chunk["explanation"].tolist()
        # encode genera gli embeddings
        # batch_size=32: processa 32 frasi alla volta per non saturare la RAM
        # show_progress_bar=True: mostra una barra di caricamento durante l'esecuzione
        embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True)

        # DataFrame risultati di questo chunk: Latent ID ed Embedding
        # converto gli embeddings (che sono numpy arrays) in liste per essere sicuro della compatibilità
        df_results = pd.DataFrame({
            "Latent ID": df_chunk["Latent ID"].values,
            "Embedding": list(embeddings)
        })

        # salvataggio nel file di output
        print(f"Salvataggio in {OUTPUT_FILE}...")

        if os.path.exists(OUTPUT_FILE):
            # uso pyarrow come engine per l'aggiornamento del file .parquet
            existing_df = pd.read_parquet(OUTPUT_FILE, engine="pyarrow")
            updated_df = pd.concat([existing_df, df_results], ignore_index=True)
            updated_df.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)
        else:
            # uso pyarrow come engine per la creazione del file .parquet
            df_results.to_parquet(OUTPUT_FILE, engine="pyarrow", index=False)

        print(f"Chunk salvato. Progresso: {current_end}/{total_rows}")

    # stampo feedback finale di elaborazione completata
    print(f"\n--- Elaborazione totale completata. File salvato in {OUTPUT_FILE} ---")

# esecuzione della funzione
if __name__ == "__main__":
    generate_all_embeddings()