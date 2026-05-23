import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import os
import gc
import json
import concurrent.futures

# --------------------
# 1. CONFIGURAZIONE
# --------------------

# percorso del file di input (contenente gli embeddings generati nello step 2)
INPUT_FILE = "../2_embeddings_gen/results/Minerva-1B-ALL-EMBEDDINGS.parquet"

# cartella di output dove salvo i 53.000 file singoli
OUTPUT_FOLDER = "results/similarity_parts"

# percorso del file .json in cui salvare la mappa indici-latents (e viceversa)
INDEX_MAPPING_FILE = "results/latent_index_mapping.json"

# definisco una misura per il batch per il calcolo matematico (non per il salvataggio, che avverrà per singoli file)
BATCH_SIZE = 1000

def create_matrix_and_mapping():
    print("--- Inizio procedura generazione file di similarità singoli ---")

    # ---------------------------------------
    # 2. CARICAMENTO E PREPARAZIONE MATRICE
    # ---------------------------------------

    # se non esiste il file di input segnalo un errore e interrompo la funzione
    if not os.path.exists(INPUT_FILE):
        print(f"Errore: File {INPUT_FILE} non trovato. Esegui prima lo script 2b_embeddings_gen.py")
        return

    print("Caricamento embeddings in memoria...")
    # salvo il file .parquet in un df pandas. Uso come engine pyarrow
    df_emb = pd.read_parquet(INPUT_FILE, engine="pyarrow")

    # estraggo la lista ordinata di tutti i latent id e calcolo il numero totale di latent
    latent_ids = df_emb["Latent ID"].tolist()
    total_latents = len(latent_ids)

    # .values() restituisce un array 1D i cui valori sono gli embeddings. Quindi, ho un array (contenitore) di array (embeddings)
    # con .stack() creo la matrice numpy 2D (n_latents x embedding_length) per i calcoli
    embeddings_matrix = np.stack(df_emb["Embedding"].values)

    # avendo estratto le colonne di mio interesse, forzo la pulizia dei dati richiamando il garbage collector
    del df_emb
    gc.collect()

    # normalizzazione per velocizzare il calcolo della cosine similarity dopo (diventa un semplice prodotto scalare)
    print("Normalizzazione vettori (L2 norm)...")
    # axis=1 specifica su una intera singola riga
    # la norma L2 rende tutti i vettori dei versori (lunghezza=1) dividendo ogni componente del vettore per la lunghezza totale del vettore stesso
    embeddings_matrix = normalize(embeddings_matrix, axis=1, norm="l2")

    # -------------------------------
    # 3. MAPPATURA INDICI-LATENT
    #  -------------------------------

    # creo un dizionario che associa ogni latent ID alla sua posizione, con enumerate() che crea tuple (indice, valore)
    # ad esempio: {"layers.14_latent18": 0, "layers.14_latent24": 1, ...}
    id_to_index = {latent_id: idx for idx, latent_id in enumerate(latent_ids)}

    # creo anche il percorso inverso
    index_to_id = {idx: latent_id for idx, latent_id in enumerate(latent_ids)}

    # mi garantisco che le cartelle indicate per salvare gli output esistano. Se non esistono, le creo ora
    os.makedirs(OUTPUT_FOLDER, exist_ok=True) # nome della cartella
    os.makedirs(os.path.dirname(INDEX_MAPPING_FILE), exist_ok=True) # nome del percorso del file, uso dirname() per trovare il nome della cartella

    # salvo il file .json (composto dalle due mappature inverse) nella cartella indicata
    with open(INDEX_MAPPING_FILE, "w") as f:
        json.dump({"id_to_index": id_to_index, "index_to_id": index_to_id}, f)
    print(f"Mappatura indici salvata in: {INDEX_MAPPING_FILE}")

    # ---------------------------------------------------
    # 4. CALCOLO SIMILARITA' E SALVATAGGIO SINGOLI FILE
    # ---------------------------------------------------

    # creo un set dei file già generati (estraggo tutti gli elementi della cartella indicata per l'output) per permettere la ripresa in caso di crash
    # (set anziché list in quanto la ricerca "in" è molto più efficiente - set usa delle tabelle hash dietro le quinte)
    file_esistenti = set(os.listdir(OUTPUT_FOLDER))

    print("Inizio calcolo e salvataggio in file singoli...")

    # itero su dei blocchi di latent solo per migliore gestione delle risorse (salverò singoli file)
    for i in range(0, total_latents, BATCH_SIZE):
        # selezione l'ìndice finale effettivo (all'ultimo blocco c'è da considerare la lunghezza totale dei latents)
        end_i = min(i + BATCH_SIZE, total_latents)

        # seleziono solo i latent che ci sono in questo batch
        latents_nel_batch = latent_ids[i:end_i]
        # estraggo dal batch solo gli ID dei latent che non sono ancora stati salvati come file
        file_da_calcolare = [l for l in latents_nel_batch if f"{l}.parquet" not in file_esistenti]

        # se tutti i file di questo blocco esistono già, skippo il calcolo matematico e passo al batch successivo
        if not file_da_calcolare:
            print(f"Blocco {i}->{end_i} già completato. Si passa al prossimo...")
            continue

        print(f"Elaborazione calcolo blocco {i} -> {end_i}...")

        # seleziono solo gli embeddings del batch che sto analizzando
        batch_embeddings = embeddings_matrix[i:end_i]
        # calcolo la similarità di questi 1000 latent con ogni altro latent (53.211) come prodotto scalare (merito di l2 norm fatta prima)
        # uso la matrice trasposta (.T) per rispettare la regola del prodotto tra matrici (NxM . MxQ). Il risultato è una sottomatrice di dimensioni (BATCH_SIZE, total_latents)
        similarity_batch = np.dot(batch_embeddings, embeddings_matrix.T)

        # definisco una piccola funzione interna che gestisce il salvataggio sul disco del file di un singolo latent per abilitare multithreading
        def save_single_file(local_idx):
            # calcolo l'indice assoluto del latent rispetto all'intero dataset
            global_idx = i + local_idx
            # estraggo il nome identificativo e genero il nome del file finale
            latent_name = latent_ids[global_idx]
            file_name = f"{latent_name}.parquet"

            # se il file esiste già (perché magari il calcolo si era interrotto a metà batch), lo salto
            if file_name in file_esistenti:
                return None # ritorno None per segnalare che non ho fatto nulla

            # estraggo la singola riga di similarità (un array 1D di 53.000 float)
            sim_array = similarity_batch[local_idx]

            # qui c'è un trucco per la massima efficienza.
            # creo un DataFrame con una sola colonna ("Similarity") e 53.000 righe.
            # così è molto più efficiente da salvare e leggere rispetto a 1 riga e 53.000 colonne (parquet è un formato colonnare)
            # l'ordine delle righe corrisponde esattamente all'ordine del JSON INDEX MAPPING calcolato
            df_single_latent = pd.DataFrame({
                "Similarity": sim_array.astype(np.float32)
            })

            # creo il percorso per il salvataggio e creo il file del singolo latent
            file_path = os.path.join(OUTPUT_FOLDER, file_name)
            df_single_latent.to_parquet(file_path, engine="pyarrow", index=False)

            # restituisco il nome del file creato per aggiornare il set dei file esistenti
            return file_name

        # uso ThreadPoolExecutor per parallelizzare le chiamate a save_single_file
        # max_workers=4 dice a Python quanti thread lanciare in parallelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # sottomettiamo tutti i task (gli indici da 0 a BATCH_SIZE-1)
            futures = [executor.submit(save_single_file, idx) for idx in range(similarity_batch.shape[0])]

            # as_completed ci permette di raccogliere i risultati man mano che finiscono
            for future in concurrent.futures.as_completed(futures):
                saved_file_name = future.result()
                if saved_file_name: # Se non è None
                    # aggiungo al set in memoria
                    file_esistenti.add(saved_file_name)

        # pulizia RAM a fine batch
        del batch_embeddings
        del similarity_batch
        gc.collect()

    print("\n --- Generazione di tutti i file completata! ---")

if __name__ == "__main__":
    create_matrix_and_mapping()