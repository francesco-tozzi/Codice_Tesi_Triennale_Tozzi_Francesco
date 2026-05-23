import torch
import pandas as pd
import numpy as np
import os
import sys
import json
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from safetensors.torch import load_file

# ---------------------------------------------------------------
# 3. FUNZIONE PER L'ESTRAZIONE GLOBALE (eseguita una sola volta)
# ---------------------------------------------------------------
# con questa funzione calcolo i k vicini spaziali (spazio SAE) ed estraggo le similarità di essi (calcolate nello spazio embedding)
def precompute_global_data(W_enc_full, mapping, MAX_K, BATCH_SIZE, SIMILARITY_DIR, KNN_OUTPUT_FILE):
    print(f"\n--- Calcolo dei {MAX_K} vicini nello spazio SAE (2048D) ed estraggo le similarità di essi (calcolate nello spazio embedding) ---")

    # applico la normalizzazione L2 e mi assicuro che il risultato sia su cpu (RAM)
    W_enc_norm = torch.nn.functional.normalize(W_enc_full, p=2, dim=1).cpu()

    # salvo le due mappature
    id_to_index = mapping["id_to_index"]
    index_to_id = mapping["index_to_id"]

    # mi assicuro che le chiavi siano interi per poter fare ricerche
    index_to_id = {int(k): v for k, v in index_to_id.items()}

    # ricavo la lista ordinata di tutti e soli i 53.119 ID di cui ho le spiegazioni
    target_ids = [index_to_id[i] for i in range(len(index_to_id))]
    total_targets = len(target_ids)

    # estraggo l'indice numerico reale (nella matrice ordinata da 0 a 65536) per questi 53.119 latent. Ad esempio: "layers.14_latent18" -> 18
    original_indices = [int(tid.split("latent")[-1]) for tid in target_ids]

    # advanced indexing: estraggo solo i vettori dei 53k latent di cui ho le spiegazioni. Forma della nuova matrice: [53119, 2048]
    W_enc_subset = W_enc_norm[original_indices]

    # creo una lista che conterrà ogni latent e i rispettivi top k vicini
    all_neighbors_data = []

    # creo un dizionario per salvare, per ogni latent, la lista delle sue similarità semantiche
    master_semantic_data = {}

    print(f"Calcolo (a blocchi) dei top {MAX_K} vicini per i {total_targets} selezionati...")

    # calcolo a blocchi per il calcolo dei vicini. NOTA_1: 'i' assume il ruolo di indice di partenza del batch corrente (con misura del batch = 1000)...
    for i in range(0, total_targets, BATCH_SIZE):
        # ...fermandosi all'ultimo step valido (non può andare oltre il numero totale di latent)
        end_idx = min(i + BATCH_SIZE, total_targets)

        # seleziono i vettori del blocco corrente
        batch_vectors = W_enc_subset[i:end_idx]

        # prodotto matrice [BATCH_SIZE, 2048] per matrice trasposta [2048, 53119]. Matrice risultante: [BATCH_SIZE, 53119]
        similarities = torch.matmul(batch_vectors, W_enc_subset.T)

        # uso la funzione .topk() per trovare gli indici dei top K+1 (da scartare il primo indice dato che è il latent stesso)
        # top_values sono i top-k valori trovati, ma non utili per la mia analisi (scritti perché la funzione restituisce una tupla, ma avanti considero solo top_idx)
        top_values, top_idx = torch.topk(similarities, MAX_K + 1, dim=1) # top_idx è un tensore 2D

        # itero su ogni latent. NOTA_2: 'local_idx' è un contatore incrementale sulla singola riga
        for local_idx in range(len(batch_vectors)):
            # calcolo l'indice effettivo del latent target di questa iterazione
            actual_idx = i + local_idx
            # estraggo l'ID del latent target ecorrent
            target_id = target_ids[actual_idx]

            # creo una lista dei k vicini (trovo il latent target sulla dimensione 0 con 'local_idx',
            # estraggo i top-k vicini escludendo il primo - l'indice del latent stesso)
            top_neighbors_indices = top_idx[local_idx, 1:].tolist()

            # uso questi indici (da 0 a 53119) per recuperare i nomi esatti dei latent vicini da salvare in una lista
            neighbor_ids = [target_ids[idx] for idx in top_neighbors_indices]

            # creo la "riga" (un dizionario) del latent corrente da appendere alla lista all_neighbors_data
            row_data = {"LatentID": target_id}
            # salvo nel dizionario appena creato i top k vicini
            for k in range(MAX_K):
                row_data[f"Neighbor_{k+1}"] = neighbor_ids[k]

            all_neighbors_data.append(row_data)

            # percorso del file che contiene le similarità tra un latent x e tutti gli altri 53119 latent (anche con sé stesso)
            sim_file_path = os.path.join(SIMILARITY_DIR, f"{target_id}.parquet")

            # mi assicuro che il percorso esista, proteggo il codice da file corrotti o mancanti
            if not os.path.exists(sim_file_path):
                # in caso di file mancante, inserisco una lista vuota nel master e vado avanti
                master_semantic_data[target_id] = []
                continue

            # leggo il file .parquet come dataframe pandas usando come engine pyarrow
            df_sim = pd.read_parquet(sim_file_path, engine="pyarrow")

            # creo una lista che conterrà le similarità dei k vicini
            semantic_similarities = []

            # ciclo sul numero massimo dei vicini
            for k in range(MAX_K):
                # estraggo l'ID del vicino direttamente dalla lista appena generata
                n_id = neighbor_ids[k]

                # nella mappatura latent_id-indice, trovo la riga esatta in cui si trova il valore nel file con le 53119 similarità
                row_index = id_to_index[n_id]

                # estraggo il valore della similarità usando la posizione esatta (con .iat uso la coppia riga/colonna)
                sim_val = df_sim.iat[row_index, 0]
                # aggiungo la similarità alla lista che conterrà le k similarità dei vicini del target
                semantic_similarities.append(sim_val)

            # salvo le similarità nel dizionario master associare all'id del latent target
            master_semantic_data[target_id] = semantic_similarities

        # stampo un feedback (ogni 5000 latent) per avere contezza del progresso
        if (i + BATCH_SIZE) % 5000 == 0 or (i + BATCH_SIZE) >= total_targets:
            print(f"  ... {min(i + BATCH_SIZE, total_targets)} / {total_targets} completati.")

    # pulizia della memoria tramite il garbage collector di python
    del W_enc_norm
    del W_enc_subset
    del similarities
    gc.collect()

    # pandas trasforma una lista di dizionari (struttura chiave-valore) in una "tabella", dove:
    # - le "colonne" (gli header) sono le chiavi
    # - i contenuti delle "celle delle righe" sono i valori
    df_knn = pd.DataFrame(all_neighbors_data)
    # index=False impedisce a Pandas di scrivere nel file una colonna inutile contenente i numeri di riga sequenziali
    df_knn.to_parquet(KNN_OUTPUT_FILE, index=False)
    print(f"Dati KNN salvati in {KNN_OUTPUT_FILE}")

    # restituisco in output il dizionario contenente tutte le similarità pre-calcolate
    return master_semantic_data


# ---------------------------------------------------------------
# 4. FUNZIONE PER L'ANALISI QUANTITATIVA (eseguita per ogni k)
# ---------------------------------------------------------------
def quantitative_analysis_func(K_NEIGHBORS, master_semantic_data, METRICS_OUTPUT_FILE, PLOT_DENSITY_FILE, PLOT_UMAP_FILE, UMAP_CSV_PATH):
    print(f"\n--- INIZIO ANALISI QUANTITATIVA CONSIDERANDO I TOP {K_NEIGHBORS} VICINI ---")

    # ------------------------------
    # FASE A: ESTRAZIONE METRICHE
    # ------------------------------
    print(f"Calcolo Metriche (Media, Mediana, Min, Max) dei top {K_NEIGHBORS} vicini...")

    # creo la lista che conterrà i valori delle metriche calcolate
    metrics_data = []

    # itero sul dizionario master precalcolato
    for target_id, all_similarities in master_semantic_data.items():
        # salto i latent che eventualmente avevano la lista vuota
        if not all_similarities:
            continue

        # taglio la lista delle similarità fermandomi all'indice del k corrente
        sliced_similarities = all_similarities[:K_NEIGHBORS]

        # aggiungo alla lista delle metriche un dizionario contenente: id, media, mediana, minimo, massimo
        metrics_data.append({
            "LatentID": target_id,
            "mean_sim_knn": np.mean(sliced_similarities),
            "median_sim_knn": np.median(sliced_similarities),
            "min_sim_knn": np.min(sliced_similarities),
            "max_sim_knn": np.max(sliced_similarities)
        })

    # salvo le metriche calcolate in un df pandas e quindi in un file .csv
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics.to_csv(METRICS_OUTPUT_FILE, index=False)
    print(f"Metriche calcolate e salvate in {METRICS_OUTPUT_FILE}")

    # -----------------------
    # FASE B: DENSITY PLOT
    # -----------------------
    print("Generazione density plot...")

    # imposto il tema
    sns.set_theme(style="whitegrid")

    # creo una singola tela invece di una griglia 2x2
    # estraggo la tupla generata dalla funzione subplots() e salvo i due oggetti in due variabili distinte
    fig, ax = plt.subplots(figsize=(12, 8))
    # imposto il titolo della figura
    fig.suptitle(f"Confronto metriche nei top {K_NEIGHBORS} vicini spaziali", fontsize=16)

    # creo una lista dei colori da usare per ognuno dei 4 grafici
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    # creo una lista con i nomi delle colonne delle 4 metriche del df pandas precedentemente salvato
    columns_to_plot = ["mean_sim_knn", "median_sim_knn", "min_sim_knn", "max_sim_knn"]
    # creo una lista con le etichette di ciascuno dei 4 grafici (le etichette andranno nella legenda)
    labels = ["Media", "Mediana", "Minimo", "Massimo"]

    # itero sugli indici delle colonne da plottare
    for i in range(len(columns_to_plot)):
        # estraggo il nome della colonna della metrica che analizzo in questa iterazione
        col = columns_to_plot[i]
        # disegno tutto le linee sullo stesso oggetto 'ax'
        # aggiungo il parametro 'label' per la legenda e abbasso l'alpha a 0.4 per evitare sovrapposizioni invasive
        sns.kdeplot(data=df_metrics, x=col, fill=True, color=colors[i], ax=ax, alpha=0.4, label=labels[i])

    # imposto nomi e assi una sola volta fuori dal ciclo
    ax.set_title("Density Plot delle 4 Metriche", fontsize=14)
    ax.set_xlabel("Similarità Coseno")
    ax.set_ylabel("Densità")
    ax.set_xlim(0.0, 1.0)

    # invoco la legenda per mostrare a quale metrica corrisponde ogni colore
    ax.legend(fontsize=12, loc="upper left")

    # ridimensiono in modo intelligente la figura creata (aggiunta padding ecc.)
    plt.tight_layout()
    # salvo la figura creata
    plt.savefig(PLOT_DENSITY_FILE, dpi=300, bbox_inches="tight")
    # elimino la tela creata per liberare memoria RAM
    plt.close()
    # stampo un feedback
    print(f"Density plot salvati in {PLOT_DENSITY_FILE}")

    # ------------------------------------------------------
    # FASE C: GRAFICO UMAP COLORATO PER SIMILARITA' MEDIA
    # ------------------------------------------------------
    print("Generazione Mappa UMAP Cromatica...")

    # mi assicuro che il file .csv con le coordinate UMAP esista
    if not os.path.exists(UMAP_CSV_PATH):
        print(f"File '{UMAP_CSV_PATH}' non trovato. Salto la fase C.")
        return

    # apro il file .csv come un dataframe pandas
    df_umap = pd.read_csv(UMAP_CSV_PATH)

    # creo una tabella contenente, per ogni latent ID, sia le coordinate UMAP che il valore della media della similarità
    # inner join mi garantisce che la tabella finale non abbia lacune
    # left_on e right_on mi permettono di specificare la "chiave", non avendo lo stesso nome nelle due tabelle.
    # Per questo nella tabella finale compaiono entrambi (due colonne di stessi dati -- i latent ID)
    df_merged = pd.merge(df_umap, df_metrics[["LatentID", "mean_sim_knn"]],
                         left_on="Latent ID", right_on="LatentID", how="inner")

    # elimino una delle due colonne contenente gli stessi dati (nel df resta la colonna "Latent ID") e sovrascrivo
    df_merged = df_merged.drop("LatentID", axis=1)

    # creo la figura, specificando le dimensioni
    plt.figure(figsize=(14, 10))
    # uso lo stile base della libreria per la rappresentazione grafica
    plt.style.use("default")

    # creazione del grafico UMAP
    scatter = plt.scatter(
        df_merged["UMAP 1"], # ascissa
        df_merged["UMAP 2"], # ordinata
        c=df_merged["mean_sim_knn"], # valori da usare
        cmap="winter", # mappa dei colori
        s=10, # dimensione del punto
        alpha=0.8, # livello di trasparenza
        vmin=0.0, # valore minimo a cui associare il colore blu
        vmax=1.0 # valore massimo a cui associare il colore verde
    )

    # impostazioni della colorbar con limiti esplicitati
    cbar = plt.colorbar(scatter, label="Media Similarità Coseno", ticks=[0.0, 1.0])
    cbar.ax.set_yticklabels(["0.0 (Min)", "1.0 (Max)"])

    # titolo del grafico e spaziatura (pad)
    plt.title(f"UMAP colorato per similarità coseno media dei top {K_NEIGHBORS} vicini", fontsize=16, pad=15)

    # nomi dell'asse x e y
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)

    # salvataggio della figura
    plt.savefig(PLOT_UMAP_FILE, dpi=300, bbox_inches="tight")
    # liberazione della memoria dalla figura creata
    plt.close()
    # stampo feedback
    print(f"Visualizzazione UMAP salvata in {PLOT_UMAP_FILE}")


if __name__ == "__main__":
    # ------------------------------------------
    # 1. CONFIGURAZIONE PARAMETRI E PERCORSI
    # ------------------------------------------

    # parametro
    BATCH_SIZE = 1000 # dimensione del blocco per non saturare la RAM

    # percorsi di input

    # ISTRUZIONI PER L'UTENTE GITHUB:
    # 1. Scarica il modello SAE da: https://huggingface.co/alessandrobondielli/sae-Minerva-1B-32x
    # 2. Inserisci qui sotto il TUO percorso locale assoluto fino alla cartella 'layers.14'
    # Esempio: SAE_WEIGHTS_PATH = "/home/user/.cache/huggingface/hub/..."
    SAE_WEIGHTS_PATH = ""

    # controllo di sicurezza per gli utenti github
    if not SAE_WEIGHTS_PATH or not os.path.exists(SAE_WEIGHTS_PATH):
        print("\n" + "="*70)
        print("ERRORE: PERCORSO MODELLO SAE NON CONFIGURATO O NON VALIDO")
        print("="*70)
        print("Per eseguire questo script è necessario possedere i pesi del modello SAE.")
        print("1. Scarica il modello da: https://huggingface.co/alessandrobondielli/sae-Minerva-1B-32x")
        print("2. Apri questo script e modifica la variabile 'SAE_WEIGHTS_PATH' inserendo")
        print("   il percorso assoluto del file scaricato sul tuo computer.")
        print(f"Percorso attualmente letto: '{SAE_WEIGHTS_PATH}'")
        print("="*70 + "\n")
        sys.exit(1) # il codice 1 indica al sistema operativo un'uscita per errore

    SIMILARITY_DIR = "../3_matrix_similarity/results/similarity_parts"
    INDEX_MAPPING_FILE = "../3_matrix_similarity/results/latent_index_mapping.json"
    UMAP_CSV_PATH = "dataset/dataset_umap.csv"  # generato nello step 4a_umap

    # percorsi di output
    DATASET_DIR = "dataset"
    RESULTS_DIR = "results"

    # verifico se esistono le cartelle, altrimenti le creo
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # lista dei valori da assegnare a k
    k_values = [1, 3, 5, 10, 15, 20, 25]

    # trovo il massimo valore di k (utile la prima volta per calcolare il numero massimo di vicini)
    MAX_K = max(k_values)
    # creo il percorso per il file che conterrà tutti i MAX_K vicini di ogni latent
    GLOBAL_KNN_FILE = os.path.join(DATASET_DIR, f"all_latents_{MAX_K}_neighbors.parquet")

    # -----------------------------------------
    # 2. CARICAMENTO DATI E AVVIO ITERAZIONI
    # -----------------------------------------

    # carico i file più grandi (da riusare a ogni iterazione) una sola volta
    print("Caricamento globale pesi SAE e mappature in RAM...")
    state_dict = load_file(SAE_WEIGHTS_PATH)
    W_enc_full = state_dict["encoder.weight"] # forma originale: [65536, 2048]

    # carico la mappatura index-latent id salvata in .json
    with open(INDEX_MAPPING_FILE, "r") as f:
        mapping = json.load(f)

    # avvio il pre-cacolo pesante passando esplicitamente gli argomenti alla funzione, vedi "3. FUNZIONE PER L'ESTRAZIONE GLOBALE"
    master_data = precompute_global_data(
        W_enc_full=W_enc_full,
        mapping=mapping,
        MAX_K=MAX_K,
        BATCH_SIZE=BATCH_SIZE,
        SIMILARITY_DIR=SIMILARITY_DIR,
        KNN_OUTPUT_FILE=GLOBAL_KNN_FILE
    )

    # cancello la matrice dei pesi originale dato che non serve più
    del W_enc_full
    gc.collect()

    # itero sui vari k passando solo i dati "leggeri" alla funzione di plotting
    for k_value in k_values:
        RESULTS_DIR_KNN = f"{RESULTS_DIR}/top_{k_value}"
        METRICS_OUTPUT_FILE = os.path.join(DATASET_DIR, f"latents_metrics_{k_value}_neighbors.csv")
        PLOT_DENSITY_FILE = os.path.join(RESULTS_DIR_KNN, f"density_plots_{k_value}_neighbors.png")
        PLOT_UMAP_FILE = os.path.join(RESULTS_DIR_KNN, f"umap_colored_{k_value}_neighbors.png")

        # mi assicuro che la cartella di output esista per questo specifico K. Se esiste vado avanti, altrimenti la creo
        os.makedirs(RESULTS_DIR_KNN, exist_ok=True)

        # chiamata alla funzione
        quantitative_analysis_func(
            K_NEIGHBORS=k_value,
            master_semantic_data=master_data,
            METRICS_OUTPUT_FILE=METRICS_OUTPUT_FILE,
            PLOT_DENSITY_FILE=PLOT_DENSITY_FILE,
            PLOT_UMAP_FILE=PLOT_UMAP_FILE,
            UMAP_CSV_PATH=UMAP_CSV_PATH
        )

    print("\n ANALISI QUANTITATIVA DELLO STEP 4 COMPLETATA CON SUCCESSO!")