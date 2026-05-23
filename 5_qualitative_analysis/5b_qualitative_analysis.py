import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import glob
import json
from datasets import load_dataset

# --------------------------------------------
# 1. CONFIGURAZIONE E CREAZIONE LISTA LATENT
# --------------------------------------------

# percorsi di input
INDEX_MAPPING_FILE = "../3_matrix_similarity/results/latent_index_mapping.json"
SIMILARITY_DIR = "../3_matrix_similarity/results/similarity_parts"
KNN_FILE = "../4_quantitative_analysis/dataset/all_latents_25_neighbors.parquet"
DATASET_PATH = "dataset/dataset_explanation_and_umap.csv"

# percorso di output
OUTPUT_HTML = "results/UMAP_Interactive_MultiK_Analysis.html"

# CREAZIONE DINAMICA DELLA LISTA DEI LATENT PER L'ANALISI QUALITATIVA

# 1. definisco i latent scelti da me (3)
TARGET_LATENTS = [
    "layers.14_latent46378",  # ho letto le spiegazioni dei punti fisicamente attorno e ci sono spiegazioni molto simili (tutto legato agli orari)
    "layers.14_latent27119",  # ho letto le spiegazioni dei punti fisicamente attorno e ci sono spiegazioni molto simili (tutto legato alla lettera K)
    "layers.14_latent30628",  # ho letto le spiegazioni dei punti fisicamente attorno e ci sono spiegazioni molto simili  (tutto legato all'inizio di una nuova frase)
]

# 2. aggiungo altri 6 latent:
# - 3 latent gold che abbiano media di similarità massima considerando i top: 1, 10, 25 vicini
# - 3 latent gold che abbiano media di similarità minima considerando i top: 1, 10, 25 vicini

# estraggo la lista dei latent GOLD da HuggingFace
print("Caricamento dataset gold da HuggingFace...")
try:
    dataset = load_dataset("colinglab/EXPLAINITA-task1")
    df_gold = dataset["train_gold"].to_pandas()
    # trasformo la colonna in una lista per una ricerca veloce
    gold_latents_list = df_gold["Latent ID"].tolist()
except Exception as e:
    print(f"ERRORE: scarica il dataset da HuggingFace o problema di connessione a HuggingFace: {e}")
    gold_latents_list = []

# estraggo le informazioni dell'analisi quantitativa dello step precedente
# definisco la cartella dove si trovano i .csv dello step 4
METRICS_DIR = "../4_quantitative_analysis/dataset"
# definisco i valori di k di cui voglio cercare il minimo e il massimo
K_TARGETS = [1, 10, 25] # ho scelto questi valori in quanto sono, rispettivamente: estremo inferiore, valore intermedio, estremo superiore delle k analizzate precedentemente

# se sono stati scaricati i gold, procedo con l'estrazione dinamica
if gold_latents_list:
    for k in K_TARGETS:
        metrics_file = os.path.join(METRICS_DIR, f"latents_metrics_{k}_neighbors.csv")
        try:
            # carico il .csv delle metriche per il k corrente
            df_metrics = pd.read_csv(metrics_file)

            # filtro il dataframe con una maschera booleana facendo rimanere solo le righe gold
            df_gold_metrics = df_metrics[df_metrics["LatentID"].isin(gold_latents_list)]

            if not df_gold_metrics.empty:
                # trovo l'indice dela riga con la media massima usando .idxmax() e ne estraggo l'ID
                max_id = df_gold_metrics.loc[df_gold_metrics["mean_sim_knn"].idxmax(), "LatentID"]
                # trovo l'indice dela riga con la media minima usando .idxmax() e ne estraggo l'ID
                min_id = df_gold_metrics.loc[df_gold_metrics["mean_sim_knn"].idxmin(), "LatentID"]

                # appendo i due id trovati alla lista principale
                TARGET_LATENTS.append(max_id)
                TARGET_LATENTS.append(min_id)
                print(f"Per K={k}: aggiunto MAX ({max_id}) e MIN ({min_id})")

        except FileNotFoundError:
            print(f"ATTENZIONE: File metriche non trovato per K={k}: {metrics_file}")

# rimuovo eventuali duplicati usando un trucco: con dict converto la lista in chiavi di dizionario (rimuovendo i doppioni) e la riconverto in lista (mantenendo l'ordine originale)
TARGET_LATENTS = list(dict.fromkeys(TARGET_LATENTS))

print(f"\nLista finale TARGET_LATENTS ({len(TARGET_LATENTS)} latent): {TARGET_LATENTS}\n")

# definisco in una lista i valori di k da esplorare dinamicamente
K_VALUES = [1, 3, 5, 10, 15, 20, 25]

# carico il file .parquet con i vicini precalcolati (fatto nello step 4)
print("Caricamento dei 25 vicini precalcolati...")
df_neighbors = pd.read_parquet(KNN_FILE)

# creo un dizionario dei vicini nello spazio SAE
neighbors_dict = {}
# itero sulle righe del dataframe che contiene SOLO gli id del latent target e dei latent vicini (non il valore di similarità)
for index, row in df_neighbors.iterrows():
    # estraggo l'ID del latent di questa iterazione
    t_id = row["LatentID"]
    # verifico se questo ID c'è nella lista dei latent target
    if t_id in TARGET_LATENTS:
        # creo dinamicamente una lista dei 25 vicini estraendo i valori delle celle da 'Neighbor_1' a 'Neighbor_25'
        neighbors_dict[t_id] = [row[f"Neighbor_{k}"] for k in range(1, 26)]

# apro il file .json con la mappatura indici-latent id
print("Caricamento mappatura indici...")
with open(INDEX_MAPPING_FILE, "r") as f:
    mapping = json.load(f)
    id_to_index_map = mapping["id_to_index"]

# --------------------------------------------------------
# 2. FUNZIONI PER ESTRARRE NUMERO LATENT E LE SIMILARITA'
# --------------------------------------------------------

# estraggo il numero del latent e ci aggiungo spazi a destra finché non raggiungo 5 caratteri (numero massimo di
# caratteri che può avere un numero dell'ID del latent). Utile nella visualizzazione.
def format_latent_name(latent_string):
    num_str = latent_string.split("latent")[-1]
    return num_str.ljust(5, " ") # left justified

# funzione che estrae le similarità dei 25 vicini del latent target e restituisce una Series sparsa
def get_similarity_for_targets(target_ids, similarity_dir, id_to_index_map, neighbors_dict):
    print(f"Estrazione similarità dei top 25 vicini dei latent id target {target_ids}...")
    target_similarities = {}

    for target in target_ids:
        # recupero i 25 vicini (stringa del latent ID) per questo specifico target
        # uso .get() con fallback a lista vuota per evitare errori se il target non è nel dict
        neighbors_for_target = neighbors_dict.get(target, [])

        # se la lista è vuota stampo un avviso per questo latent
        if not neighbors_for_target:
            print(f"ATTENZIONE: Nessun vicino precalcolato trovato per {target}.")
            continue

        file_path = os.path.join(similarity_dir, f"{target}.parquet")

        try:
            # trovo quali sono i numeri di riga corrispondenti ai 25 vicini (il Parquet ha solo una colonna di float e nessuna intestazione)
            rows_to_read = [id_to_index_map[neighbor_id] for neighbor_id in neighbors_for_target]

            # leggo il file
            df_sim = pd.read_parquet(file_path, engine="pyarrow")

            # .iloc estrae solo le righe specifiche basate sui numeri calcolati, restituendo un dataframe pandas (dato che in input aveva una lista di indici)
            # .values restituisce i valori del df in formato numpy, .flatten() trasforma l'array numpy 2D in un array numpy 1D
            extracted_similarities = df_sim.iloc[rows_to_read].values.flatten()

            # creo una Pandas Series associando ogni valore estratto al suo Latent ID. Questa serie avrà 25 elementi (chiave=LatentID, valore=similarità)
            sim_series = pd.Series(data=extracted_similarities, index=neighbors_for_target)

            # salvo nel dizionario
            target_similarities[target] = sim_series

        except FileNotFoundError:
            print(f"ERRORE: Impossibile trovare il file delle similarità per {target} in {file_path}")
        except KeyError as e:
            print(f"ERRORE: Chiave ID non trovata nella mappatura per {target}: {e}")
        except Exception as e:
            print(f"ERRORE per {target}: {e}")

    return target_similarities

# -----------------------------------------
# 3. CARICAMENTO DATI E CONVERSIONE TIPI
# -----------------------------------------
print("Caricamento dataset UMAP e spiegazioni...")

# creo un df pandas leggendo il file csv che contiene, tra le altre cose, le coordinate UMAP
df_umap = pd.read_csv(DATASET_PATH)
# mi assicuro che non ci siano spiegazioni nulle. Se dovessero esserci, riempio con la stringa "Nessuna spiegazione"
df_umap["explanation"] = df_umap["explanation"].fillna("Nessuna spiegazione")
# creo una nuova colonna del df che contiene le spiegazioni della lunghezza opportuna per la visualizzazione
# prendo la spiegazione, rimpiazzo "<" e ">" quando circondano delle lettere con le loro entità HTML (&lt; less than e &gt; greater than). Nella visualizzazione, HTML interpretava questa stringa come codice di markup
# con .str.wrap() inserisco un \n dopo 120 caratteri, sostituisco \n con il tag HTML per l'interruzione di riga <br> (mi serve per la visualizzazione successiva)
df_umap["explanation_wrap"] = (
    df_umap["explanation"]
    .str.replace(r"<([a-zA-Z])>", r"&lt;\1&gt;", regex=True)
    .str.wrap(120)
    .str.replace("\n", "<br>", regex=False)
)

# salvo il dizionario delle similarità estraendo i valori con la funzione definita prima
similarities_dict = get_similarity_for_targets(TARGET_LATENTS, SIMILARITY_DIR, id_to_index_map, neighbors_dict)

# itero (con chiave-valore) sugli elementi del dizionario creato
for target_id, sim_series in similarities_dict.items():
    # creo il nome della colonna con una f string
    col_name = f"sim_{target_id}"
    # creo una colonna del df inserendo, per ogni latent id, la riga contenente le similarità
    df_umap[col_name] = df_umap["Latent ID"].map(sim_series)
    # forzo esplicitamente Pandas a trattare questa colonna come numeri float (mi ha dato un po' di problemi in passato senza questa specifica)
    df_umap[col_name] = pd.to_numeric(df_umap[col_name], errors="coerce")
    # gestisco eventuali NaN rimasti riempiendoli con 0.0 per evitare problemi nei calcoli
    df_umap[col_name] = df_umap[col_name].fillna(0.0)

# ----------------------------
# 4. VISUALIZZAZIONE PLOTLY
# ----------------------------
print("Generazione dashboard interattiva Plotly...")

# creo la figura base plotly
fig = go.Figure()
# creo il testo hover di base tramite concatenazione di tag HTML, stringhe e codice
hover_base = "<b>" + df_umap["Latent ID"] + "</b>" + "<br><br><b>Spiegazione:</b><br>" + df_umap["explanation_wrap"]

# NOTE DI BASE sul codice che segue:
# 1. uso add_trace perché aggiungo alla figura iniziale uno "strato" (traccia)
# 2. uso la classe Scattergl per efficienza

# TRACCIA 0: vista di base (tutti i punti grigi, visibile all'avvio)
fig.add_trace(go.Scattergl(
    x=df_umap["UMAP 1"], y=df_umap["UMAP 2"],
    mode="markers",
    marker=dict(size=4, color="lightgray", line=dict(width=0.5, color="darkgray")),
    text=hover_base, hoverinfo="text",
    name="Tutti i Latent",
    visible=True # questo strato visibile non appena viene aperto il file HTML
))

# calcolo il numero totale delle tracce: 1 (BASE) + (NUM_TARGETS * NUM_K_VALUES * 3 TRACCE PER COMBINAZIONE)
NUM_TOTAL_PERMUTATIONS = len(TARGET_LATENTS) * len(K_VALUES)
NUM_TOTAL_TRACES = 1 + (NUM_TOTAL_PERMUTATIONS * 3)

# preparo la logica per i bottoni del menu
buttons = [ # è una lista in quanto ci sono più pulsanti
    dict( # ogni pulsante è definito da un dizionario
        label="Vista Base (Grigia)",
        method="update", # accetta una lista di dizionari per aggiornare il grafico
        args=[{"visible": [True] + [False] * (NUM_TOTAL_TRACES - 1)}, # primo dizionario: spengo tutte le tracce tranne la 0
              {"title": {
                  "text": "Esplorazione Base", # contenuto testuale del titolo
                  "x": 0.5, # posizione sull'asse orizzontale (da 0 a 1, dove 0.5 è il centro esatto)
                  "xanchor": "center" # specifica che il punto di ancoraggio del testo è il suo centro
              }
              } # secondo dizionario: modifica le proprietà del layout
              ]
    )
]

# variabile contatore per tenere traccia di quale permutazione sto calcolando
permutation_index = 0

# per ogni target, calcolo i vicini e creo le 3 tracce specifiche
for i, target_id in enumerate(TARGET_LATENTS):

    if f"sim_{target_id}" not in df_umap.columns:
        continue

    # se per qualche motivo il target non è nel dizionario, passo oltre
    if target_id not in neighbors_dict:
        continue

    # recupero la lista dei 25 ID vicini precalcolati nello spazio 2048D
    full_neighbors_list = neighbors_dict[target_id]

    # formatto il nome del latent con la funzione apposita (ad es. '18   ') per inserirlo poi nel menù
    formatted_target_name = format_latent_name(target_id)
    target_expl = df_umap[df_umap["Latent ID"] == target_id]["explanation"].values[0]

    # questo bottone funge da intestazione visiva nel menù a tendina. Se cliccato, torna a una visione base
    buttons.append(
        dict(
            # nota: la parola LATENT e i trattini sono scritti con caratteri speciali Unicode per simulare il grassetto (per rendere più netto il distacco)
            label=f"━━━ 𝗟𝗔𝗧𝗘𝗡𝗧 {formatted_target_name.strip()} ━━━",
            method="update",
            args=[{"visible": [True] + [False] * (NUM_TOTAL_TRACES - 1)},
                  {"title": f"Menu Latent {formatted_target_name.strip()} aperto: ora seleziona il valore di K dal menù a tendina."}]
        )
    )

    # itero sui diversi valori di k
    for current_k in K_VALUES:

        # taglio la lista dei vicini in base al k corrente (ad es. prendo solo i primi 5 o i primi 10)
        true_neighbors_list = full_neighbors_list[:current_k]

        # creo tre sotto-dataframe per la visualizzazione tramite filtri logici

        # 1. il punto target (selezionato con l'ID esatto)
        df_target = df_umap[df_umap["Latent ID"] == target_id]

        # 2. i 'current_k' vicini (selezionati usando .isin() che verifica se l'ID è nella mia lista)
        df_vicini = df_umap[df_umap["Latent ID"].isin(true_neighbors_list)]

        # 3. tutti gli altri punti (Sfondo). Uso il simbolo ~ (NOT) per escludere target e vicini
        tutti_da_escludere = [target_id] + true_neighbors_list
        df_sfondo = df_umap[~df_umap["Latent ID"].isin(tutti_da_escludere)]

        # TRACCIA A: lo sfondo (i punti lontani, molto sbiaditi per dare contesto)
        fig.add_trace(go.Scattergl(
            x=df_sfondo["UMAP 1"], y=df_sfondo["UMAP 2"],
            mode="markers",
            marker=dict(size=3, color="rgba(200, 200, 200, 0.1)"), # grigio molto trasparente
            hoverinfo="skip", # disabilito l'hover per lo sfondo per evitare distrazioni
            visible=False
        ))

        # TRACCIA B: gli n vicini (colorati in base alla similarità)
        fig.add_trace(go.Scattergl(
            x=df_vicini["UMAP 1"], y=df_vicini["UMAP 2"],
            mode="markers",
            marker=dict(
                size=10,
                color=df_vicini[f"sim_{target_id}"],
                colorscale="Inferno",
                showscale=True,
                colorbar=dict(title="Sim. Coseno", x=1.05),
                cmin=0.0, cmax=1.0,
                line=dict(width=1, color="white")
            ),

            text="<b>" + df_vicini["Latent ID"] + "</b>" +"<br><br><b>Spiegazione:</b><br>" + df_vicini["explanation_wrap"] +
                 "<br><br><b>Sim. Coseno:</b> " + df_vicini[f"sim_{target_id}"].round(3).astype(str),
            hoverinfo="text",
            name=f"Top {current_k} vicini",
            visible=False
        ))

        # TRACCIA C: il latent target (evidenziato con una stella ciano)
        fig.add_trace(go.Scattergl(
            x=df_target["UMAP 1"], y=df_target["UMAP 2"],
            mode="markers",
            marker=dict(size=18, color="cyan", symbol="star", line=dict(width=2, color="black")),
            text="<b>TARGET:</b> " + df_target["Latent ID"] + "<br><br><b>Spiegazione:</b><br>" + df_target["explanation_wrap"],
            hoverinfo="text",
            name=f"Target {formatted_target_name.strip()}",
            visible=False
        ))

        # CREO LA LOGICA DELLA VISIBILITA' PER QUESTA SPECIFICA COMBINAZIONE (TARGET + K)

        # creo una lista di False lunga quanto il numero totale di tracce
        visibility = [False] * NUM_TOTAL_TRACES

        # calcolo gli indici esatti delle 3 tracce che ho appena aggiunto al grafico (tracce A, B, C: vedi sopra)
        # la traccia 0 è quella Base (che è True solo con "Vista Base (Grigia)" e "LATENT [NUMERO]") e corrisponde quindi
        # al primo elemento della lista di valori booleani. Poi ogni permutazione aggiunge 3 tracce consecutive (e quindi ogni
        # permutazione ha 3 valori booleani nell'array, che partono dal secondo elemento e procedono di 3 in 3).
        trace_offset = 1 + (permutation_index * 3)
        visibility[trace_offset] = True # accendo lo sfondo (puntini grigi sfocati)
        visibility[trace_offset + 1] = True # accendo i vicini (punti da nero - minimo - a giallo - massimo)
        visibility[trace_offset + 2] = True # accendo il latent target (stella)

        # aggiungo il bottone della combinazione latent + K, che imposta la visibilità sopra descritta
        buttons.append(
            dict(
                label=f"Latent {formatted_target_name} (Top {current_k})",
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Latent {target_id} - Analisi dei Top {current_k} Vicini Strutturali<br><sub>Target Expl: {target_expl}</sub>"}]
            )
        )

        # incremento l'inidice di permutazione per il prossimo ciclo
        permutation_index += 1

fig.update_layout(
    # aggiornamento menù rispetto a default: modificata la posizione e la leggibilità quando il cursore va sopra i bottoni
    updatemenus=[dict(
        active=0,
        buttons=buttons,
        x=0.0,
        xanchor="left",
        y=1.15,
        yanchor="top",
        bgcolor="#cbd5e1",
        font=dict(color="#0f172a"),
        bordercolor="#475569"
    )],

    # imposto il titolo testuale di base e configuro il suo allineamento globale
    title_text="Esplorazione Base",
    title_x=0.5, # forzo la posizione x globale a 0.5 (centro)
    title_xanchor="center", # forzo l'ancora globale al centro

    # posiziono la legenda in alto a sinistra
    legend=dict(
        yanchor="top",
        y=0.98,
        xanchor="left",
        x=0.02,
        bgcolor="rgba(15, 23, 42, 0.7)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(color="white")
    ),

    xaxis_title="UMAP 1", yaxis_title="UMAP 2",
    template="plotly_dark", height=800, hovermode="closest"
)

# creo il file HTML di output
fig.write_html(OUTPUT_HTML)
print(f"Dashboard creata con successo: '{OUTPUT_HTML}'.")