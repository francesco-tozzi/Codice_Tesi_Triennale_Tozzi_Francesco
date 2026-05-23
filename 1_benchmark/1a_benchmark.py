import pandas as pd
from sentence_transformers import SentenceTransformer, util
import gc
from scipy.stats import spearmanr
import json

# CONFIGURAZIONE PERCORSI INPUT E OUTPUT

# carico il dataset in input
file_path = "dataset/SimilEx_dataset.csv"
try:
    # creo un dataframe per leggere il .csv
    df = pd.read_csv(file_path)
    print("Dataset caricato con successo!")
except FileNotFoundError:
    print(f"Errore: Il file '{file_path}' non è stato trovato. Controlla il percorso.")
    exit()

# creo il percorso per salvare il file .json di output
benchmark_results_path = "results/benchmark_results.json"

# creo il df che conterrà, per ogni Pair_ID, i valori da confrontare (media giudizi umani e valore di ogni modello)
df_comparison = df[["Pair_ID"]].copy() # doppia parentesi quadra per mantenere la struttura 2D del DataFrame

# ---------------------------------------------------------------------
# STEP 1: CALCOLO LA SIMILARITA' SECONDO GLI UMANI (MEDIA ARITMETICA)
# ---------------------------------------------------------------------

# definisco una lista delle colonne contenenti i voti degli annotatori
cols_annotators = [
    "A1", "A2", "A3", "A4", "A5",
    "A6", "A7", "Stud_1", "Stud_2"
]

# calcolo la media per riga. Specifico axis=1 per fare la media per riga, di default opera su colonna
# dato che le colonne A6, A7, Stud_1 e Stud_2 sono talvolta vuote, imposto il parametro skipna che salta i valori nulli
df_comparison["Media_umana"] = df[cols_annotators].mean(axis=1, skipna=True)

print("\nPrime 5 righe del DataFrame di confronto:")
print(df_comparison.head())

# ---------------------------------------------------
# STEP 2: CALCOLO LA SIMILARITA SECONDO LA MACCHINA
# ---------------------------------------------------

# lista dei modelli da testare e lista (ora vuota) dei nomi brevi
model_names = [
    "sentence-transformers/distiluse-base-multilingual-cased-v1",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
]

short_model_names = []

# estraggo le frasi in formato lista. Aggiungo dei "filtri" difensivi:
# - fillna("") sostituisce i NaN con stringhe vuote
# - astype(str) forza il tipo dato a stringa per evitare crash di SBERT
frasi1 = df["Sentence_1"].fillna("").astype(str).tolist()
frasi2 = df["Sentence_2"].fillna("").astype(str).tolist()

print("\n--- Inizio SBERT e calcolo delle similarità ---")

for model_name in model_names:
    print(f"\nCaricamento modello: {model_name}...")
    model = SentenceTransformer(model_name)

    # genero gli embeddings per tutte le frasi con batch che lavora contemporaneamente su 32 frasi (batch_size=32)
    # il parametro convert_to_tensor=True è utile per velocizzare i calcoli e per la successiva funzione pairwise_cos_sim(), che prende in input due tensori
    embeddings1 = model.encode(frasi1, batch_size=32, convert_to_tensor=True)
    embeddings2 = model.encode(frasi2, batch_size=32, convert_to_tensor=True)

    # calcolo la similarità coseno. Uso la funzione specifica di SBERT per il calcolo coppia per coppia
    similarity = util.pairwise_cos_sim(embeddings1, embeddings2).tolist()

    # estraggo solo il nome effettivo del modello e salvo il coseno calcolato in una colonna del df di confronto
    short_name = model_name.split("/")[-1]
    df_comparison[f"cos_sim_{short_name}"] = similarity

    # creo una lista dei nomi "puliti" del modello, da usare dopo
    short_model_names.append(short_name)

    # stampo un feedback
    print(f"DataFrame di confronto aggiornato: aggiunte le similarità coseno del modello '{short_name}'")

    # elimino esplicitamente modello ed embeddings: evito la saturazione della RAM al passaggio al modello successivo
    del model, embeddings1, embeddings2
    gc.collect()

print("\nDataFrame di confronto aggiornato con le similarità coseno di ogni modello")


# -------------------------------------------------
# STEP 3: CONFRONTO SIMILARITA: UMANI VS MACCHINA
# -------------------------------------------------
print("\n--- Risultati Benchmark (Correlazione di Spearman) ---")

best_score = -1
best_model = ""
comparison_results = {}

# itero sui nomi "puliti" dei modelli
for short_model_name in short_model_names:
    # calcolo Spearman tra "Media_umana" e "cos_sim" del modello
    # spearmanr restituisce due valori: (correlazione, p-value)
    corr, p_value = spearmanr(df_comparison["Media_umana"], df_comparison[f"cos_sim_{short_model_name}"])

    # stampo, per ogni modello, i due valori di Spearman
    print(f"Modello: {short_model_name}")
    print(f" -> Spearman Correlation: {corr:.4f}")
    print(f" -> P-value: {p_value:.4e}")

    # salvo questi risultati in un dizionario
    comparison_results[short_model_name] = {
        "Spearman Correlation": corr,
        "P-value": p_value
    }

    # confronto fra lo score corrente e il migliore per trovare il modello vincitore
    if corr > best_score:
        best_score = corr
        best_model = short_model_name

# salvo il vincitore nel dizionario
comparison_results["WINNER"] = {
    "Model": best_model,
    "Score": best_score
}
# salvo il dizionario dei risultati in un file json in output
with open(benchmark_results_path, "w") as benchmark_results_file:
    json.dump(comparison_results, benchmark_results_file, indent=4) # indent rende il json più leggibile agli umani

# stampo a video il vincitore
print("-" * 60)
print(f"IL VINCITORE È: {best_model}")
print(f"Con una correlazione di: {best_score:.4f}")
print("-" * 60)