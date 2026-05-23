NOTE SUL CODICE:

- Il codice è organizzato in modo sequenziale: prima del nome di una cartella o di un file .py si trova l'indicazione numerica sull'ordine. Questo permette la navigazione della pipeline usata in modo agevole. 

- Ogni fase del codice è pensata come sotto-analisi dell'analisi principale: durante ogni fase vengono usati i file presenti nella cartella "dataset" come file di input o di elaborazione intermedia e i file prodotti come output finale vengono salvati nella cartella "results". 

- Le cartelle "results" degli step 2, 3, 4 e 5 sono state caricate vuote per via dei limiti di GitHub. BISOGNA FAR GIRARE IL CODICE PER GENERARE I RISULTATI DEGLI STEP DA 2 A 5.

- Il codice è stato reso efficiente ed eseguibile su PC con appena 8GB di RAM

- ATTENZIONE: il modello SAE e il dataset delle spiegazioni manuali (263) e generate da GPT-5 (3,000) non sono presenti nel codice.
  Essi possono essere facilmente scaricati da HuggingFace ai seguenti link:
  1. SCARICA IL SAE: https://huggingface.co/alessandrobondielli/sae-Minerva-1B-32x
  2. SCARICA IL DATASET: https://huggingface.co/datasets/colinglab/EXPLAINITA-task1
