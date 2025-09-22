import spacy
from spacy import displacy

# Carica il modello per l'italiano
nlp = spacy.load("it_core_news_sm")

# Testo da analizzare
testo = "Elencami tutte le strade di cagliari, con il numero civico, il CAP e la zona di appartenenza."

# Analisi
doc = nlp(testo)

# Output analisi
for token in doc:
    print(f"Token: {token.text:12} | Lemma: {token.lemma_:12} | POS: {token.pos_:10} | Dipendenza: {token.dep_:10} | Head: {token.head.text}")

# Visualizzazione delle dipendenze sintattiche
displacy.serve(doc, style="ent", port=8654, host="127.0.0.1") 
