import warnings

from azure_connections import get_azure_embedding_model, get_llm_from_lmstudio
from ddgs_scripts import ddgs_results, web_search_and_format
from faiss_code import load_or_build_vectorstore, make_retriever
from rag_structure import build_rag_chain, keywords_generation, rag_answer
from ragas_scripts import ragas_evaluation
from utils import Settings, load_documents, scan_docs_folder

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    settings = Settings()

    name_index = input(
        "Inserisci il nome dell'indice FAISS (o premi Invio per 'faiss_index'): "
    ).strip()
    if name_index:
        name_index = settings.set_persist_dir_from_query(name_index)

    print(f"📂 Usando indice FAISS in: {settings.persist_dir}")

    # 1) Componenti
    embeddings = get_azure_embedding_model()
    llm = get_llm_from_lmstudio()

    # 2) Dati simulati e indicizzazione (load or build)
    req = (
        input("Vuoi eseguire una ricerca web per arricchire il contesto? (s/n): ")
        .strip()
        .lower()
    )
    if req == "n":
        # docs = simulate_corpus()
        file_paths = scan_docs_folder("docs_test")
        docs = load_documents(file_paths)
        vector_store = load_or_build_vectorstore(settings, embeddings, docs)
    else:
        query = input("Inserisci il termine di ricerca web: ").strip()
        print(f"🔍 Eseguo ricerca web su: {query}")
        keywords = keywords_generation(query)
        print(f"🔑 Keywords generate: {keywords}")
        str_new = " ".join(keywords)
        ddgs_request = ddgs_results(str_new)
        print(ddgs_request)
        all_docs = []
        for url in ddgs_request:
            docs = web_search_and_format(url)
            all_docs.extend(docs)
        print(f"📚 Caricati {len(all_docs)} documenti dalla ricerca web")
        vector_store = load_or_build_vectorstore(settings, embeddings, all_docs)

    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, settings)

    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)

    while True:
        question = input("Inserisci la tua domanda (o 'exit'/'quit'/'q' per uscire): ")
    # questions = [
    #     "Qual è il sogno dell'aquila astronata?",
    #     "Che dieta pratica il leone vegano?",
    #     "Di che colore è un cavallo felice?",
    #     "Quando sparisce il pappagallo invisibile?",
    # ]

#     ground_truth = {
#         questions[0]: """L'aquila astronauta possiede il sogno di esplorare lo spazio.""",
#         questions[1]: """Il leone vegano, una specie di leone molto rara ha scelto una dieta completamente vegana, nutrendosi 
# esclusivamente di arcobaleni e fiori di zucchero.""",
#         questions[2]: """Un
# cavallo felice appare come un arcobaleno vivente, mentre uno arrabbiato diventa completamente
# trasparente.""",
#         questions[3]: """Il pappagallo invisibile sparisce 
# completamente alla vista quando percepisce la parola “banana”""",
#     }

    # Controlla se l'utente vuole uscire
        if question.lower().strip() in ["exit", "quit", "q", "esci"]:
            print("👋 Arrivederci!")
            break

        # for question in questions:
        print("=" * 80)
        print("Q:", question)
        print("-" * 80)
        ans = rag_answer(question, chain)
        print(ans)
        print()

        rag_eval = ragas_evaluation(
            question, chain, llm, embeddings, retriever, settings, 
            # ground_truth=ground_truth
        )
        print("\n METRICHE OTTENUTE:\n", rag_eval)


if __name__ == "__main__":
    main()
