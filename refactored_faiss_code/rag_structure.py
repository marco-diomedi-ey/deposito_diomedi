import warnings
from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from azure_connections import get_llm_from_lmstudio
from utils import format_docs_for_prompt

warnings.filterwarnings("ignore", category=UserWarning)


def get_contexts_for_question(retriever, question: str, k: int) -> List[str]:
    """Ritorna i testi dei top-k documenti (chunk) usati come contesto."""
    docs = docs = retriever.invoke(question)[:k]
    return [d.page_content for d in docs]


def build_rag_chain(llm, retriever):
    """
    Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
    """
    system_prompt = (
        "Sei un assistente esperto. Rispondi in italiano. "
        "Usa esclusivamente il CONTENUTO fornito nel contesto. "
        "Se l'informazione non è presente, dichiara che non è disponibile. "
        "Includi citazioni tra parentesi quadre nel formato [source:...]. "
        "Sii conciso, accurato e tecnicamente corretto."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Domanda:\n{question}\n\n"
                "Contesto (estratti selezionati):\n{context}\n\n"
                "Istruzioni:\n"
                "1) Rispondi solo con informazioni contenute nel contesto. Immedesimati in un mondo fittizio dove le informazioni vere sono quelle del contesto fornito. \n"
                "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
                "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.' \n"
                "4) Se la risposta è ambigua o incompleta, chiedi ulteriori chiarimenti all'utente. \n"
                "5) Se la domanda riguarda informazioni contrarie al contesto, scrivi la risposta corretta secondo il contesto.\n",
            ),
        ]
    )

    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)


def keywords_generation(query: str):
    llm = get_llm_from_lmstudio()

    prompt = f"""You are a helpful assistant. Generate keywords separated with commas for web search based on the user's query.
    
    Query: {query}
    
    Keywords:"""

    response = llm.invoke(prompt)
    keywords = response.content.strip().split(", ")
    print(keywords)
    return keywords
