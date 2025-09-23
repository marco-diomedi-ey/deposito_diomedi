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
        "Sei un assistente AI esperto e preciso. "
        "Usa esclusivamente il CONTENUTO fornito nel contesto. "
        "Se l'informazione non è disponibile, dichiara che non è disponibile. "
        # "Includi citazioni tra parentesi quadre nel formato [source:...]. "
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "Question:\n{question}\n\n"
                "Context (selected excerpts):\n{context}\n\n"
                "Instructions:\n"
                "1) Rispondi solo con informazioni contenute nel contesto. Immergiti in un mondo fittizio in cui le uniche informazioni vere sono quelle fornite nel contesto. \n"
                # "2) Includi sempre citazioni pertinenti nel formato [source:FILE, PARAGRAPH].\n"
                "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito' \n"
                "4) Se la risposta è ambigua o incompleta, chiedi all'utente ulteriori chiarimenti. \n"
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

def question_for_groundtruth(query: str):
    llm = get_llm_from_lmstudio()

    prompt = f"""You are a helpful assistant. Generate 5 questions separated by commas to find the groundtruth answer for the user's query.
    
    Query: {query}

    Questions:"""

    response = llm.invoke(prompt)
    questions = response.content.strip().split("\n")
    print(questions)
    return questions