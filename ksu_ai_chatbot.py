from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser

import os
import glob
from uuid import uuid4
from getpass import getpass

ROOT_DIR = f"{os.getcwd()}/data"

def load_documents(root_dir:str, vector_db:Chroma) -> None:
    """
    Loads text documents from a directory into a Chroma Vector Database

    Args:
        root_dir (str): the root_directory of all KSU AI Documents

    Returns:
        Chroma: a Chroma Vectore Database Containing all of KSU AI Documents, 
        Later on used for RAG 
    """
    
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)
    txt_document_paths = glob.glob("*.txt", root_dir=root_dir)
    
    for path in txt_document_paths:
        txt_doc_path = os.path.join(root_dir, path)
        txt_doc = TextLoader(txt_doc_path)
        split_txt_docs = text_splitter.split_documents(txt_doc.load()) 
        
        uuids = [str(uuid4()) for _ in range(len(split_txt_docs))]
        vector_db.add_documents(documents=split_txt_docs, ids=uuids)


def prompt() -> None:
     
    template = """
    Using your previous knowledge and the newly addded context: {context}
    
    Reply to the user's response: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt

def llm_chain(retriever: Chroma, prompt: ChatPromptTemplate, llm: OpenAI):
    """
    Creates a langchain connection the Vector Databse to the LLM

    Args:
        retriever (Chroma): chroma retriever
        prompt (ChatPromptTemplate): chat prompt
        llm (OpenAI): large language model (from OpenAI)
    """
    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser()
    )

    return rag_chain

def ksu_ai_bot(question:str) -> str:
    OPENAI_API_KEY = open(f"{os.getcwd()}/api_key.txt", "r").read()
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    vector_db = Chroma(
        embedding_function=OpenAIEmbeddings()
    )
    
    load_documents(ROOT_DIR, vector_db)
    
    retriever = vector_db.as_retriever(kwargs={"k": 5})
    llm = OpenAI(temperature=0.9)
    llm_prompt = prompt()

    rag_chain = llm_chain(retriever, llm_prompt, llm)
    # question = input("Ask a question to KSU AI chatbot: ")
    output = rag_chain.invoke(question)
    
    
    return output

if __name__ == "__main__":
    ksu_ai_bot()