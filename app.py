import streamlit as st
import google.generativeai as genai
import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from IPython.display import display
from IPython.display import Markdown
from langchain.chains import RetrievalQA
from langchain import LLMChain
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory


st.title("HR Chatbot 🤖")
@st.cache_resource
def load_and_split_pdf(pdf_path):
    pdf_loader = PyPDFLoader(pdf_path)
    pages = pdf_loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    context = "\n\n".join(str(p.page_content) for p in pages)
    texts = text_splitter.split_text(context)
    return texts
pdf_path = "ZETA_CORPORATION.pdf"
texts = load_and_split_pdf(pdf_path)
GOOGLE_API_KEY=os.getenv('GOOGLE_API')
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY,convert_system_message_to_human=True)
@st.cache_resource
def vector():

    vector_index =  Chroma.from_texts(texts, embeddings)
    vector_index=vector_index.as_retriever(search_type='mmr')
    return vector_index
vector_index = vector()

model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,
                             temperature=0.7,convert_system_message_to_human=True)
# Define the function
def answer_question(model, vector_index):
    template = """be act like a HR officer of "ZETA CORPORATION" and answer the questions to the employye in detail
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain
query = st.text_input('Enter the query')

if st.button('➤'):
    if query:
        qa_chain = answer_question(model, vector_index)
        result = qa_chain({"query": query})

        st.write(result["result"])
    else:
        st.write("Please enter a query.")
