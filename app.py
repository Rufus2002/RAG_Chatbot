#Import Necessary Packages
import os
import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from pprint import pprint
from pypdf import PdfReader
import re
import panel as pn
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import groq
from groq import Groq
client = Groq(api_key=("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"))

#Data Preparation
Contract_project_file = "SampleContract-Shuttle.pdf"
reader = PdfReader(Contract_project_file)
contract_texts = [page.extract_text().strip() for page in reader.pages]


char_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0.2
)
texts_char_splitted = char_splitter.split_text('\n\n'.join(contract_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0.2,
    tokens_per_chunk=256
)

# Split text into tokens
texts_token_splitted = []
for text in texts_char_splitted:
    try:
        texts_token_splitted.extend(token_splitter.split_text(text))
    except Exception as e:
        print(f"Error in text: {text} - {e}")
        continue

#Vector Database
chroma_client = chromadb.PersistentClient(path="db")
chroma_collection = chroma_client.get_or_create_collection("Contract_project_file")

# Add documents to the collection
ids = [str(i) for i in range(len(texts_token_splitted))]
chroma_collection.add(
    ids=ids,
    documents=texts_token_splitted
)

#Model
def rag(query, n_results=5):
    print("RAG")
    res = chroma_collection.query(query_texts=[query], n_results=n_results)
    docs = res["documents"][0]
    joined_information = ';'.join([f'{doc}' for doc in docs])
    chat_completion=client.chat.completions.create(
      model="llama3-8b-8192",
      messages=[
          {"role": "system", "content": "You are a Contract Analyst. Answer the user's question based on the provided document."},
              {"role": "user", "content":  f"Question: {query}. \n Information: {joined_information}"}],
    )
    content = chat_completion.choices[0].message.content
    print(content)
    print(docs)
    return content, docs

#Header of the page
st.header("Contract Clauses Chatbot")

# text input field
user_query = st.text_input(label="", help="Ask here to learn about Contract Management", placeholder="What do you want to know about the Project contract?")
rag_response, raw_docs = rag(user_query)

#Raw Information for better analysis
"""st.header("Raw Information")
print(len(raw_docs))
st.text(f"Raw Response 0: {raw_docs[0]}")
st.text(f"Raw Response 1: {raw_docs[1]}")
st.text(f"Raw Response 2: {raw_docs[2]}")
st.text(f"Raw Response 3: {raw_docs[3]}")
st.text(f"Raw Response 4: {raw_docs[4]}")"""


st.header("RAG Response")
st.write(rag_response)
