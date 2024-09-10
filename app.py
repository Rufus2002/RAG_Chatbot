#%% packages
import os
import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import groq
from groq import Groq
client = Groq(api_key=("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"))

#%% data prep
chroma_client = chromadb.PersistentClient(path="db")
chroma_collection = chroma_client.get_or_create_collection("ipcc")

#%%
def rag(query, n_results=5):
    res = chroma_collection.query(query_texts=[query], n_results=n_results)
    docs = res["documents"][0]
    joined_information = ';'.join([f'{doc}' for doc in docs])
    chat_completion=client.chat.completions.create(
      model="llama3-8b-8192",
      messages=[
          {"role": "system", "content": "You are a climate specialist. Answer the user's question based on the provided document."},
              {"role": "user", "content":  f"Question: {query}. \n Information: {joined_information}"}],
    )
    content = chat_completion.choices[0].message.content
    return content

#%%
st.header("Climate Change Chatbot")

# text input field
user_query = st.text_input(label="", help="Ask here to learn about Climate Change", placeholder="What do you want to know about climate change?")

rag_response, raw_docs = rag(user_query)

st.header("Raw Information")
print(f"raw at 0: {raw_docs[0]}")
print(len(raw_docs))
st.text(f"Raw Response 0: {raw_docs[0]}")
st.text(f"Raw Response 1: {raw_docs[1]}")
st.text(f"Raw Response 2: {raw_docs[2]}")
st.text(f"Raw Response 3: {raw_docs[3]}")
st.text(f"Raw Response 4: {raw_docs[4]}")


st.header("RAG Response")
st.write(rag_response)
