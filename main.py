import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from pprint import pprint
from ollama import Client
import streamlit as st
import pandas as pd
import datetime
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import ollama


# Load Secrets
load_dotenv()
api_key = os.getenv("API_KEY")

# Chroma Settings
CHROMA_DATA_PATH = "chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "sum_collection"

# Session Variables
if "min_date" not in st.session_state:
    st.session_state.min_date = datetime.date(datetime.datetime.now().year, 1, 1)

if "max_date" not in st.session_state:
    st.session_state.max_date = datetime.date(datetime.datetime.now().year, 12, 31)

if "top_n" not in st.session_state:
    st.session_state.top_n = 25

if "temperature" not in st.session_state:
    st.session_state.temperature = 50

if "top_p" not in st.session_state:
    st.session_state.top_p = 90

chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(EMBED_MODEL)


collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)


tp_path = "rule_docs/StoaTeamPolicyRules2024-25.pdf"
ld_path = "rule_docs/StoaLincolnDouglasRules2024-25_v2.pdf"
parli_path = "rule_docs/StoaParliRules2023-24.pdf"

rule_docs = []


for path in [tp_path, ld_path, parli_path]:
    loader = PyPDFLoader(path)
    rule_docs.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
chunks = text_splitter.split_documents(rule_docs)
chunks = filter_complex_metadata(chunks)

rule_collection = chroma_client.get_or_create_collection(
    name="rule_collection",
    embedding_function=embedding_func,
    metadata={"hnsw:space": "cosine"},
)

for chunk in chunks:
    doc = dict(chunk)

    rule_collection.add(
        documents=doc["page_content"],
        ids=[f"id{chunks.index(chunk)}"],
        metadatas=[{"source": doc["metadata"]["source"], "page": doc["metadata"]["page"]}],
    )

ollama_client = Client(host='http://host.docker.internal:11434')


# UI Titles
st.title("AI Case Maker", anchor=None)
st.subheader("Create Case Ideas from Congress", anchor=None)

def get_summary(summaries, collection=collection):
    df = pd.DataFrame(columns=["date", "title", "text"])

    collect_dict = collection.get()


    for summary, index in zip(summaries, range(0, len(summaries))):
        summary = dict(summary)

        date = BeautifulSoup(summary["updateDate"], "html.parser").get_text()
        title = BeautifulSoup(summary["bill"]["title"], "html.parser").get_text()
        text = BeautifulSoup(summary["text"], "html.parser").get_text()

        if collect_dict["metadatas"]==[]:
            collection.add(
                documents=text,
                ids=[f"id{summaries.index(summary)}"],
                metadatas=[{"date": date, "title": title}],
            )

        elif title not in collect_dict["metadatas"][index]["title"]:
                collection.add(
                    documents=text,
                    ids=[f"id{summaries.index(summary)}"],
                    metadatas=[{"date": date, "title": title}],
                )
        else:
            print("skipped")

        df.loc[len(df)] = [date, title, text]
    return df


def personalized_summary(search_text,  top_n, collection=collection):

    query_results = collection.query(
        query_texts=[search_text],
        n_results=top_n,
    )

    df = pd.DataFrame(columns=["date", "title", "text", "relevence"])

    for document, metadata, distance in zip(query_results["documents"][0], query_results["metadatas"][0], query_results["distances"][0]):

        date = metadata["date"]
        title = metadata["title"]
        relevence = 1-distance
        text = document

        df.loc[len(df)] = [date, title, text, relevence]

    return df




# Response Settings
session = requests.Session()

headers = {"x-api-key": api_key}

session.headers.update(headers)

summaries = ""

# Settings Section
with st.sidebar:
    st.title("Settings:")

    with st.form(key='settings'):
        with st.expander("Change Date Range:"):
            st.session_state.min_date = st.date_input(label="Start Date", value=st.session_state.min_date)
            st.session_state.max_date = st.date_input(label="End Date", value=st.session_state.max_date)

        with st.expander("Change Number of Results:"):
            value=st.session_state.top_n = st.slider(label="Top N", min_value=5, max_value=250, value=st.session_state.top_n)

        with st.expander("Chatbot Settings"):
            st.session_state.temperature = st.slider(label="Temperature", min_value=0, max_value=100, value=st.session_state.temperature)
            st.session_state.top_p = st.slider(label="Top_P", min_value=0, max_value=100, value=st.session_state.top_p)

        update_button = st.form_submit_button(label='Update Settings')


if update_button:
    print(f"Change Date to {st.session_state.min_date} - {st.session_state.max_date}")
    session.params = {"format": "json", "fromDateTime": f"{st.session_state.min_date}T00:00:00Z", "toDateTime": f"{st.session_state.max_date}T23:59:59Z", "sort": " updateDate+desc", "limit": "250"}


# Retrieve Cases
with st.expander("Get List of Cases from Congress"):
    if st.button("Retrieve Cases"):

        session.params = {"format": "json", "fromDateTime": f"{st.session_state.min_date}T00:00:00Z", "toDateTime": f"{st.session_state.max_date}T23:59:59Z", "sort": " updateDate+desc", "limit": "250"}
        response = session.get("https://api.congress.gov/v3/summaries")
        summaries = response.json()["summaries"]

        with st.spinner("Retrieving and Embedding Cases..."):
            summaires_df = get_summary(summaries)
            st.dataframe(summaires_df, use_container_width=True)

# Query Cases
with st.expander("Get List of Personalized Cases from Congress"):
    with st.form(key='search_summaries'):
        text_input = st.text_input(label='Enter a Resolution to Get Related Cases:')
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        if collection.get() == dict({'ids': [], 'embeddings': None, 'metadatas': [], 'documents': [], 'uris': None, 'data': None, 'included': ['metadatas', 'documents']}):
            session.params = {"format": "json", "fromDateTime": f"{st.session_state.min_date}T00:00:00Z", "toDateTime": f"{st.session_state.max_date}T23:59:59Z", "sort": " updateDate+desc", "limit": "250"}
            response = session.get("https://api.congress.gov/v3/summaries")
            summaries = response.json()["summaries"]
            print(summaries)
            with st.spinner("Retrieving and Embedding Cases..."):
                summaires_df = get_summary(summaries)
                personalized_df = personalized_summary(text_input, st.session_state.top_n)
                st.dataframe(personalized_df, use_container_width=True)
        else:
            with st.spinner(f"Retrieving Personalized Cases"):
                personalized_df = personalized_summary(text_input, st.session_state.top_n)
                st.dataframe(personalized_df, use_container_width=True)

# Bot Section

st.markdown("#### Generate Cases with AI")

messages = st.container(height=500)
if prompt := st.chat_input("Say something"):
        query_results = collection.query(
            query_texts=[prompt],
            n_results=1
        )

        rule_results = rule_collection.query(
            query_texts=[prompt],
            n_results=7,
        )
        print("Generating Results")
        print("Documents" + str(rule_results["documents"][0]))

        messages.chat_message("user").write(prompt)

        context_prompt = "You are an AI debate assistant who helps create debate cases and answer questions. USER: " + prompt + " Use the following summaries as data for your answer: {}, and the following rules as context: {}".format(query_results["documents"][0], rule_results["documents"][0])

        print("Context prompt: " + context_prompt)

        with st.spinner('Generating Response'):

            resp = ollama_client.chat(model='llama3:8b', messages=[{'role': 'user', 'content': context_prompt}], options={"temperature":st.session_state.temperature, "top_p": st.session_state.top_p})

        messages.chat_message("assistant").write(resp["message"]["content"])