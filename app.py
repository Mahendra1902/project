import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings

# Set Gemini API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBmUYQdImYbjPJesYFoMHVEfibp5l1CKBc"  # Replace with your Gemini API key

# Initialize LLM and vector database using Gemini and local embeddings with FAISS persistence
from glob import glob
import os

# Initialize LLM and vector database using Gemini and local embeddings with FAISS persistence
def initialize_rag():
    db_path = "faiss_index"
    
    # Load all incident logs from the current directory
    incident_files = glob("*.txt")  # This loads all .txt files in the directory
    all_documents = []
    
    for file in incident_files:
        loader = TextLoader(file)
        all_documents.extend(loader.load())  # Load documents from all .txt files

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Check if FAISS index exists
    if os.path.exists(db_path):
        vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(all_documents)
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(db_path)

    retriever = vector_store.as_retriever()
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.3)
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Dummy functions to simulate agent behavior
def analyze_video_feed():
    return ["No helmet detected near Furnace 3", "Unauthorized entry detected in Zone B"]

def check_sensor_data(sensor_df):
    alerts = []
    if sensor_df['gas_level'].iloc[-1] > 300:
        alerts.append("High gas level detected")
    if sensor_df['temperature'].iloc[-1] > 80:
        alerts.append("High temperature in Boiler Room")
    if sensor_df['noise_level'].iloc[-1] > 85:
        alerts.append("Noise level exceeds safety threshold")
    return alerts

def generate_prevention_checklist():
    return ["Wear helmet and safety gear", "Check gas detector calibration", "Inspect fire extinguishers"]

def generate_compliance_report():
    return "Safety compliance is at 92% this month. Helmet violations decreased by 15%."

# Initialize RAG system
qa_chain = initialize_rag()

def retrieve_similar_incidents(query="helmet violation near furnace"):
    result = qa_chain.run(query)
    return [result]

# Streamlit UI
st.set_page_config(page_title="AI-Powered Safety Monitoring", layout="wide")
st.title("AI-Powered Industrial Safety Monitoring")

st.sidebar.header("Control Panel")
shift_start = st.sidebar.time_input("Shift Start Time", value=datetime.time(8, 0))
selected_area = st.sidebar.selectbox("Select Area", ["Furnace", "Boiler Room", "Assembly Line"])

st.header("Real-time Hazard Alerts")
video_alerts = analyze_video_feed()
sensor_df = pd.DataFrame({
    'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=10, freq='T'),
    'gas_level': np.random.randint(250, 350, 10),
    'temperature': np.random.randint(60, 90, 10),
    'noise_level': np.random.randint(70, 95, 10)
})
sensor_alerts = check_sensor_data(sensor_df)

for alert in video_alerts + sensor_alerts:
    st.error(alert)

st.header("Prevention Checklist")
checklist = generate_prevention_checklist()
for item in checklist:
    st.checkbox(item, value=False)

st.header("Similar Historical Incidents (RAG-based)")
user_query = st.text_input("Describe current risk or incident", value="helmet violation near furnace")
if user_query:
    rag_results = retrieve_similar_incidents(user_query)
    for res in rag_results:
        st.write("-", res)
#header
st.header("Safety Compliance Report")
report = generate_compliance_report()
st.info(report)

st.header("Sensor Readings (Last 10 min)")
st.dataframe(sensor_df)