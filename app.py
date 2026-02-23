import streamlit as st
import pandas as pd
import requests
import os

st.title("🏃‍♂️ VS Code LLM Auditor")

# --- SETTINGS ---
API_KEY = "your_groq_key_here" # Get it at console.groq.com

@st.cache_data
def load_data():
    if os.path.exists("TrackData_Clean.csv"):
        return pd.read_csv("TrackData_Clean.csv")
    return None

df = load_data()

if df is not None:
    if prompt := st.chat_input("Ask a question:"):
        st.chat_message("user").write(prompt)
        
        # Connect to the AI Website
        with st.chat_message("assistant"):
            try:
                res = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": f"Data: {df.head(10).to_string()}\n\nQuestion: {prompt}"}]
                    }
                )
                st.write(res.json()['choices'][0]['message']['content'])
            except:
                st.error("AI connection failed. Check your API key.")
else:
    st.error("Please run 'python data_fix.py' in the terminal.")