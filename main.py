import pandas as pd 
import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("InsightAI App")
uploaded_file = st.file_uploader("upload  csv file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    prompt = f"Give a quick summary of this data: \n  {df.head().to_csv(index=False)}"
    resp = client.chat.completions.create(
        model="gpt-4",
        messages = [{"role": "user","content": prompt}]
    )

    st.subheader("AI Summary")
    st.wrirte(resp.choices[0].message.content)