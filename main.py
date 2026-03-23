import pandas as pd 
import streamlit as st
import ollama

st.title("InsightAI App")

uploaded_file = st.file_uploader("upload csv file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # same prompt style as your original code
    prompt = f"""
You are a professional data analyst.

Analyze the dataset below and write a detailed, human-like report.

Dataset sample:
{df.head().to_csv(index=False)}

Instructions:
- Describe what the dataset is about
- Explain each column in simple terms
- Mention patterns or observations
- Include small examples using the data (like explaining one row)
- Write in paragraph style (not bullet points)
- Make it easy to understand like a report

Generate a detailed summary.
"""

    # 🔥 Ollama replaces OpenAI here
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    st.subheader("AI Summary")

    # ✅ fixed typo also (st.write)
    st.write(response["message"]["content"])