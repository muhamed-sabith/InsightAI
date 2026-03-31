import streamlit as st
import pandas as pd
import ollama

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("🚀 InsightAI - Smart Data Analyzer")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Data")
    st.write(df.head())

    # -----------------------------
    # 🔹 LIMIT DATA (AVOID MEMORY ERROR)
    # -----------------------------
    if len(df) > 3000:
        df = df.sample(3000, random_state=42)
        st.info("Dataset reduced to 3000 rows for faster processing")

    # -----------------------------
    # 1. DATA PREPROCESSING
    # -----------------------------
    st.subheader("🧹 Data Preprocessing")

    df = df.dropna()

    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    st.success("Data cleaned & encoded")

    # -----------------------------
    # 2. FEATURE ENGINEERING
    # -----------------------------
    st.subheader("⚙ Feature Engineering")

    target_column = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    st.write("Features:", list(X.columns))

    # -----------------------------
    # 3. MODEL TRAINING (LIGHTWEIGHT)
    # -----------------------------
    st.subheader("🤖 Model Training")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ lightweight model
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    st.success("Model trained successfully!")

    # -----------------------------
    # 4. ACCURACY EVALUATION
    # -----------------------------
    st.subheader("📈 Accuracy Evaluation")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    st.write(f"Accuracy: {acc:.2f}")

    # -----------------------------
    # 5. AI AUTO SUMMARY (DETAILED)
    # -----------------------------
    st.subheader("🧠 AI Dataset Summary")

    auto_prompt = f"""
    You are a professional data analyst.

    Dataset sample:
    {df.sample(5).to_csv(index=False)}

    Describe:
    - What the dataset is about
    - Explain columns
    - Mention patterns
    - Give small examples

    Write in paragraph style.
    """

    with st.spinner("Generating AI report..."):
        auto_response = ollama.chat(
            model="llama3",
            messages=[{"role": "user", "content": auto_prompt}]
        )

        st.markdown(auto_response["message"]["content"])

    # -----------------------------
    # 6. AI CUSTOM QUESTION
    # -----------------------------
    st.subheader("❓ Ask Questions About Data")

    user_question = st.text_input("Type your question:")

    if st.button("Analyze Question"):
        with st.spinner("Analyzing..."):

            custom_prompt = f"""
            Dataset sample:
            {df.sample(5).to_string()}

            Question: {user_question}

            Give a clear and detailed answer.
            """

            response = ollama.chat(
                model="llama3",
                messages=[{"role": "user", "content": custom_prompt}]
            )

            st.success(response["message"]["content"])