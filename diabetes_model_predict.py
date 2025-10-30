# 💉 Diabetes Prediction Dashboard 
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import base64

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="💉 Diabetes Prediction Dashboard", layout="wide")

def add_bg_from_local(image_file, blur=True):
    """Set a blurred background from a local image."""
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    blur_style = "backdrop-filter: blur(6px);"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
            {blur_style}
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("blood_background.png", blur=True)
text_color = "white"
box_color = "rgba(20, 20, 20, 0.85)"

# ========== LOAD DATA ==========
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df = load_data()

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# ========== METRICS ==========
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Precision": round(precision_score(y_true, y_pred), 3),
        "Recall": round(recall_score(y_true, y_pred), 3),
        "F1 Score": round(f1_score(y_true, y_pred), 3),
    }

train_metrics = get_metrics(y_train, model.predict(X_train))
test_metrics = get_metrics(y_test, model.predict(X_test))

# ========== HEADER ==========
st.markdown(
    f"<h1 style='text-align:center; color:{text_color}; text-shadow: 0 0 8px rgba(255,255,255,0.2);'>💉 Diabetes Prediction Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='text-align:center; color:{text_color}; font-size:18px;'>Predict diabetes risk and manage patient data efficiently.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ========== MODEL PERFORMANCE ==========
st.subheader(f"📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Training Accuracy", f"{train_metrics['Accuracy'] * 100:.1f}%")
col2.metric("Test Accuracy", f"{test_metrics['Accuracy'] * 100:.1f}%")

col3, col4, col5, col6 = st.columns(4)
col3.metric("Precision", f"{test_metrics['Precision']:.2f}")
col4.metric("Recall", f"{test_metrics['Recall']:.2f}")
col5.metric("F1 Score", f"{test_metrics['F1 Score']:.2f}")
col6.metric("Samples", f"{len(df)}")

# ========== DATABASE ==========
conn = sqlite3.connect("patient_data.db")
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    name TEXT,
    email TEXT,
    phone TEXT,
    Pregnancies REAL,
    Glucose REAL,
    BloodPressure REAL,
    SkinThickness REAL,
    Insulin REAL,
    BMI REAL,
    DiabetesPedigreeFunction REAL,
    Age REAL,
    Probability REAL,
    Result TEXT
)"""
)
conn.commit()

# ========== TABS ==========
tab1, tab2, tab3 = st.tabs(
    ["🧾 Prediction Form", "🗂️ Patient Records", "🧹 Data Management"]
)

# ========== 🧾 PREDICTION FORM ==========
with tab1:
    st.subheader("Enter Patient Details")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("👤 Full Name")
            email = st.text_input("📧 Email Address")
            phone = st.text_input("📞 Contact Number")
            Age = st.number_input("Age", 10, 100, 25)
            Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            Glucose = st.number_input("Glucose", 0.0, 300.0, 120.0)
        with col2:
            BloodPressure = st.number_input("Blood Pressure", 0.0, 200.0, 70.0)
            SkinThickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0)
            Insulin = st.number_input("Insulin", 0.0, 900.0, 85.0)
            BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
            DiabetesPedigreeFunction = st.number_input(
                "Diabetes Pedigree Function", 0.0, 2.5, 0.5
            )

        submitted = st.form_submit_button("🔍 Predict Diabetes")

    if submitted:
        input_data = np.array(
            [
                [
                    Pregnancies,
                    Glucose,
                    BloodPressure,
                    SkinThickness,
                    Insulin,
                    BMI,
                    DiabetesPedigreeFunction,
                    Age,
                ]
            ]
        )
        scaled_data = scaler.transform(input_data)
        pred = model.predict(scaled_data)[0]
        prob = model.predict_proba(scaled_data)[0][1] * 100

        result_text = "⚠️ Diabetic" if pred == 1 else "✅ Non-Diabetic"
        color = "#ff4b4b" if pred == 1 else "#2ecc71"

        # Clean professional result card
        st.markdown("## 📋 Prediction Result")
        st.markdown(
            f"""
            <div style="
                background-color:{box_color};
                padding:25px;
                border-radius:16px;
                box-shadow: 0 0 25px rgba(0,0,0,0.4);
                text-align:center;
                ">
                <h3 style="color:{text_color};">Probability: <span style="color:{color};">{prob:.2f}%</span></h3>
                <h3 style="color:{color}; text-shadow: 0 0 10px {color};">{result_text}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cursor.execute(
            """
            INSERT INTO predictions (
                timestamp, name, email, phone,
                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age, Probability, Result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                name,
                email,
                phone,
                Pregnancies,
                Glucose,
                BloodPressure,
                SkinThickness,
                Insulin,
                BMI,
                DiabetesPedigreeFunction,
                Age,
                float(prob),
                "Diabetic" if pred == 1 else "Non-Diabetic",
            ),
        )
        conn.commit()
        st.success("✅ Prediction saved successfully!")

# ========== 🗂️ RECORDS ==========
with tab2:
    st.subheader("📁 Stored Patient Records")
    records = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC", conn)
    st.dataframe(records, use_container_width=True)

# ========== 🧹 DATA MANAGEMENT ==========
with tab3:
    st.subheader("🧹 Manage Database")

    if st.button("❌ Delete All Records"):
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        st.warning("All records have been deleted!")

    del_email = st.text_input("Enter Email to Delete Specific Record")
    if st.button("🗑️ Delete by Email"):
        cursor.execute("DELETE FROM predictions WHERE email=?", (del_email,))
        conn.commit()
        st.info(f"Records for {del_email} deleted (if existed).")

    records = pd.read_sql("SELECT * FROM predictions", conn)
    buffer = io.BytesIO()
    records.to_csv(buffer, index=False)
    st.download_button(
        "⬇️ Download Data as CSV", buffer.getvalue(), "patient_data.csv", "text/csv"
    )

# ========== FOOTER ==========
st.markdown("---")
st.markdown(
    f"""
<h3 style='color:{text_color};'>👨‍💻 About Developer</h3>
<p style='color:{text_color};'>
<b>Name:</b> Rahul kumar <br>
<b>Email:</b> <a href='mailto:rahulrajmahi611@gmail.com' style='color:#64b5f6;'>rahulrajmahi611@gmail.com</a><br>
<b>GitHub:</b> <a href='https://github.com/Rahul8243' style='color:#64b5f6;'>github.com/Rahul8243</a><br>
<b>LinkedIn:</b> <a href='https://linkedin.com/in/rahul8243' style='color:#64b5f6;'>linkedin.com/in/rahul8243</a>
</p>
""",
    unsafe_allow_html=True,
)
