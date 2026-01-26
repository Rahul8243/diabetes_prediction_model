# 🩺 Diabetes Prediction Dashboard  

> **An intelligent Machine Learning web app** that predicts the likelihood of diabetes based on patient medical parameters.  
> Built with **Python**, **Streamlit**, **Scikit-learn**, and **SQLite** for real-time data storage and visualization.

---

## 📖 Overview  

This project showcases how **machine learning** can assist healthcare professionals by providing **instant, data-driven predictions** of diabetes risk.  
The dashboard is **interactive**, **visually appealing**, and designed for **clarity, transparency, and professional use**.

It employs a **Logistic Regression** model trained on the **Pima Indians Diabetes Dataset**, integrated seamlessly with a database-backed UI using Streamlit.

---

## 🧠 Features  

- ⚡ **Instant Diabetes Prediction** (real-time model inference)  
- 🗂️ **SQLite database integration** to store patient records  
- 🧾 **Downloadable CSV reports** of all saved predictions  
- 🧍 **Patient Information Capture** (Name, Email, Contact)  
- 🌗 **Dark-mode optimized UI** with blurred blood background  
- 💡 **Modern, minimal design** with professional layout  
- 🔐 **Ethically transparent AI usage disclaimer**  

---

## ⚙️ Tech Stack  

| Layer | Technologies |
|:------|:-------------|
| **Frontend** | Streamlit |
| **Backend / ML** | Python, Scikit-learn |
| **Database** | SQLite |
| **Visualization** | Streamlit metrics |
| **Model** | Logistic Regression (Binary Classification) |
| **Dataset** | Pima Indians Diabetes Dataset (Kaggle) |

---

## 🧩 Model Workflow  

1. **Data Loading:** Load the `diabetes.csv` dataset.  
2. **Preprocessing:**
   - Standardize numerical columns using `StandardScaler`.  
   - Encode categorical variables if present.  
3. **Split Dataset:** 80% training / 20% testing.  
4. **Model Training:** Logistic Regression (binary classification).  
5. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.  
6. **Prediction:** Probability (%) and class (Diabetic / Non-Diabetic).  
7. **Data Storage:** Save predictions and patient details to SQLite.

---

## 📊 Example Model Performance  

| Metric | Train | Test |
|:--------|:------|:------|
| **Accuracy** | 0.79 | 0.71 |
| **Precision** | 0.76 | 0.61 |
| **Recall** | 0.59 | 0.52 |
| **F1 Score** | 0.66 | 0.56 |

---

## 🧾 Example Patient Input  

| Feature | Example Value |
|:----------|:----------------|
| Name | John Doe |
| Email | johndoe@gmail.com |
| Phone | +91-9876543210 |
| Pregnancies | 2 |
| Glucose | 95 |
| BloodPressure | 70 |
| SkinThickness | 22 |
| Insulin | 85 |
| BMI | 26.3 |
| DiabetesPedigreeFunction | 0.45 |
| Age | 29 |

---

## 🔍 Example Prediction  

**Predicted Probability:** `23.45%`  
**Result:** ✅ Non-Diabetic  

---

## 🧰 Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Rahul8243/Diabetes_Prediction_Dashboard.git
cd Diabetes_Prediction_Dashboard

2️⃣ Create a Virtual Environment
python -m venv venv
venv\Scripts\activate    # Windows


3️⃣ Install Dependencies
pip install -r requirements.txt
 to update pip install --upgrade pip

4️⃣ Run the App
streamlit run diabetes_model_predict.py

🧬 Dataset

Source: Kaggle – https://www.kaggle.com/datasets/mragpavank/diabetes?resource=download

Samples: 768

Features: 8 medical predictors + 1 target label (Outcome)

🔒 Ethical Disclaimer

⚠️ This application is developed for educational and research purposes only.
It is not a substitute for professional medical diagnosis or treatment.
Always consult qualified healthcare providers for clinical decisions.

👨‍💻 Developer

Name: Rahul kumar 
Email: rahulrajmahi611@gmail.com

GitHub: github.com/Rahul8243

LinkedIn: https://www.linkedin.com/in/rahul-kumar-ab8843198/

“Bridging data science and human insight through clean, ethical AI solutions.”

🚀 Future Enhancements

Implement Deep Learning version (TensorFlow / PyTorch)

Deploy on Streamlit Cloud / Hugging Face Spaces

Add user authentication for clinical use

Introduce visual analytics dashboard for hospital reports

🏁 Final Note

This project embodies AI for Good — combining predictive analytics and ethical responsibility to empower healthcare decision-making.
Designed for researchers, students, and AI practitioners aiming to bridge the gap between data science and real-world healthcare.
