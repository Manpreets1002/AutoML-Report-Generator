# 🧠 AutoML Report Generator
AutoML Report Generator is a Streamlit-based application that allows you to upload your dataset, perform automated machine learning tasks (regression or classification), and receive a detailed, AI-generated analysis report. It handles preprocessing, model training, evaluation, and even lets you download a natural language summary powered by LLMs.

## 🚀 Features
📤 Upload your own CSV dataset

🧹 Data cleaning: outlier removal, null value handling, and duplicate removal

📊 Correlation heatmaps for numeric features

⚙️ Model selection: choose between regression or classification

🧠 Auto-trains multiple ML models (Random Forest, XGBoost, CatBoost, etc.)

🏆 Automatically selects and displays the best-performing model

📈 Metrics support: R² Score, Accuracy, F1 Score

🤖 Uses a Groq-powered LLM (Llama3) to generate an in-depth analytical report

📥 One-click download for your machine learning report

## 📦 Tech Stack
Frontend: Streamlit

ML Frameworks: scikit-learn, XGBoost, CatBoost

Data Handling: pandas, NumPy

Visualization: Plotly

LLM Integration: LangChain + Groq + Llama 3

## 📄 Report Output Includes:
Dataset overview and shape

Data cleaning summary

Feature analysis and correlations

Model comparison table

Best model selection explanation

Insights and recommendations

Downloadable .txt report

## ⚙️ Requirements
Make sure to install dependencies:
pip install -r requirements.txt

Include your GROQ_API_KEY in a .env file for LLM support:
GROQ_API_KEY=your_groq_api_key_here

## 🛠 How to Run 
In your Terminal write:
streamlit run report_generator.py

## 📌 Note
The app will fail if you don't provide a valid .env with your Groq API key.

If your target column is a string and you're running regression, you may see errors. Either encode it or switch to classification.
