import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
import plotly.express as px
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score,r2_score,f1_score
import numpy as np
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
st.set_page_config("Report",layout="wide")
st.title("AutoML Report Generator")
st.header("Summary")
st.write("AutoML Report Generator is a Streamlit-based application that allows you to upload your dataset, perform automated machine learning tasks (regression or classification), and receive a detailed, AI-generated analysis report. It handles preprocessing, model training, evaluation, and even lets you download a natural language summary powered by LLMs.")
upload_file = st.file_uploader("Add CSV file for Report",type=["csv"])

if upload_file is None:
    st.info("Add CSV for Report Generation")
else:
    original_df = pd.read_csv(upload_file)

    st.header("Preview Data")
    st.dataframe(original_df)
    
    def outliers_removal(df,columns=None):
        clean_df = df.copy()

        if columns is None:
            columns = clean_df.select_dtypes(include="number").columns

        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            clean_df = clean_df[
                (df[col] >= lower_bound) & (df[col] <= upper_bound)
            ]

        return clean_df
    
    
    remove_outliers = st.sidebar.radio("Remove Outliers",options=['Yes',"No"],index=1)

    if remove_outliers == "Yes":
        clean_df = outliers_removal(original_df)
    else:
        clean_df = original_df.copy()

    drop_col = st.sidebar.multiselect("Columns you want to Drop",options=clean_df.columns.to_list())
    if len(drop_col) == 0:
        pass
    else:
        clean_df.drop(drop_col,inplace=True,axis=1)
    clean_df.drop_duplicates(inplace=True)
    columns = clean_df.columns.tolist()
    null_col = {}
    for col in columns:
        a = int(clean_df[col].isnull().sum())
        if a > 0:
            null_col[col] = a

    st.info(f"NULL Values are:")
    if len(null_col) > 0:
        for x,y in null_col.items():
            st.write(f"{x} : {y}")
    else:
        st.write("No Null Values")
    
    for col in null_col:
        if clean_df[col].dtype == "O":
            clean_df[col].fillna(clean_df[col].mode()[0],inplace=True)
        else:
            if remove_outliers == "Yes":
                clean_df[col].fillna(clean_df[col].mean(),inplace=True)
            else:
                clean_df[col].fillna(clean_df[col].median(),inplace=True)

    st.info("After Processing:")
    for x in null_col.keys():
        st.write(f"{x} : {clean_df[x].isnull().sum()}")

    st.header("Cleaned Data")
    st.dataframe(clean_df)
    
    numeric_df = clean_df.select_dtypes(include="number")

    corr = numeric_df.corr()
    st.subheader("Correlation Matrix")
    st.dataframe(corr.round(2))

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
        labels=dict(color="Correlation")
    )

    st.plotly_chart(fig, use_container_width=True)
    
    target_col = st.sidebar.multiselect("Choose Target Column",options=clean_df.columns.tolist(),max_selections=1)

    if len(target_col) == 0:
        st.error("Choose Target Column")
    else:
        x = clean_df.drop(target_col[0],axis=1)
        y = clean_df[target_col[0]]

        xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)
        obj_col = x.select_dtypes(include="object").columns
        num_col = x.select_dtypes(exclude="object").columns
        
        if len(x.select_dtypes(include='object').columns.to_list()) > 0:
            transformer = ColumnTransformer([
                ("OHE",OneHotEncoder(handle_unknown="ignore"),obj_col),
                ("Std",StandardScaler(),num_col)
            ],remainder="passthrough")
            xtrain_scaled = transformer.fit_transform(xtrain)
            xtest_scaled = transformer.transform(xtest)
        else:
            transformer = StandardScaler()
            xtrain_scaled = transformer.fit_transform(xtrain)
            xtest_scaled = transformer.transform(xtest)

        ytrain_encoded = None
        if ytrain.dtype == "O":
            lb = LabelEncoder()
            ytrain_encoded = lb.fit_transform(ytrain)
            ytest_encoded = lb.transform(ytest)

        model_type = st.sidebar.radio("Choose Model",options=["Regression","Classification"])

        if model_type == "Regression":
            model_performance_choose = st.sidebar.radio("Choose Metrics",options=["R2 Score (For Regression)","Accuracy","F1 Score"],index=0)
        else:
            model_performance_choose = st.sidebar.radio("Choose Metrics",options=["R2 Score (For Regression)","Accuracy","F1 Score"],index=1)
        
        if model_type == "Regression":
            from sklearn.linear_model import LinearRegression,Lasso,Ridge
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.tree import DecisionTreeRegressor
            from xgboost import XGBRegressor
            from catboost import CatBoostRegressor

            models = {
                "Linear Regession" : LinearRegression(),
                "Lasso Regression" : Lasso(),
                "Ridge Regression" : Ridge(),
                "Random Forest Regressor" : RandomForestRegressor(),
                "Decision Tree Regressor" : DecisionTreeRegressor(),
                "XGBoost Regressor" : XGBRegressor(),
                "CatBoost Regressor" : CatBoostRegressor()
            }

            model_performance = {}
            try:
                for i in models.keys():
                    model = models[i]
                    model.fit(xtrain_scaled,ytrain)
                    ypred = model.predict(xtest_scaled)

                    if model_performance_choose == "R2 Score (For Regression)":
                        model_performance[i] = r2_score(ypred,ytest)
                    elif model_performance_choose == "Accuracy":
                        model_performance[i] = accuracy_score(ypred,ytest)
                    else:
                        model_performance[i] = f1_score(ypred,ytest)

            except:
                st.error("""Target Columns is Non Numeric and the user has chosen Regression\\
                    So, Please Map the target column accordingly or use Classification instead.""")
                

            best_model_val = 0
            best_model_name = ""
            for x,y in model_performance.items():
                if best_model_val < y:
                    best_model_val = y
                    best_model_name = x
            
            if best_model_val == 0:
                pass
            else:
                st.header(f"Best Model (Target Columns: {target_col[0]})")
                st.success(f"{best_model_name} : {best_model_val*100}")


        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier
            from xgboost import XGBClassifier
            from catboost import CatBoostClassifier
            from sklearn.naive_bayes import GaussianNB

            models = {
                "Logistic Regession" : LogisticRegression(),
                "Random Forest Classifier" : RandomForestClassifier(),
                "Decision Tree Classifier" : DecisionTreeClassifier(),
                "XGBoost Classifier" : XGBClassifier(),
                "CatBoost Classifier" : CatBoostClassifier(),
            }
            model_performance = {}
            try:
                for i in models.keys():
                    model = models[i]
                    if ytrain_encoded is None:
                        model.fit(xtrain_scaled,ytrain)
                        ypred = model.predict(xtest_scaled)
                    else:
                        model.fit(xtrain_scaled,ytrain_encoded)
                        ypred_encoded = model.predict(xtest_scaled)
                        ypred_encoded = np.round(ypred_encoded).astype(int)
                        ypred = lb.inverse_transform(ypred_encoded)

                    if model_performance_choose == "R2 Score (For Regression)":
                        model_performance[i] = r2_score(ypred,ytest)
                    elif model_performance_choose == "Accuracy":
                        model_performance[i] = accuracy_score(ypred,ytest)
                    else:
                        model_performance[i] = f1_score(ypred,ytest)
            except:
                st.error("""Target Columns is Numeric and the user has chosen Classification\\
                    So, Please use Regression instead.\\
                    If it is Binary Classification Ignore this message """)


            best_model_val = 0
            best_model_name = ""
            for name,acc in model_performance.items():
                if best_model_val < acc:
                    best_model_val = acc
                    best_model_name = name
            
            if best_model_val == 0:
                pass
            else:
                st.header(f"Best Model (Target Column: {target_col[0]})")
                st.success(f"{best_model_name} : {best_model_val*100}")


        if best_model_val == 0:
            pass
        else:
            api_key = os.getenv("GROQ_API_KEY")
            api_key = st.secrets["GROQ_API_KEY"]
            llm_model = ChatGroq(model="Llama3-8b-8192",groq_api_key="api_key")

            report_data = {
                "num_rows": clean_df.shape[0],
                "num_columns": clean_df.shape[1],
                "target": target_col[0],
                "model_used": best_model_name,
                "accuracy": best_model_val,
            }

            work = f"""
            Generate a detailed data analysis report from the following:

            - Rows: {report_data['num_rows']}
            - Columns: {report_data['num_columns']}
            - Target: {report_data['target']}
            - Chosen Model: {report_data['model_used']} with accuracy {report_data['accuracy']}
            - All Trained Models: {model_performance}
            
            Include: dataset summary, model justification, feature analysis, and key insights.
            Also show all trained models(they are not best-performance based sorted).
            """

            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages([
                ("system","You are an expert assisant having expertise in providing detailed data analysis"),
                ("user","{work}")
            ])

            output_parser = StrOutputParser()
            report_generate = prompt | llm_model | output_parser
            response = report_generate.invoke({"work":work})
            st.header("Detailed Analysis")
            st.write(response)

            st.download_button(
                "Download your Report",
                response,
                "ML Report.txt",
                "text/plain"
            )