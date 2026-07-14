import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# 1. Environment variables load karein
load_dotenv(override=True)

# Set up the page title and layout
st.set_page_config(page_title="AI CSV Data Explorer", layout="wide")
st.title("📊 AI-Powered CSV Data Visualizer & Analyzer (Groq Free Edition)")
st.write(
    "CSV file upload karein, advanced charts banayein, aur Groq AI (Llama 3.3) se apne data ke baare me questions puchein!"
)

# 2. Groq API Configure karein using OpenAI library
api_key = os.getenv("GROQ_API_KEY")
api_available = False
client = None

# Groq ke liye standard settings
BASE_URL = "https://api.groq.com/openai/v1" 
MODEL_NAME = "llama-3.3-70b-versatile"  # Groq ka sabse fast aur behtareen model

if api_key:
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
        )
        api_available = True
        st.sidebar.title("⚙️ AI Provider Settings")
        st.sidebar.success(f"✅ Connected to Groq Lite!")
        st.sidebar.info(f"Model: {MODEL_NAME}")
    except Exception as init_error:
        st.sidebar.error(f"⚠️ API Initialization failed: {init_error}")
else:
    st.error("⚠️ GROQ_API_KEY nahi mili! Kripya apni .env file check karein.")


# File Uploader Component
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Display data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))
        st.write(f"**Total Rows:** {df.shape[0]} | **Total Columns:** {df.shape[1]}")

        # Chart Settings Section
        st.markdown("---")
        st.subheader("📈 Chart Settings")

        columns = df.columns.tolist()

        x_axis = st.selectbox("Select X-axis column (Used for all charts)", options=columns)
        y_axis = st.selectbox(
            "Select Y-axis column (Numeric recommended - Ignored for Histogram/Pie Chart)", 
            options=columns
        )

        # Chart types
        chart_type = st.selectbox(
            "Select Chart Type", 
            options=["Line Plot", "Bar Chart", "Scatter Plot", "Histogram", "Pie Chart", "Box Plot", "Area Plot"]
        )

        # Matplotlib Chart Generation
        if st.button("Generate Chart"):
            fig, ax = plt.subplots(figsize=(10, 5))

            try:
                if chart_type == "Line Plot":
                    ax.plot(df[x_axis], df[y_axis], marker="o", color="#2b5c8f")
                    ax.set_ylabel(y_axis)
                    ax.set_xlabel(x_axis)
                    
                elif chart_type == "Bar Chart":
                    ax.bar(df[x_axis], df[y_axis], color="#388e3c")
                    ax.set_ylabel(y_axis)
                    ax.set_xlabel(x_axis)
                    
                elif chart_type == "Scatter Plot":
                    ax.scatter(df[x_axis], df[y_axis], color="#e65100")
                    ax.set_ylabel(y_axis)
                    ax.set_xlabel(x_axis)
                    
                elif chart_type == "Histogram":
                    ax.hist(df[x_axis].dropna(), bins=20, color="#8e24aa", edgecolor="black")
                    ax.set_ylabel("Frequency / Count")
                    ax.set_xlabel(x_axis)
                    
                elif chart_type == "Pie Chart":
                    pie_data = df[x_axis].value_counts().head(10)
                    ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                    ax.axis('equal') 
                    
                elif chart_type == "Box Plot":
                    ax.boxplot(df[y_axis].dropna())
                    ax.set_xticklabels([y_axis])
                    ax.set_ylabel("Values")
                    
                elif chart_type == "Area Plot":
                    ax.fill_between(df[x_axis], df[y_axis], color="#00acc1", alpha=0.4)
                    ax.plot(df[x_axis], df[y_axis], color="#00acc1", alpha=0.8)
                    ax.set_ylabel(y_axis)
                    ax.set_xlabel(x_axis)

                # Chart title aur basic formatting
                ax.set_title(f"{chart_type}: {x_axis if chart_type in ['Histogram', 'Pie Chart'] else f'{y_axis} vs {x_axis}'}")
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Render Chart
                st.pyplot(fig)

            except Exception as plot_error:
                st.error(f"Chart nahi ban saka. Error: {plot_error}")

        # AI Q&A Section using Groq
        st.markdown("---")
        st.subheader("🤖 Ask Groq AI About Your Data")

        with st.expander("📊 View Automated Data Summary"):
            st.write(df.describe(include="all").fillna(""))

        user_question = st.text_input(
            "Apne data ya charts se related koi bhi question puchein:",
            placeholder="e.g., Is data me sabse highest value kis column me hai?",
        )

        if st.button("Ask AI") and user_question:
            if not api_available or client is None:
                st.warning("AI Client initialize nahi ho saka. Sidebar check karein.")
            else:
                with st.spinner("Groq AI data analyze kar raha hai..."):
                    try:
                        data_summary = df.describe(include="all").to_string()
                        data_sample = df.head(5).to_string()
                        columns_info = ", ".join(df.columns)

                        prompt = f"""
                        You are a data analyst assistant. Analyze the following tabular CSV data summary and answer the user's question clearly.
                        
                        Columns in data: {columns_info}
                        
                        Data Sample (first 5 rows):
                        {data_sample}
                        
                        Statistical Summary of Data:
                        {data_summary}
                        
                        User Question: {user_question}
                        
                        Please answer in a helpful, analytical, and easy-to-understand way based ONLY on the data provided above.
                        Use Urdu-English mix (Roman Urdu) so it's easy to read.
                        """

                        # Groq Request format
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": "You are a helpful data analyst."},
                                {"role": "user", "content": prompt}
                            ]
                        )

                        st.success("🤖 Groq AI Analysis:")
                        st.markdown(response.choices[0].message.content)

                    except Exception as ai_error:
                        st.error(f"AI se answer lene me error aaya: {ai_error}")

    except Exception as e:
        st.error(f"Error parsing file: {e}")

else:
    st.info("⬆️ Upar se CSV file upload karein taaki app shuru ho sake.")