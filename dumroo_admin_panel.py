import streamlit as st
import pandas as pd
import google.generativeai as genai
import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import os
from datetime import datetime, timedelta

# Set up page configuration for a better UI
st.set_page_config(page_title="Dumroo Admin Panel", layout="wide", initial_sidebar_state="expanded")

# Set up Gemini API key
gemini_api_key = st.secrets.get("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("Gemini API key not found. Please configure it in secrets.toml or environment variables.")
    st.stop()
os.environ["GEMINI_API_KEY"] = gemini_api_key
genai.configure(api_key=gemini_api_key)

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;
        color: #333333;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>input {
        border-radius: 5px;
        padding: 10px;
        background-color: #ffffff;
        color: #333333;
    }
    .stExpander {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        background-color: #ffffff;
    }
    .stForm {
        padding: 10px;
        background-color: #ffffff;
        border-radius: 5px;
    }
    .result-table {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Sample dataset
@st.cache_data
def load_data():
    data = {
        "student_name": ["Alice Smith", "Bob Johnson", "Charlie Brown", "Diana Wilson"],
        "grade": [8, 8, 9, 9],
        "class": ["A", "A", "B", "B"],
        "region": ["North", "North", "South", "South"],
        "homework_submitted": [True, False, True, False],
        "quiz_score": [85, 0, 90, 0],
        "quiz_date": ["2025-07-10", "2025-07-10", "2025-07-17", "2025-07-17"],
        "upcoming_quiz": ["2025-07-20", "2025-07-20", "2025-07-21", "2025-07-21"],
    }
    df = pd.DataFrame(data)
    df["quiz_date"] = pd.to_datetime(df["quiz_date"])
    df["upcoming_quiz"] = pd.to_datetime(df["upcoming_quiz"])
    return df

# Role-based access control
admin_scopes = {
    "grade_8_admin": {"grade": 8, "class": "A", "region": "North"},
    "grade_9_admin": {"grade": 9, "class": "B", "region": "South"},
}

def check_access(admin_role, query_grade, query_class, query_region):
    scope = admin_scopes.get(admin_role)
    if not scope:
        return False
    effective_grade = query_grade if query_grade is not None else scope["grade"]
    effective_class = query_class if query_class is not None else scope["class"]
    effective_region = query_region if query_region is not None else scope["region"]
    return (
        effective_grade == scope["grade"]
        and effective_class == scope["class"]
        and effective_region == scope["region"]
    )

# LangChain setup with Gemini
model = genai.GenerativeModel("gemini-2.0-flash")
prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    You are an AI assistant for an admin panel. Parse the following natural language query and extract:
    1. The type of data requested (e.g., homework, performance, quizzes)
    2. The grade (if mentioned, e.g., 8 or 9)
    3. The class (if mentioned, e.g., A or B)
    4. The region (if mentioned, e.g., North or South)
    5. The time period (if mentioned, e.g., last week, next week)
    
    Return ONLY a valid JSON object with keys: "data_type", "grade", "class", "region", "time_period", and NO additional text or comments.
    
    Query: {query}
    """,
)

# Create a RunnableLambda to handle Gemini response
def generate_gemini_response(prompt_value):
    query_text = prompt_value.text if hasattr(prompt_value, "text") else str(prompt_value)
    response = model.generate_content(query_text)
    return response.text

chain = prompt | RunnableLambda(generate_gemini_response)

# Process query and fetch data with fallback
def process_query(query, admin_role):
    try:
        parsed = chain.invoke({"query": query})
        st.write("Raw response:", parsed)
        cleaned_response = re.search(r"\{.*\}", parsed, re.DOTALL)
        if cleaned_response:
            parsed = cleaned_response.group(0)
        else:
            parsed = parsed.strip().replace("undefined", "")
        if not isinstance(parsed, str):
            parsed = str(parsed)
        parsed_data = json.loads(parsed)

        data_type = parsed_data.get("data_type")
        query_grade = parsed_data.get("grade")
        query_class = parsed_data.get("class")
        query_region = parsed_data.get("region")
        time_period = parsed_data.get("time_period")

        if not check_access(admin_role, query_grade, query_class, query_region):
            return "Access denied: You don't have permission to view this data."

        df = load_data()
        scope = admin_scopes[admin_role]
        filtered_df = df[
            (df["grade"] == (query_grade if query_grade is not None else scope["grade"]))
            & (df["class"] == (query_class if query_class is not None else scope["class"]))
            & (df["region"] == (query_region if query_region is not None else scope["region"]))
        ]

        if data_type == "homework":
            result = filtered_df[filtered_df["homework_submitted"] == False][
                ["student_name"]
            ]
            return result if not result.empty else pd.DataFrame({"Message": ["All homework submitted"]})
        elif data_type == "performance":
            if time_period == "last week":
                last_week = datetime.now() - timedelta(days=7)
                result = filtered_df[filtered_df["quiz_date"] >= last_week][
                    ["student_name", "quiz_score"]
                ]
                return result if not result.empty else pd.DataFrame({"Message": ["No performance data for last week"]})
        elif data_type == "quizzes":
            if time_period == "next week":
                next_week = datetime.now() + timedelta(days=7)
                result = filtered_df[filtered_df["upcoming_quiz"] <= next_week][
                    ["student_name", "upcoming_quiz"]
                ]
                return result if not result.empty else pd.DataFrame({"Message": ["No quizzes scheduled for next week"]})
        return pd.DataFrame({"Message": ["Unable to process query"]})

    except json.JSONDecodeError:
        return pd.DataFrame({"Error": [f"Invalid JSON response from AI. Raw response: {parsed}"]})
    except Exception as e:
        return pd.DataFrame({"Error": [f"Error processing query: {str(e)}"]})

# Streamlit UI with Enhanced Design
with st.sidebar:
    st.image("https://via.placeholder.com/150", use_container_width=True)
    st.title("Admin Dashboard")
    admin_role = st.selectbox("Select your admin role", ["grade_8_admin", "grade_9_admin"], key="role_select")

st.title("Dumroo Admin Panel")
st.write("Ask questions about student data in plain English")

with st.form(key="query_form"):
    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Enter your query", placeholder="e.g., Which students haven't submitted their homework yet?", key="query_input")
    with col2:
        submit_button = st.form_submit_button("Submit Query")
    if submit_button and query:
        result = process_query(query, admin_role)
        if "query_history" not in st.session_state:
            st.session_state.query_history = []
        st.session_state.query_history.append({"query": query, "result": result})
        st.success("Query processed!")
        st.write("Result:")
        st.dataframe(result, use_container_width=True, hide_index=True)

if st.session_state.get("query_history"):
    with st.expander("Query History"):
        for i, entry in enumerate(reversed(st.session_state.query_history[-5:])):
            st.write(f"**Query {len(st.session_state.query_history) - i}:** {entry['query']}")
            st.dataframe(entry["result"], use_container_width=True, hide_index=True)

st.subheader("Example Queries")
st.write("- Which students haven't submitted their homework yet?")
st.write("- Show me performance data for Grade 8 from last week")
st.write("- List all upcoming quizzes scheduled for next week")

with st.expander("Dataset Preview"):
    st.dataframe(load_data(), use_container_width=True)

# Bonus: Modular database connection (placeholder for real DB)
def connect_to_db():
    st.write("Database connection not implemented. Using cached data.")
    return load_data()

if st.checkbox("Use Database (Demo)"):
    df = connect_to_db()
    st.write("Data loaded from database (simulated):", df)
