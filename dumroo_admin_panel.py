import streamlit as st
import pandas as pd
from langchain_community.llms import OpenAI  # Updated import
from langchain_core.prompts import PromptTemplate  # Updated import
from langchain_core.runnables import RunnableSequence  # Updated import
import os
import json
from datetime import datetime, timedelta

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


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
def check_access(admin_role, query_grade, query_class, query_region):
    admin_scopes = {
        "grade_8_admin": {"grade": 8, "class": "A", "region": "North"},
        "grade_9_admin": {"grade": 9, "class": "B", "region": "South"},
    }
    scope = admin_scopes.get(admin_role)
    if not scope:
        return False
    return (
        query_grade == scope["grade"]
        and query_class == scope["class"]
        and query_region == scope["region"]
    )


# LangChain setup with fallback
llm = OpenAI(temperature=0.3)
prompt = PromptTemplate(
    input_variables=["query"],
    template="""
    You are an AI assistant for an admin panel. Parse the following natural language query and extract:
    1. The type of data requested (e.g., homework, performance, quizzes)
    2. The grade (if mentioned, e.g., 8 or 9)
    3. The class (if mentioned, e.g., A or B)
    4. The region (if mentioned, e.g., North or South)
    5. The time period (if mentioned, e.g., last week, next week)
    
    Return the extracted information in JSON format.
    
    Query: {query}
    """,
)

# Create a RunnableSequence instead of LLMChain
chain = prompt | llm


# Process query and fetch data with fallback
def process_query(query, admin_role):
    try:
        # Parse query using LangChain
        parsed = chain.invoke({"query": query})  # Use invoke instead of run
        parsed_data = json.loads(parsed) if isinstance(parsed, str) else parsed

        # Extract parsed information
        data_type = parsed_data.get("data_type")
        query_grade = parsed_data.get("grade", 8)  # Default to grade 8
        query_class = parsed_data.get("class", "A")  # Default to class A
        query_region = parsed_data.get("region", "North")  # Default to North
        time_period = parsed_data.get("time_period")

        # Check access
        if not check_access(admin_role, query_grade, query_class, query_region):
            return "Access denied: You don't have permission to view this data."

        # Load data
        df = load_data()

        # Filter data based on query
        filtered_df = df[
            (df["grade"] == query_grade)
            & (df["class"] == query_class)
            & (df["region"] == query_region)
        ]

        if data_type == "homework":
            result = filtered_df[filtered_df["homework_submitted"] == False][
                ["student_name"]
            ]
            return result.to_string() if not result.empty else "All homework submitted."

        elif data_type == "performance":
            if time_period == "last week":
                last_week = datetime.now() - timedelta(days=7)
                result = filtered_df[filtered_df["quiz_date"] >= last_week][
                    ["student_name", "quiz_score"]
                ]
                return (
                    result.to_string()
                    if not result.empty
                    else "No performance data for last week."
                )

        elif data_type == "quizzes":
            if time_period == "next week":
                next_week = datetime.now() + timedelta(days=7)
                result = filtered_df[filtered_df["upcoming_quiz"] <= next_week][
                    ["student_name", "upcoming_quiz"]
                ]
                return (
                    result.to_string()
                    if not result.empty
                    else "No quizzes scheduled for next week."
                )

        return "Unable to process query."

    except Exception as e:
        if "429" in str(e) or "insufficient_quota" in str(e):
            return "Error: API quota exceeded. Please check your OpenAI plan or try again later."
        return f"Error processing query: {str(e)}"


# Streamlit UI
st.title("Dumroo Admin Panel")
st.write("Ask questions about student data in plain English")

# Admin role selection
admin_role = st.selectbox("Select your admin role", ["grade_8_admin", "grade_9_admin"])

# Query input
query = st.text_input(
    "Enter your query",
    placeholder="e.g., Which students haven't submitted their homework yet?",
)

# Process query when button is clicked
if st.button("Submit Query"):
    if query:
        result = process_query(query, admin_role)
        st.write("Result:")
        st.write(result)
    else:
        st.write("Please enter a query.")

# Example queries
st.subheader("Example Queries")
st.write("- Which students haven't submitted their homework yet?")
st.write("- Show me performance data for Grade 8 from last week")
st.write("- List all upcoming quizzes scheduled for next week")
