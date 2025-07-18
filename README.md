# Dumroo Admin Panel

This repository contains my implementation of the Dumroo Admin Panel, developed as part of the Dumroo AI Developer Assignment. The project focuses on building an AI-powered interface for natural language querying and role-based access control for student data.

## Task Overview

The goal was to design a system that allows admins to ask questions in plain English and receive filtered results from a structured dataset. Access to data is restricted based on the admin’s assigned grade, class, or region.

### Core Requirements

1. **Dataset Creation**  
   I created a small dataset in dictionary format (converted to a Pandas DataFrame) containing student names, grades, classes, submission status, quiz scores, and dates.

2. **Natural Language Querying**  
   I used LangChain with Google Gemini (instead of OpenAI) to parse user queries and convert them into a structured format for data retrieval.

3. **Role-Based Access Control**  
   Admins can only access student data that falls within their assigned scope (e.g., grade 8, class A, region North).

4. **Query Examples Supported**  
   - Which students haven’t submitted their homework yet?  
   - Show me performance data for Grade 8 from last week  
   - List all upcoming quizzes scheduled for next week

5. **Bonus Features Implemented**  
   - A basic user interface built with Streamlit  
   - Modular code with a placeholder for future database integration

## How It Works

- The system reads a static dataset and loads it into a Pandas DataFrame.
- Natural language queries are parsed using the Gemini model via LangChain, which extracts key elements such as data type, grade, class, region, and time period.
- A function checks the user’s role and filters the data accordingly.
- Results are displayed in a clean table through the Streamlit UI.
- The interface allows users to select their role, enter queries, and view results, along with a history log and dataset preview.

## Tech Stack

- Python  
- Pandas  
- LangChain  
- Google Gemini (`google-generativeai`)  
- Streamlit  
- Datetime

## Limitations

- The current version handles only queries that match predefined patterns. New or complex queries may fail if not parsed correctly by Gemini.
- The dataset is static. Although the code includes a placeholder for a database connection, real-time updates are not yet supported.

## Setup Instructions

1. Clone the repository:

   git clone <your-repo-link>
   cd dumroo-admin-panel


2. Install the required dependencies:

   pip install -r requirements.txt


3. Run the Streamlit app:
 
   streamlit run dumroo_admin_panel.py
 

## Notes

- Instead of OpenAI, I used Gemini AI via the `google-generativeai` API for query processing.
