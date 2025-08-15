import os
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import re
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.azzgent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import AzureOpenAIEmbeddings
from sqlalchemy.dialects import registry

from io import BytesIO
import base64

import pandas as pd
import seaborn as sns
import signal
import time
import plotly.express as px
import plotly.graph_objects as go
#------------------------------------------------------------------------
import os
import streamlit as st
# Configure Google Cloud credentials using Streamlit secrets
def configure_environment():
    json_path = st.secrets.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not json_path or not os.path.exists(json_path):
        st.error("Invalid or missing GOOGLE_APPLICATION_CREDENTIALS path in secrets.")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path
#---------------------------------------------------------------------------

#------------------------------------------------------------------------
import streamlit as st
from langchain_openai import AzureChatOpenAI

# Initialize Azure OpenAI language model using Streamlit secrets
def initialize_azure_llm():
    api_key = st.secrets.get("AZURE_OPENAI_API_KEY")
    endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT")
    version = st.secrets.get("AZURE_OPENAI_API_VERSION")
    deployment_name = st.secrets.get("AZURE_OPENAI_DEPLOYMENT_NAME")
    temperature = float(st.secrets.get("TEMPERATURE", 0.0))
    top_p = float(st.secrets.get("TOP_P"))
    max_tokens = int(st.secrets.get("MAX_TOKENS"))
    if not all([api_key, endpoint, version, deployment_name]):
        st.error("Missing one or more Azure OpenAI environment variables.")
    return AzureChatOpenAI(
        openai_api_version=version,
        azure_deployment=deployment_name,
        azure_endpoint=endpoint,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
#---------------------------------------------------------------------------





#------------------------------------------------------------------------
import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings

# Initialize Azure OpenAI embeddings using Streamlit secrets
def initialize_azure_embeddings():
    api_key = st.secrets.get("AZURE_OPENAI_API_KEY")
    endpoint = st.secrets.get("AZURE_OPENAI_ENDPOINT")
    version = st.secrets.get("AZURE_OPENAI_API_VERSION")
    embedding_deployment = st.secrets.get("EMBEDDING_MODEL")
    if not all([api_key, endpoint, version, embedding_deployment]):
        st.error("Missing one or more Azure OpenAI environment variables.")
    return AzureOpenAIEmbeddings(
        azure_deployment=embedding_deployment,
        openai_api_version=version,
        azure_endpoint=endpoint,
        api_key=api_key
    )
#---------------------------------------------------------------------------






#------------------------------------------------------------------------
from sqlalchemy.dialects import registry
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.memory import ConversationBufferMemory

# Create a SQL agent with BigQuery using Streamlit secrets
def create_sql_agent_with_bigquery():
    registry.register("bigquery", "sqlalchemy_bigquery", "BigQueryDialect")
    db_url = f"bigquery://{st.secrets['GCP_PROJECT_ID']}/{st.secrets['BIGQUERY_DATASET']}"
    db = SQLDatabase.from_uri(db_url)
    llm = initialize_azure_llm()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input"
    )
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        agent_type="openai-tools",
        verbose=True,
        memory=memory,
        agent_executor_kwargs={"return_intermediate_steps": False}
    )
    return agent
#---------------------------------------------------------------------------

    
#------------------------------------------------------------------------
from google.cloud import bigquery

# Retrieve the schema of a specified BigQuery table
def get_table_schema(dataset_project, dataset, table):
    try:
        client = bigquery.Client()
        table_ref = f"{dataset_project}.{dataset}.{table}"
        schema = client.get_table(table_ref).schema
        return [(field.name, field.field_type, field.description) for field in schema]
    except Exception as e:
        return f"SCHEMA_ERROR: {str(e)}"
#---------------------------------------------------------------------------


#------------------------------------------------------------------------
import time
from google.cloud import bigquery

# Execute a BigQuery SQL query with retry logic
def run_bigquery_query(sql_query, max_retries=2, sleep_secs=2):
    for attempt in range(max_retries + 1):
        try:
            client = bigquery.Client()
            job = client.query(sql_query)
            return job.result().to_dataframe()
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries:
                time.sleep(sleep_secs)
                continue 
            else:
                return f"Query failed after {max_retries + 1} attempts: {error_msg}"
#---------------------------------------------------------------------------

    
# def is_sql_query(text):
#     """Enhanced SQL query detection"""
#     text_lower = text.strip().lower()
    
#     # Check if it starts with common SQL keywords
#     sql_starters = ['select', 'with', 'insert', 'update', 'delete', 'create', 'alter', 'drop']
#     starts_with_sql = any(text_lower.startswith(keyword) for keyword in sql_starters)
    
#     # Check if it contains SQL keywords and structure
#     has_sql_keywords = ('select' in text_lower and 'from' in text_lower) or \
#                       ('with' in text_lower and 'select' in text_lower)
    
#     # Check if it looks like conversational text
#     conversational_indicators = [
#         'here are', 'these are', 'the results', 'based on', 'according to',
#         'i found', 'the data shows', 'analysis shows', 'hello', 'hi', 'good morning',
#         'chart generation', 'summary', 'visualization'
#     ]
#     is_conversational = any(indicator in text_lower for indicator in conversational_indicators)
    
#     # If it's clearly conversational, it's not SQL
#     if is_conversational and not has_sql_keywords:
#         return False
    
#     # If it starts with SQL or has SQL structure, it's likely SQL
#     return starts_with_sql or has_sql_keywords

#------------------------------------------------------------------------

# Enhanced SQL query detection
def is_sql_query(text):
    text_lower = text.strip().lower()
    
    # Check if it starts with common SQL keywords
    sql_starters = ['select', 'with', 'insert', 'update', 'delete', 'create', 'alter', 'drop']
    starts_with_sql = any(text_lower.startswith(keyword) for keyword in sql_starters)
    
    # Check if it contains SQL keywords and structure
    has_sql_keywords = ('select' in text_lower and 'from' in text_lower) or \
                      ('with' in text_lower and 'select' in text_lower)
    
    # Check if it looks like conversational text
    conversational_indicators = [
        'here are', 'these are', 'the results', 'based on', 'according to',
        'i found', 'the data shows', 'analysis shows', 'hello', 'hi', 'good morning',
        'chart generation', 'summary', 'visualization'
    ]
    is_conversational = any(indicator in text_lower for indicator in conversational_indicators)
    
    # If it's clearly conversational, it's not SQL
    if is_conversational and not has_sql_keywords:
        return False
    
    # If it starts with SQL or has SQL structure, it's likely SQL
    return starts_with_sql or has_sql_keywords
#---------------------------------------------------------------------------

#------------------------------------------------------------------------
import re

# Generate SQL query using direct Azure OpenAI LLM without auto-execution
def generate_sql_query(llm, question, memory=None, dataset_project="genaiteam", dataset="infovision_internal", table="car_sales_data"):
    try:
        # Get conversation history if memory is provided and is a valid memory object
        history_context = ""
        if memory and hasattr(memory, 'chat_memory') and hasattr(memory.chat_memory, 'messages'):
            history = memory.chat_memory.messages
            if history:
                history_context = "\nRECENT CONVERSATION HISTORY:\n"
                for i in range(0, len(history), 2):
                    if i + 1 < len(history):
                        human_msg = history[i].content
                        ai_msg = history[i + 1].content
                        history_context += f"Q: {human_msg}\nA: {ai_msg[:100]}{'...' if len(ai_msg) > 100 else ''}\n\n"
                history_context += "Use this context to understand follow-up questions and references.\n"
        
        prompt = f'''You are a SQL expert and data analyst for BigQuery.
{history_context}

TABLE SCHEMA for {dataset_project}.{dataset}.{table}:
- `year` (INTEGER): The year the car was manufactured
- `make` (STRING): The brand or manufacturer of the car  
- `model` (STRING): The specific model of the car
- `trim` (STRING): The trim level of the car
- `body` (STRING): The body style (Coupe, Sedan, etc.)
- `transmission` (STRING): Type of transmission
- `vin` (STRING): Vehicle Identification Number
- `state` (STRING): U.S. state where sale took place (2-letter abbreviation)
- `condition` (INTEGER): Numeric representation of car condition
- `odometer` (INTEGER): Total distance traveled in miles
- `color` (STRING): Exterior color
- `interior` (STRING): Interior color
- `seller` (STRING): Name of selling entity
- `mmr` (INTEGER): Manheim Market Report value
- `sellingprice` (INTEGER): Final selling price
- `saledate` (STRING): Date and time when car was sold (format: "Wed Jan 14 2015 03:00:00 GMT-0800 (PST)")

IMPORTANT FIELD HANDLING:
- For date queries, use the `year` field instead of `saledate` since `saledate` is a complex string format
- The `year` field contains the manufacturing year, which is more reliable for year-based queries
- If you must use `saledate`, use PARSE_DATETIME() with proper format string, but prefer `year` field
- For date queries involving specific months/dates, use `saledate` field with proper parsing
- The `saledate` field contains strings in format: "Wed Jan 14 2015 03:00:00 GMT-0800 (PST)"
- NEVER use PARSE_DATETIME - it doesn't support timezone formats
- ALWAYS use SAFE.PARSE_TIMESTAMP (not PARSE_DATETIME) for saledate parsing
- ALWAYS wrap saledate parsing in SAFE function to handle any malformed dates
- To extract month: EXTRACT(MONTH FROM SAFE.PARSE_TIMESTAMP('%a %b %d %Y %H:%M:%S GMT%z', REGEXP_REPLACE(`saledate`, r' \([^)]+\)', '')))
- To extract year: EXTRACT(YEAR FROM SAFE.PARSE_TIMESTAMP('%a %b %d %Y %H:%M:%S GMT%z', REGEXP_REPLACE(`saledate`, r' \([^)]+\)', '')))
- Add WHERE clause to filter out NULL parsing results: WHERE SAFE.PARSE_TIMESTAMP('%a %b %d %Y %H:%M:%S GMT%z', REGEXP_REPLACE(`saledate`, r' \([^)]+\)', '')) IS NOT NULL
- For simple year-based queries, prefer using the `year` field instead of parsing saledate

CRITICAL: NEVER use PARSE_DATETIME with saledate - always use SAFE.PARSE_TIMESTAMP

DATETIME PARSING EXAMPLES:
- For December 2014 (CORRECT APPROACH): 
  SELECT SUM(`sellingprice`) 
  FROM `{dataset_project}.{dataset}.{table}` 
  WHERE EXTRACT(MONTH FROM SAFE.PARSE_TIMESTAMP('%a %b %d %Y %H:%M:%S GMT%z', REGEXP_REPLACE(`saledate`, r' \([^)]+\)', ''))) = 12 
  AND EXTRACT(YEAR FROM SAFE.PARSE_TIMESTAMP('%a %b %d %Y %H:%M:%S GMT%z', REGEXP_REPLACE(`saledate`, r' \([^)]+\)', ''))) = 2014
  AND SAFE.PARSE_TIMESTAMP('%a %b %d %Y %H:%M:%S GMT%z', REGEXP_REPLACE(`saledate`, r' \([^)]+\)', '')) IS NOT NULL

INSTRUCTIONS:
1. If the user greets you (e.g., "Hi", "Hello", "Good morning"), respond with a friendly greeting .
2. If the user asks a data question, generate ONLY the SQL query - no explanations, no formatting.
3. For year-based queries, use the `year` field, not `saledate`
4. For complex questions like "which car sold most each year":
   - Use CTEs with window functions
   - Use ROW_NUMBER() OVER (PARTITION BY year ORDER BY COUNT(*) DESC) 
   - Filter NULL values: WHERE make IS NOT NULL AND model IS NOT NULL
5. Always use backticks around column names
6. Use full table name: `{dataset_project}.{dataset}.{table}`

EXAMPLE PATTERNS:
- "Which car sold most each year" → Use CTE with COUNT(*) grouped by year/make/model, then ROW_NUMBER() to get top per year
- "Top selling cars" → GROUP BY make, model with COUNT(*) ORDER BY count DESC
- "Sales by state" → GROUP BY state with COUNT(*)
- "Sales for 2025" → SELECT COUNT(*) FROM table WHERE `year` = 2025

Question: {question}

Response (SQL query only if data question, or greeting if greeting or if not related)'''

        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Clean up any code block markers
        cleaned_response = re.sub(r"^```sql\s*|```$", "", response_text, flags=re.MULTILINE).strip()
        
        return cleaned_response
        
    except Exception as e:
        print(f"\n SQL Generation Failed!")
        print(f"Error Details: {str(e)}")
        
        error_str = str(e).lower()
        
        print(f"\n Possible Reasons:")
        
        if "timeout" in error_str or "connection" in error_str:
            print("   • Network connectivity issues with Azure OpenAI")
            print("   • Request timeout - try a simpler question")
            
        elif "authentication" in error_str or "unauthorized" in error_str or "401" in error_str:
            print("   • Invalid Azure OpenAI API key")
            print("   • API key expired or doesn't have proper permissions")
            print("   • Check your config.py file for correct credentials")
            
        elif "quota" in error_str or "rate limit" in error_str or "429" in error_str:
            print("   • Azure OpenAI API quota exceeded")
            print("   • Too many requests - wait a moment and try again")
            
        elif "model" in error_str or "deployment" in error_str:
            print("   • Invalid deployment name in configuration")
            print("   • Model deployment not found or not available")
            print("   • Check your AZURE_OPENAI_DEPLOYMENT_NAME in config.py")
            
        else:
            print("   • Unknown error occurred")
            print("   • Try rephrasing your question")
            print("   • Check your Azure OpenAI configuration")
            
        print(f"\n Suggestions:")
        print("   • Make sure your question is clear and specific")
        print("   • Try asking about data that exists in your tables")
        print("   • Use simpler language in your question")
        print("   • Check if your Azure OpenAI service is running")
        
        return None

# Retry-based wrapper around generate_sql_query to increase reliability
def robust_generate_sql_query(llm, question, memory=None, dataset_project="genaiteam", dataset="infovision_internal", table="car_sales_data", max_retries=2):
    for attempt in range(max_retries + 1):
        sql_query = generate_sql_query(llm, question, memory, dataset_project, dataset, table)
        if sql_query and sql_query.strip().lower().startswith("select") and "from" in sql_query.lower():
            return sql_query
    return sql_query  
#------------------------------------------------------------------------



# def explain_query_failure(question, error_message):
#     """Provide detailed explanation for query failures"""
#     print(f"\n Query Generation Failed for: '{question}'")
#     print(f"Error: {error_message}")
    
#     print(f"\n Analysis:")
    
#     if any(word in question.lower() for word in ['what', 'how', 'show', 'get', 'find', 'list']):
#         print("   • Your question appears to be in natural language ✓")
#     else:
#         print("   • Your question might not be clear enough")
        
#     if len(question.split()) < 3:
#         print("   • Question is very short - try being more specific")
        
#     if not any(word in question.lower() for word in ['car', 'sales', 'data', 'table', 'customer', 'birth']):
#         print("   • Question doesn't reference available data")
#         print("   • Available tables: car_sales_data, customer_data3, student_data")
        

#------------------------------------------------------------------------
from google.cloud import bigquery

# Retrieve schema details and a sample of rows from a BigQuery table
def get_table_schema_and_sample(project_id, dataset_id, table_id, sample_rows=5):
    client = bigquery.Client()
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    # Fetch table schema
    schema = client.get_table(table_ref).schema
    columns = []
    for field in schema:
        columns.append({
            "Field Name": field.name,
            "Type": field.field_type,
            "Description": field.description if field.description else ""
        })
    
    # Fetch sample rows
    sample_query = f"SELECT * FROM `{table_ref}` LIMIT {sample_rows}"
    sample_df = client.query(sample_query).result().to_dataframe()
    
    return columns, sample_df
#------------------------------------------------------------------------


#------------------------------------------------------------------------
from google.cloud import bigquery

# Execute a SQL query on BigQuery and return the results as a DataFrame
def execute_sql_query(query):
    try:
        client = bigquery.Client()
        if not client:
            return "Query failed: Failed to setup BigQuery client"
            
        query_job = client.query(query)
        df = query_job.result().to_dataframe()
        return df
    except Exception as e:
        return f"Query failed: {str(e)}"
#------------------------------------------------------------------------


#------------------------------------------------------------------------
import pandas as pd

# Generate a conversational text summary of the given DataFrame using the LLM
def generate_data_summary(df, user_question, sql_query, llm):
    """Generate a conversational text summary of the DataFrame results"""
    try:
        if not isinstance(df, pd.DataFrame):
            return "No data available to summarize."
        
        if df.empty:
            return "The query returned no results."
        
        # Convert DataFrame to string for analysis (sample for large datasets)
        if len(df) > 10:
            df_sample = df.sample(n=10, random_state=42)
            df_str = df_sample.to_string(index=False)
            analysis_note = f"analyzing patterns across your {len(df):,} records"
        else:
            df_str = df.to_string(index=False)
            analysis_note = f"analyzing your {len(df)} records"
        
        analysis_prompt = f"""
        You are a data analyst. A user asked: "{user_question}"
        
        The system generated this SQL query: {sql_query}
        
        Here is the query result data for analysis:
        {df_str}
        
        You are {analysis_note}. Please provide a conversational summary focusing on key insights and patterns. Include:
        1. Main trends and patterns you observe in the data
        2. Important numbers, findings, or standout values
        3. Direct answer to the user's question based on the SQL query and results
        4. Keep it concise but informative (2-4 sentences)
        
        IMPORTANT: Write in plain text format only. No markdown, bold text, headers, bullet points, or special characters. Write as if you're having a normal conversation - just regular sentences with proper punctuation.
        
        Focus on what the data reveals rather than analysis methodology. Write in a natural, conversational tone.
        Do NOT mention like the based on SQL, queries, or code. Write as if you are talking to the user about their data in plain language. Use a conversational, friendly style.
        """
        
        response = llm.invoke(analysis_prompt)
        return response.content.strip()
        
    except Exception as e:
        return f"Summary generation failed: {str(e)}"
#------------------------------------------------------------------------

#------------------------------------------------------------------------
import streamlit as st
import pandas as pd

# Generate a summary for the most recent query result stored in session state
def generate_summary_on_demand(llm):
    """Generate summary for the most recent query result"""
    try:
        last_result = None
        for turn in reversed(st.session_state.chat_history):
            if turn.get("source") == "query_result" and isinstance(turn.get("response"), pd.DataFrame):
                last_result = turn
                break
        
        if not last_result:
            st.session_state.chat_history.append({
                "question": "Generate Summary",
                "sql": "N/A", 
                "response": "No recent query results found to summarize."
            })
            return
        
        # Generate summary
        df = last_result["response"]
        question = last_result["question"]
        sql_query = last_result.get("sql", "N/A")
        summary = generate_data_summary(df, question, sql_query, llm)
        
        # Add to chat history
        st.session_state.chat_history.append({
            "question": "Data Summary",
            "sql": "N/A",
            "response": summary,
            "source": "summary"
        })
        
    except Exception as e:
        st.session_state.chat_history.append({
            "question": "Generate Summary",
            "sql": "N/A",
            "response": f"Summary generation failed: {str(e)}"
        })
#------------------------------------------------------------------------



#------------------------------------------------------------------------
import plotly.express as px
import plotly.graph_objects as go

# Use LLM to generate valid Plotly chart code (compatible with Streamlit)
def generate_chart_code_from_response_with_type(response_text: str, llm, chart_type: str) -> str:
    """Generate Python Plotly code using LLM (Streamlit-compatible)"""

    if chart_type != "Auto":
        prompt = f"""
You are a data visualization expert. The user requested a {chart_type.upper()} chart.

Data Snapshot:
{response_text}

Instructions:
1. If data is suitable for a {chart_type.lower()} chart:
   - Generate valid Python code using Plotly  with labels in that (prefer plotly.express)
   - Use the variable 'df' for the DataFrame
   - Return only Python code (no HTML, no markdown, no explanations)
   - Do NOT use fig.show()
   - Just return the fig object – it will be displayed using Streamlit's st.plotly_chart(fig)

2. If not suitable, return a go.Figure() with a clear annotation using fig.add_annotation()

Rules:
- Never use fig.to_html(), fig.to_json(), or fig.show()
- No backticks, markdown, or extra text
- Only return valid, executable Python code
"""
    else:
        prompt = f"""
You are a data visualization expert. Choose the best chart based on the dataset below.

Data Snapshot:
{response_text}

Instructions:
1. Select the most appropriate chart type: bar, line, pie, scatter, bubble, area
2. Use Plotly (prefer plotly.express) and the variable 'df'
3. Return valid Python code inlude clear labels for the chart 
4. Do NOT call fig.show()
5. If charting is not possible, return go.Figure() with an annotation using fig.add_annotation()

Rules:
- No HTML, markdown, or explanations
- Only return valid, executable Python code
- Streamlit will render the chart using st.plotly_chart(fig)
"""

    result = llm.invoke(prompt)
    return result.content.strip()
#------------------------------------------------------------------------



#------------------------------------------------------------------------
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Execute the given Python Plotly code string using a DataFrame and return a Plotly figure
def execute_chart_code(code: str, df: pd.DataFrame):
    """Execute Plotly code and return a figure object or explanation string"""
    code = code.strip().replace("```python", "").replace("```", "")

    if not code or df is None or not isinstance(df, pd.DataFrame):
        return "Chart generation failed: Invalid code or DataFrame."

    try:
        local_vars = {"df": df, "px": px, "go": go}
        exec(code, {}, local_vars)
        fig = local_vars.get("fig")
        if isinstance(fig, go.Figure):
            return fig
        else:
            return "Chart generation failed: No Plotly figure object returned."
    except Exception as e:
        return f"Chart generation failed: {str(e)}"
#------------------------------------------------------------------------





#------------------------------------------------------------------------
import os
import time
import signal
import pandas as pd
import streamlit as st

# Streamlit UI function for interacting with the BigQuery chatbot
def interactive_query_ui(agent, dataset_project, dataset, table):
    st.title("BigQuery Chatbot")

    if "show_shutdown_message" not in st.session_state:
        st.session_state.show_shutdown_message = False

    user_input = st.chat_input("Ask your data question")
    
    if user_input:
        sql_query = robust_generate_sql_query(
            llm_model, user_input,
            memory=agent.memory,
            dataset_project=dataset_project,
            dataset=dataset,
            table=table
        )

        if sql_query:
            if sql_query.startswith("Error generating SQL query"):
                st.session_state.chat_history.append({
                    "question": user_input,
                    "sql": "N/A",
                    "response": sql_query,
                    "error": True
                })
                return
            
            if not sql_query.strip().lower().startswith("select") and "from" not in sql_query.lower():
                st.session_state.chat_history.append({
                    "question": user_input,
                    "sql": "N/A",
                    "response": sql_query.strip()
                })
                return
            
            df = run_bigquery_query(sql_query)

            if isinstance(df, str) and df.startswith("Query"):
                st.session_state.chat_history.append({
                    "question": user_input,
                    "sql": sql_query,
                    "response": df,
                    "error": True
                })

            elif df is not None:
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df_cleaned = df.drop_duplicates()
                    st.session_state.chat_history.append({
                        "question": user_input,
                        "sql": sql_query,
                        "response": df_cleaned,
                        "source": "query_result"
                    })
                else:
                    st.session_state.chat_history.append({
                        "question": user_input,
                        "sql": sql_query,
                        "response": df,
                        "source": "query_result"
                    })
        else:
            st.session_state.chat_history.append({
                "question": user_input,
                "sql": "N/A",
                "response": "Failed to generate SQL query. Please try rephrasing your question.",
                "error": True
            })

    # Shutdown mechanism
    if st.session_state.show_shutdown_message:
        st.empty()
        st.markdown("<h2 style='text-align: center; color: green;'>App successfully terminated.</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2em;'>Close this tab to complete.</p>", unsafe_allow_html=True)
        st.warning("Please note: Due to browser security, this tab might not close automatically. You may need to close it manually.")
        time.sleep(1)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception as e:
            print(f"Error attempting to shut down the Python process: {e}")
    else:
        if st.sidebar.button("STOP APP"):
            st.session_state.show_shutdown_message = True
            st.rerun()
#------------------------------------------------------------------------






#------------------------------------------------------------------------
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Generate a Plotly chart using LLM-generated Python code and add to chat history
def generate_chart_and_add_to_chat(llm, chart_type="Auto"):
    """Generate a Plotly chart using LLM-generated Python code and add to chat history"""

    if not st.session_state.get("chat_history") or len(st.session_state.chat_history) == 0:
        st.session_state.chat_history.append({
            "question": "Generate Chart",
            "response": "No previous response available to generate a chart.",
            "error": True
        })
        return

    # Get the latest response with a DataFrame
    last_turn = None
    for turn in reversed(st.session_state.chat_history):
        if turn.get("source") == "query_result" and isinstance(turn.get("response"), pd.DataFrame):
            last_turn = turn
            break

    if not last_turn:
        st.session_state.chat_history.append({
            "question": "Generate Chart",
            "response": "No valid DataFrame found in recent responses to generate a chart.",
            "error": True
        })
        return

    df = last_turn.get("response", "")
    if not isinstance(df, pd.DataFrame):
        st.session_state.chat_history.append({
            "question": "Generate Chart",
            "response": "Response is not in expected format to generate a chart.",
            "error": True
        })
        return

    if len(df) <= 1:
        explanation = (
            "Chart generation is not possible because the query result contains only a single row. "
            "Charts require multiple data points to visualize patterns, trends, or distributions. "
            "Try asking a broader question or one that returns more data."
        )
        st.session_state.chat_history.append({
            "question": f"Generate Chart ({chart_type})",
            "response": explanation,
            "error": False
        })
        return

    # Sample the data for LLM input
    if len(df) > 10:
        df_sample = df.sample(n=10, random_state=42)
        df_str = df_sample.to_string(index=False)
        data_note = f"Data patterns from your {len(df):,} records:\n{df_str}"
    else:
        df_str = df.to_string(index=False)
        data_note = f"Your complete dataset ({len(df)} records):\n{df_str}"

    # Store current DataFrame
    st.session_state.current_df = df

    # Generate chart code using LLM
    chart_response = generate_chart_code_from_response_with_type(data_note, llm, chart_type)

    # Try executing the generated code
    chart_result = execute_chart_code(chart_response, df)

    # Handle result types
    if isinstance(chart_result, go.Figure):
        st.session_state.chat_history.append({
            "question": f"Generate Chart ({chart_type})",
            "response": f"Chart generated using '{chart_type}' chart type.",
            "chart_figure": chart_result,
            "code": chart_response
        })
    elif isinstance(chart_result, str):
        st.session_state.chat_history.append({
            "question": f"Generate Chart ({chart_type})",
            "response": chart_result.strip(),
            "error": chart_result.strip().startswith("Chart generation failed")
        })
    else:
        st.session_state.chat_history.append({
            "question": f"Generate Chart ({chart_type})",
            "response": "Unknown error occurred while generating the chart.",
            "error": True
        })
#------------------------------------------------------------------------

#------------------------------------------------------------------------
import pandas as pd
import streamlit as st

# Run SQL generation and summarization for multiple questions from a CSV file
def run_bulk_question_tests_from_csv(file_path, llm, dataset_project, dataset, table):
    try:
        df_questions = pd.read_csv(file_path)
        if "Question" not in df_questions.columns:
            st.error("CSV must contain a column named 'Question'")
            return

        for idx, row in df_questions.iterrows():
            question = row["Question"]

            # Attempt to generate SQL query
            sql_query = robust_generate_sql_query(
                llm,
                question,
                memory=None,
                dataset_project=dataset_project,
                dataset=dataset,
                table=table
            )

            # Check for invalid or non-SQL output
            if not sql_query or not sql_query.strip().lower().startswith("select"):
                st.session_state.chat_history.append({
                    "question": question,
                    "sql": sql_query,
                    "response": "Invalid or non-SQL response.",
                    "summary": "",
                    "source": "query_result",
                    "error": True,
                    "bulk": True
                })
                continue

            # Run the query
            df = run_bigquery_query(sql_query)

            # If query succeeded
            if isinstance(df, pd.DataFrame):
                if not df.empty:
                    summary = generate_data_summary(df, question, sql_query, llm)
                else:
                    summary = "Query returned no data."

                st.session_state.chat_history.append({
                    "question": question,
                    "sql": sql_query,
                    "response": df,
                    "summary": summary,
                    "source": "query_result",
                    "error": False,
                    "bulk": True
                })

            # If query failed
            else:
                st.session_state.chat_history.append({
                    "question": question,
                    "sql": sql_query,
                    "response": str(df),
                    "summary": "",
                    "source": "query_result",
                    "error": True,
                    "bulk": True
                })

    except Exception as e:
        st.error(f"Bulk test failed: {str(e)}")
#------------------------------------------------------------------------

#------------------------------------------------------------------------
from google.cloud import bigquery

# Retrieve all table names from a given BigQuery dataset
def get_all_tables(project_id, dataset_id):
    client = bigquery.Client()
    tables = client.list_tables(f"{project_id}.{dataset_id}")
    return [table.table_id for table in tables]
#------------------------------------------------------------------------


#------------------------------------------------------------------------
from google.cloud import bigquery

# Generate a summary for any BigQuery table: structure, nulls, top values, stats, etc.
def generate_generic_table_summary(project, dataset, table):
    """
    Generate a generic, human-readable summary for any BigQuery table.
    Shows row/column count, column names/types, missing %, top values, numeric stats, date ranges, and more.
    """
    client = bigquery.Client()
    table_ref = f"{project}.{dataset}.{table}"
    
    try:
        schema = client.get_table(table_ref).schema
    except Exception as e:
        return f"Error fetching schema: {e}"

    summary = []
    summary.append(f"Table: {table}")

    # Row count
    try:
        row_count_query = f"SELECT COUNT(*) as total FROM `{table_ref}`"
        row_count = client.query(row_count_query).result().to_dataframe()['total'][0]
        summary.append(f"Number of rows: {row_count:,}")
    except Exception as e:
        summary.append(f"Number of rows: ? (error: {e})")
        row_count = None

    summary.append(f"Number of columns: {len(schema)}")
    summary.append("")

    for field in schema:
        col = field.name
        field_type = field.field_type
        summary.append(f"{col} ({field_type}):")

        if row_count is not None:
            try:
                if field_type in ["RECORD", "STRUCT", "ARRAY"]:
                    summary.append("  Missing: (null check skipped for STRUCT/ARRAY)")
                else:
                    non_null_query = f"SELECT COUNT({col}) as non_nulls FROM `{table_ref}`"
                    result = client.query(non_null_query).result().to_dataframe()
                    non_nulls = result['non_nulls'].iloc[0]
                    nulls = row_count - non_nulls
                    null_perc = (100 * nulls / row_count) if row_count > 0 else 0
                    summary.append(f"  Missing: {nulls:,} ({null_perc:.1f}%)")
            except Exception:
                summary.append("  Missing: (unable to calculate)")
        else:
            summary.append("  Missing: ? (row count unknown)")

        if field_type in ["STRING", "BYTES"]:
            try:
                top_query = f"""
                SELECT `{col}`, COUNT(*) as cnt 
                FROM `{table_ref}` 
                WHERE `{col}` IS NOT NULL
                GROUP BY `{col}` 
                ORDER BY cnt DESC 
                LIMIT 5
                """
                top_df = client.query(top_query).result().to_dataframe()
                if not top_df.empty:
                    top_values = ', '.join([f"{str(row[col])} ({row['cnt']})" for _, row in top_df.iterrows()])
                    summary.append(f"  Top values: {top_values}")

                unique_query = f"SELECT COUNT(DISTINCT `{col}`) as uniq FROM `{table_ref}`"
                unique_cnt = client.query(unique_query).result().to_dataframe()['uniq'][0]
                summary.append(f"  Unique values: {unique_cnt:,}")
            except Exception as e:
                summary.append(f"  (Could not get text stats: {e})")

        elif field_type in ["INTEGER", "FLOAT", "NUMERIC", "BIGNUMERIC"]:
            try:
                stats_query = f"""
                SELECT 
                    MIN(`{col}`) as min_val,
                    MAX(`{col}`) as max_val,
                    AVG(`{col}`) as avg_val,
                    APPROX_QUANTILES(`{col}`, 4) as quantiles
                FROM `{table_ref}` 
                WHERE `{col}` IS NOT NULL
                """
                stats = client.query(stats_query).result().to_dataframe().iloc[0]
                summary.append(f"  Min: {stats['min_val']}, Max: {stats['max_val']}, Avg: {stats['avg_val']:.2f}")
                q = stats['quantiles']
                if isinstance(q, (list, tuple)) and len(q) >= 5:
                    summary.append(f"  Quartiles: {q}")
                else:
                    summary.append(f"  Quartiles: {q}")
            except Exception as e:
                summary.append(f"  (Could not get numeric stats: {e})")

        elif "DATE" in field_type or "TIME" in field_type or "TIMESTAMP" in field_type:
            try:
                date_query = f"""
                SELECT 
                    MIN(`{col}`) as min_date,
                    MAX(`{col}`) as max_date
                FROM `{table_ref}` 
                WHERE `{col}` IS NOT NULL
                """
                minmax = client.query(date_query).result().to_dataframe().iloc[0]
                summary.append(f"  Range: {minmax['min_date']} to {minmax['max_date']}")
            except Exception as e:
                summary.append(f"  (Could not get date stats: {e})")

        elif field_type == "BOOLEAN":
            try:
                bool_query = f"""
                SELECT 
                    `{col}`, COUNT(*) as cnt
                FROM `{table_ref}` 
                WHERE `{col}` IS NOT NULL
                GROUP BY `{col}`
                ORDER BY cnt DESC
                """
                bool_df = client.query(bool_query).result().to_dataframe()
                if not bool_df.empty:
                    bool_values = ', '.join([f"{row[col]} ({row['cnt']})" for _, row in bool_df.iterrows()])
                    summary.append(f"  Values: {bool_values}")
            except Exception as e:
                summary.append(f"  (Could not get boolean stats: {e})")

        else:
            try:
                sample_query = f"""
                SELECT `{col}` 
                FROM `{table_ref}` 
                WHERE `{col}` IS NOT NULL 
                LIMIT 3
                """
                sample_df = client.query(sample_query).result().to_dataframe()
                if not sample_df.empty:
                    sample_values = [str(val) for val in sample_df[col].tolist()]
                    summary.append(f"  Sample: {sample_values}")
            except Exception as e:
                summary.append(f"  (Could not get sample: {e})")

        summary.append("")

    return "\n".join(summary)
#------------------------------------------------------------------------

#------------------------------------------------------------------------
import streamlit as st
import pandas as pd
from plotly import graph_objects as go


# Embeddings and LLM Initialization
llm_provider = "azure"
embeddings = initialize_azure_embeddings()
llm_model = initialize_azure_llm()

# Streamlit Application Entry Point
def main():
    configure_environment()
    agent = create_sql_agent_with_bigquery()
    llm = llm_model

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    dataset_project = st.secrets["GCP_PROJECT_ID"]
    dataset = st.secrets["BIGQUERY_DATASET"]
    table = st.secrets["BIGQUERY_TABLE"]

    # Chat UI
    interactive_query_ui(agent, dataset_project, dataset, table)

    # Sidebar Tools
    chart_type = st.sidebar.selectbox("Chart Type", ["Auto", "Line", "Bar", "Pie", "Scatter", "Bubble", "Area"], key="chart_type")
    st.sidebar.header("Visualization Tools")

    if "chart_type" not in st.session_state:
        st.session_state.chart_type = "Auto"

 

    if st.sidebar.button("Generate Chart from Answer", key="generate_chart_button"):
        generate_chart_and_add_to_chat(llm_model, st.session_state.chart_type)

    if st.sidebar.button("Run Bulk Tests from CSV"):
        run_bulk_question_tests_from_csv(
            file_path="./Car_Sales_All_Questions.csv",
            llm=llm_model,
            dataset_project=dataset_project,
            dataset=dataset,
            table=table
        )

    # Table Explorer
    with st.sidebar.expander("Explore BigQuery Tables", expanded=False):
        try:
            table_list = get_all_tables(dataset_project, dataset)
            selected_table = st.selectbox("Choose a table to explore", table_list, key="bq_table_explorer")
            if selected_table:
                columns, sample_df = get_table_schema_and_sample(dataset_project, dataset, selected_table)
                st.write("**Table Columns:**")
                st.dataframe(pd.DataFrame(columns, columns=["Field Name", "Type", "Description"]))
                st.write("**Sample Data:**")
                st.dataframe(sample_df)

                if st.button("Generate Table Summary", key="table_summary_btn"):
                    try:
                        summary = generate_generic_table_summary(dataset_project, dataset, selected_table)
                        if not summary:
                            st.warning("No summary could be generated. Table may be empty or access issue.")
                        else:
                            st.write("**Table Summary:**")
                            st.text(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
        except Exception as e:
            st.info(f"Could not fetch tables: {e}")

    # Render chat history
    for i, turn in enumerate(st.session_state.chat_history):
        st.chat_message("user").write(turn['question'])

        if turn.get("sql") and turn["sql"] != "N/A":
            st.chat_message("assistant").markdown("**Generated SQL:**")
            st.chat_message("assistant").code(turn["sql"], language="sql")

        if isinstance(turn["response"], pd.DataFrame):
            st.chat_message("assistant").markdown("**Query Result:**")
            st.chat_message("assistant").dataframe(turn["response"])

            # Show summary button only for the most recent DataFrame
            is_most_recent_dataframe = not any(
                j > i and turn_j.get("source") == "query_result" and isinstance(turn_j.get("response"), pd.DataFrame)
                for j, turn_j in enumerate(st.session_state.chat_history)
            )

            if is_most_recent_dataframe:
                if st.button("Generate Data Summary", key=f"summary_btn_{i}"):
                    generate_summary_on_demand(llm_model)
        else:
            st.chat_message("assistant").markdown(str(turn["response"]))

        if turn.get("summary"):
            st.chat_message("assistant").markdown("**Summary:**")
            st.chat_message("assistant").markdown(str(turn["summary"]))

        # Chart generation for query results
        if isinstance(turn["response"], pd.DataFrame) and not turn["response"].empty:
            chart_type_key = f"chart_type_select_{i}"
            chart_btn_key = f"bulk_chart_btn_{i}"

            chart_type = st.selectbox(
                "Chart Type",
                ["Auto", "Line", "Bar", "Pie", "Scatter", "Bubble", "Area"],
                key=chart_type_key
            )

            if st.button("Generate Chart", key=chart_btn_key):
                df = turn["response"]
                df_sample = df.sample(n=10, random_state=42) if len(df) > 10 else df
                chart_input = df_sample.to_string(index=False)
                data_note = f"Here is a sample of the data:\n{chart_input}"

                chart_code = generate_chart_code_from_response_with_type(data_note, llm_model, chart_type=chart_type)
                chart_result = execute_chart_code(chart_code, df)

                if isinstance(chart_result, go.Figure):
                    turn["chart_figure"] = chart_result
                else:
                    st.warning(chart_result)
                    turn["chart_figure"] = None

            if turn.get("chart_figure") is not None:
                st.plotly_chart(turn["chart_figure"], use_container_width=True, key=f"plotly_chart_{i}")

      #------------------------------------------------------------------------



if __name__ == "__main__":
    main()