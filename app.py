import streamlit as st
import pandas as pd
import requests
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- CONFIGURATION ---
MONDAY_API_URL = "https://api.monday.com/v2"
# We will use Streamlit secrets for security in the cloud
MONDAY_TOKEN = st.secrets["MONDAY_TOKEN"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DEALS_BOARD_ID = st.secrets["DEALS_BOARD_ID"]
WORK_ORDERS_BOARD_ID = st.secrets["WORK_ORDERS_BOARD_ID"]

# --- DATA RETRIEVAL & RESILIENCE ---
@st.cache_data(ttl=600) # Caches data for 10 minutes to save API calls
def fetch_monday_data(board_id):
    headers = {"Authorization": MONDAY_TOKEN, "API-Version": "2024-01"}
    
    # GraphQL query to get items and their column values
    query = f"""
    query {{
      boards(ids: {board_id}) {{
        items_page(limit: 500) {{
          items {{
            name
            column_values {{
              column {{ title }}
              text
            }}
          }}
        }}
      }}
    }}
    """
    
    response = requests.post(MONDAY_API_URL, json={'query': query}, headers=headers)
    data = response.json()
    
    # Parse the messy JSON into a clean list of dictionaries
    items = data['data']['boards'][0]['items_page']['items']
    parsed_data = []
    
    for item in items:
        row = {"Item Name": item['name']}
        for col in item['column_values']:
            # Normalizing data: replacing nulls with "Unknown" or 0 depending on context later
            val = col['text'] if col['text'] else "Unknown"
            row[col['column']['title']] = val
        parsed_data.append(row)
        
    df = pd.DataFrame(parsed_data)
    
    # Basic Data Resilience: Convert common string numbers to actual floats
    for col in df.columns:
        if df[col].astype(str).str.contains(r'^\$?\d+(,\d{3})*(\.\d+)?$').any():
            df[col] = df[col].replace(r'[\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='ignore')
            
    return df

# --- AGENT SETUP ---
def initialize_agent(selected_model="llama-3.3-70b-versatile"):
    # Load the data
    deals_df = fetch_monday_data(DEALS_BOARD_ID)
    work_orders_df = fetch_monday_data(WORK_ORDERS_BOARD_ID)
    
    # Initialize the Open Source LLM via Groq using the selected model variable
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=GROQ_API_KEY, 
        model_name=selected_model
    )
    
    # Enhanced prompt for better data resilience
    custom_prefix = """
    You are a Business Intelligence AI for Skylark Drones. Answer founder-level questions. 
    DataFrame df1 is Deals (Sales Pipeline). DataFrame df2 is Work Orders. 
    
    CRITICAL DATA CLEANING INSTRUCTIONS:
    Before performing any calculations on columns like 'Masked Deal value', 'Closure Probability', or 'Budget':
    1. Remove all string characters like '$', '%', 'TBD', 'Masked', and commas using pandas string manipulation.
    2. Convert the column to numeric (float), forcing errors to NaN.
    3. Fill missing or NaN numeric values with 0 or the median where appropriate.
    
    If data is heavily missing, state the caveats clearly, but ALWAYS attempt to provide an estimated numerical answer based on the cleanable data. Provide context and insights, not just raw numbers.
    """

    # Create the BI Agent giving it access to both dataframes
    agent = create_pandas_dataframe_agent(
        llm, 
        [deals_df, work_orders_df], 
        verbose=True,
        allow_dangerous_code=True, 
        prefix=custom_prefix
    )
    return agent

# --- FRONTEND UI ---
st.title("📊 Skylark Drones BI Executive Agent")
st.write("Ask me anything about the sales pipeline or work order operations.")

# Setup session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("E.g., How's our pipeline looking for the energy sector this quarter?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response with Fallback Logic
    with st.chat_message("assistant"):
        with st.spinner("Analyzing business data..."):
            output = ""
            try:
                # Attempt 1: Try the heavy reasoning model
                agent = initialize_agent(selected_model="llama-3.3-70b-versatile")
                response = agent.invoke(prompt)
                output = response["output"]
                
            except Exception as e:
                # If we hit a rate limit (429), trigger the transparent fallback
                if "429" in str(e) or "rate_limit_exceeded" in str(e):
                    st.warning("⚠️ API Rate Limit Reached for primary model. Automatically routing your query to the high-speed Llama 3.1 8B fallback model...")
                    try:
                        # Attempt 2: Fallback to the 8B model
                        fallback_agent = initialize_agent(selected_model="llama-3.1-8b-instant")
                        response = fallback_agent.invoke(prompt)
                        output = response["output"]
                    except Exception as fallback_e:
                         output = f"System is currently experiencing extreme traffic. Please try again in a few minutes. (Error: {fallback_e})"
                else:
                    # If it's a different error, show it
                    output = f"Data mapping error or API issue: {e}"
            
            # Display and save the final output
            if output:
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})

# Optional Leadership Update Button with Fallback Logic
if st.sidebar.button("Generate Leadership Update"):
    with st.spinner("Drafting executive summary..."):
        summary_output = ""
        prompt_text = "Generate a high-level executive summary of our current deal pipeline health and operational bottlenecks across all boards."
        
        try:
            agent = initialize_agent(selected_model="llama-3.3-70b-versatile")
            summary = agent.invoke(prompt_text)
            summary_output = summary["output"]
        except Exception as e:
            if "429" in str(e) or "rate_limit_exceeded" in str(e):
                st.sidebar.warning("⚠️ Primary model rate limit reached. Using fallback model...")
                try:
                    fallback_agent = initialize_agent(selected_model="llama-3.1-8b-instant")
                    summary = fallback_agent.invoke(prompt_text)
                    summary_output = summary["output"]
                except Exception as fallback_e:
                     summary_output = f"Error generating summary: {fallback_e}"
            else:
                 summary_output = f"Error generating summary: {e}"

        if summary_output:
            st.sidebar.markdown("### Executive Summary")
            st.sidebar.markdown(summary_output)
