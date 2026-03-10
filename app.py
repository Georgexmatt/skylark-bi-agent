import streamlit as st
import pandas as pd
import requests
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# --- CONFIGURATION ---
MONDAY_API_URL = "https://api.monday.com/v2"
MONDAY_TOKEN = st.secrets["MONDAY_TOKEN"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
DEALS_BOARD_ID = st.secrets["DEALS_BOARD_ID"]
WORK_ORDERS_BOARD_ID = st.secrets["WORK_ORDERS_BOARD_ID"]

# --- DATA RETRIEVAL ---
@st.cache_data(ttl=600)
def fetch_monday_data(board_id):
    headers = {"Authorization": MONDAY_TOKEN, "API-Version": "2024-01"}
    query = f"""
    query {{
      boards(ids: {board_id}) {{
        items_page(limit: 200) {{ 
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
    items = data['data']['boards'][0]['items_page']['items']
    parsed_data = []
    for item in items:
        row = {"Item Name": item['name']}
        for col in item['column_values']:
            val = col['text'] if col['text'] else "Unknown"
            row[col['column']['title']] = val
        parsed_data.append(row)
    return pd.DataFrame(parsed_data)

# --- AGENT SETUP ---
def initialize_agent(selected_model="llama-3.3-70b-versatile"):
    deals_df = fetch_monday_data(DEALS_BOARD_ID)
    work_orders_df = fetch_monday_data(WORK_ORDERS_BOARD_ID)
    
    essential_deals = ['Item Name', 'Masked Deal value', 'Closure Probability', 'Deal Stage', 'Sector']
    essential_work = ['Item Name', 'Status', 'Priority', 'Timeline']
    
    deals_slim = deals_df[[c for c in essential_deals if c in deals_df.columns]]
    work_slim = work_orders_df[[c for c in essential_work if c in work_orders_df.columns]]
    
    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=selected_model)
    
    custom_prefix = """
    You are a BI Agent for Skylark Drones. DataFrame df1 is Deals, df2 is Work Orders.
    Before math, clean columns (remove $, %, commas) and convert to numeric.
    Always provide an estimated numerical answer based on cleanable data.
    """

    return create_pandas_dataframe_agent(
        llm, [deals_slim, work_slim], verbose=True, allow_dangerous_code=True, 
        prefix=custom_prefix, max_iterations=15, handle_parsing_errors=True
    )

# --- UI FRONTEND ---
st.set_page_config(page_title="Skylark BI Agent", page_icon="📊")
st.title("📊 Skylark Drones BI Executive Agent")

# Sidebar for Quick Actions
st.sidebar.title("💡 Suggested Questions")
st.sidebar.write("Click a question below to test the agent:")

# We use session state to handle button clicks as prompts
if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# List of smart questions
q1 = "What is the total value of our 'Won' deals in the Energy sector?"
q2 = "How many work orders are currently marked as 'Stuck'?"
q3 = "Calculate the weighted average deal value for the energy sector considering closure probability."

if st.sidebar.button(q1): st.session_state.input_value = q1
if st.sidebar.button(q2): st.session_state.input_value = q2
if st.sidebar.button(q3): st.session_state.input_value = q3

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Handle both typed input and button clicks
prompt = st.chat_input("Ask about sales pipeline or work orders...")
if st.session_state.input_value:
    prompt = st.session_state.input_value
    st.session_state.input_value = "" # Reset after use

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            output = ""
            try:
                agent = initialize_agent(selected_model="llama-3.3-70b-versatile")
                response = agent.invoke(prompt)
                output = response["output"]
            except Exception as e:
                if any(err in str(e) for err in ["429", "rate_limit", "413"]):
                    st.warning("🔄 Primary model busy. Switching to fallback...")
                    try:
                        agent = initialize_agent(selected_model="llama-3.1-8b-instant")
                        response = agent.invoke(prompt)
                        output = response["output"]
                    except Exception as fallback_e:
                        output = f"System under load. Try again soon. (Error: {fallback_e})"
                else:
                    output = f"Error: {e}"
            
            st.markdown(output)
            st.session_state.messages.append({"role": "assistant", "content": output})
