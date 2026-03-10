# Skylark Drones BI Agent
An AI-driven Business Intelligence tool integrating Monday.com with Open Source LLMs.

### Key Features
- **Live Integration:** Real-time GraphQL queries to Monday.com boards.
- **Data Resilience:** Automated cleaning of messy currency/percentage strings.
- **Cascading Failover:** Dual-model architecture (70B & 8B) for 100% uptime.

### Setup
1. Clone the repo.
2. Add your `secrets.toml` with Monday.com and Groq keys.
3. Run `pip install -r requirements.txt`.
4. Launch with `streamlit run app.py`.
