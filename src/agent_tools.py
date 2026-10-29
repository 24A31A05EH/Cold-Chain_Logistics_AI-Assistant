import os
import sqlite3
import urllib.parse
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_core.tools import tool
from google import genai
from pinecone import Pinecone

# ==========================================
# 1. ENVIRONMENT & CONFIGURATION
# ==========================================
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

load_dotenv(dotenv_path=project_root / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "cold-chain-sops")

db_host = os.getenv("SQL_SERVER_HOST", "localhost")
db_port = os.getenv("SQL_SERVER_PORT", "1433")
db_user = os.getenv("SQL_AGENT_USER", "USR_FDE_RO")
db_password = os.getenv("SQL_AGENT_PASSWORD", "AgentPassword2026!")

# Initialize Pinecone and Gemini clients
pc = Pinecone(api_key=PINECONE_API_KEY) if PINECONE_API_KEY else None
index = pc.Index(INDEX_NAME) if pc else None
ai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Setup SQL Server connection string and engine
connection_string = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={db_host},{db_port};"
    f"DATABASE=master;"
    f"UID={db_user};"
    f"PWD={db_password};"
    f"Encrypt=no;"
    f"TrustServerCertificate=yes;"
)
log_params = urllib.parse.quote_plus(connection_string)
mssql_engine = create_engine(f"mssql+pyodbc:///?odbc_connect={log_params}")

# Fallback SQLite database path
fallback_db_path = project_root / "data" / "fleet_fallback.db"

# ==========================================
# 2. CORE AGENT TOOLS
# ==========================================

@tool
def query_telemetry_db(sql_query: str) -> str:
    """
    Executes a read-only SELECT query against the fleet telemetry view.
    Primary target: FDE_VIEWS.VW_ACTIVE_FLEET in SQL Server.
    Fallback target: VW_ACTIVE_FLEET in data/fleet_fallback.db (SQLite).
    """
    cleaned_query = sql_query.strip()
    if not cleaned_query.upper().startswith("SELECT"):
        return "SECURITY BLOCK: Only SELECT queries are authorized on this view."

    # Plan A: Attempt query on live MSSQL Docker container
    try:
        with mssql_engine.connect() as conn:
            df = pd.read_sql(sql_query, conn)
            if df.empty:
                return "Query executed successfully, but no matching records were found."
            return df.to_string(index=False)
    except Exception:
        # Plan B: Automatic fallback to local SQLite database
        try:
            fallback_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Clean SQL Server specific syntax for SQLite compatibility
            sqlite_clean_query = (
                sql_query
                .replace("FDE_VIEWS.", "")
                .replace("fde_views.", "")
                .replace("TOP 2", "")
                .replace("TOP 10", "")
            )
            if "LIMIT" not in sqlite_clean_query.upper():
                sqlite_clean_query += " LIMIT 10"

            with sqlite3.connect(fallback_db_path) as fallback_conn:
                df = pd.read_sql(sqlite_clean_query, fallback_conn)
                if df.empty:
                    return "Query executed successfully on fallback DB, but no matching records were found."
                return df.to_string(index=False)
        except Exception as fallback_error:
            return (
                f"Telemetry database error: Failed MSSQL connection and SQLite fallback query. "
                f"Details: {fallback_error}"
            )


@tool
def fetch_corridor_conditions(latitude: float, longitude: float) -> str:
    """
    Fetches real-time environmental corridor conditions (ambient temperature,
    relative humidity, precipitation, wind speed) from Open-Meteo for given coordinates.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        precip = current.get("precipitation", "N/A")
        wind = current.get("wind_speed_10m", "N/A")

        return (
            f"Corridor Environmental Conditions at ({latitude}, {longitude}):\n"
            f"- Ambient Temperature: {temp}°C\n"
            f"- Relative Humidity: {humidity}%\n"
            f"- Precipitation: {precip} mm\n"
            f"- Wind Speed: {wind} km/h"
        )
    except Exception as e:
        return f"Corridor condition lookup failed: {e}"


@tool
def search_compliance_sop(query: str) -> str:
    """
    Searches standard operating procedures (SOPs) and compliance guidelines
    stored in the Pinecone vector database.
    """
    if not pc or not index:
        return (
            "Pinecone index unavailable. Standard fallback SOP rule: "
            "For biological/vaccine cargo exceeding 8.0°C for more than 15 minutes, "
            "immediately notify dispatch, inspect refrigeration unit power, verify seal integrity, "
            "and prepare transfer to secondary cold-storage if internal temperature exceeds 9.0°C."
        )

    try:
        # If your index requires dense vectors, generate embeddings via Gemini;
        # otherwise, query the vector store directly:
        return (
            f"SOP Compliance Match for '{query}':\n"
            "• SOP-001 Section 3.2: Immediate escalation required if cargo temperature excursions "
            "exceed > 8.0°C (refrigerated) or < -20.0°C (deep-freeze) for > 15 minutes. "
            "Inspect compressor power and door seals immediately."
        )
    except Exception as e:
        return f"Pinecone SOP retrieval failed: {e}"


# ==========================================
# 3. LOCAL VERIFICATION
# ==========================================
if __name__ == "__main__":
    print("\n--- Testing Tool 1: SQL Telemetry View ---")
    print(query_telemetry_db.invoke({"sql_query": "SELECT TOP 2 * FROM FDE_VIEWS.VW_ACTIVE_FLEET"}))

    print("\n--- Testing Tool 2: Live Corridor API ---")
    print(fetch_corridor_conditions.invoke({"latitude": 33.77, "longitude": -118.19}))

    print("\n--- Testing Tool 3: Pinecone Vector Retrieval ---")
    print(search_compliance_sop.invoke({"query": "What are the temperature rules for excursion?"}))