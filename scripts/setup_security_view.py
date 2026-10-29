import pyodbc

conn_str = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=localhost,1433;"
    "DATABASE=master;"
    "UID=sa;"
    "PWD=FdeEnterprisePass123!;"
    "Encrypt=no;"
    "TrustServerCertificate=yes;"
)

statements = [
    # 1. Create dedicated schema if it doesn't already exist
    """
    IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'FDE_VIEWS')
    BEGIN
        EXEC('CREATE SCHEMA FDE_VIEWS');
    END
    """,

    # 2. Drop existing view if present
    """
    IF OBJECT_ID('FDE_VIEWS.VW_ACTIVE_FLEET', 'V') IS NOT NULL
        DROP VIEW FDE_VIEWS.VW_ACTIVE_FLEET;
    """,

    # 3. Create the Semantic View (exact column translation from the repo)
    """
    CREATE VIEW FDE_VIEWS.VW_ACTIVE_FLEET AS
    SELECT 
        [TS_UTC] AS [Timestamp],
        [V_LAT] AS [Latitude],
        [V_LON] AS [Longitude],
        CAST([IOT_TEMP_VAL_C] AS FLOAT) AS [Current_Temperature_C],
        [CGO_COND_CD] AS [Cargo_Condition_Code],
        [RISK_CLS_TXT] AS [Risk_Classification],
        [DELAY_PROB_DEC] AS [Delay_Probability],
        [PRT_CNG_LVL] AS [Port_Congestion_Level],
        [RT_RSK_IDX] AS [Route_Risk_Index]
    FROM dbo.TBL_SC_FLEET_HIST_RAW;
    """,

    # 4. Create strict Read-Only Login if not exists
    """
    IF NOT EXISTS (SELECT * FROM sys.server_principals WHERE name = 'USR_FDE_RO')
    BEGIN
        CREATE LOGIN USR_FDE_RO WITH PASSWORD = 'AgentPassword2026!', CHECK_POLICY = OFF;
    END
    """,

    # 5. Create Database User if not exists
    """
    IF NOT EXISTS (SELECT * FROM sys.database_principals WHERE name = 'USR_FDE_RO')
    BEGIN
        CREATE USER USR_FDE_RO FOR LOGIN USR_FDE_RO;
    END
    """,

    # 6. Grant access ONLY to semantic view & explicitly deny dbo schema
    "GRANT SELECT ON FDE_VIEWS.VW_ACTIVE_FLEET TO USR_FDE_RO;",
    "DENY SELECT ON dbo.TBL_SC_FLEET_HIST_RAW TO USR_FDE_RO;",
    "DENY INSERT, UPDATE, DELETE, ALTER ON SCHEMA::dbo TO USR_FDE_RO;"
]

print("Connecting to SQL Server as sa...")
conn = pyodbc.connect(conn_str, autocommit=True)
cursor = conn.cursor()

for i, stmt in enumerate(statements, start=1):
    cursor.execute(stmt)

print("✅ FDE Enterprise Security & Semantic View successfully created!")
conn.close()