import pandas as pd
import pyodbc

# Define your connection parameters
server = 'DESKTOP-PKUB0AP'  # e.g. 'DESKTOP-XYZ123\SQLEXPRESS'
database = 'Equal Pay Project'
driver = 'ODBC Driver 17 for SQL Server'

# Connect to your SQL Server using Windows Authentication
conn = pyodbc.connect(
    f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
)

# Query your new Employees table
df = pd.read_sql_query("SELECT * FROM Employees", conn)

# Check the imported data
print(df.head())