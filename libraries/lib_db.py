# lib_db.py
from config import config

import sqlite3
# import mysql.connector
import json
import pandas as pd
from sqlalchemy import create_engine, text
import os

from libraries.lib_config import load_secrets 

####################################################################################################

#############################################
#############################################
#     SQLite
#############################################
#############################################

####################################################################################################

# Function to insert positions into the SQLite database
def insert_positions_to_db(df, db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Convert valuation_date to string if it is a datetime object. This is not the best place to do this!
    if pd.api.types.is_datetime64_any_dtype(df['valuation_date']):
        df['valuation_date'] = df['valuation_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Prepare the data for batch insertion (only inserting asset_id, current_value, and valuation_date)
    data = [
        (row['asset_id'], row['current_value'], row['valuation_date'])  # No .strftime() needed if it's already a string
        for index, row in df.iterrows()
    ]

    # Insert data in batch using executemany
    cursor.executemany('''
    INSERT INTO positions (asset_id, current_value, valuation_date)
    VALUES (?, ?, ?)
    ''', data)

    conn.commit()
    conn.close()

####################################################################################################

def extract_from_db(query, db_name, params=None):
    """
    Executes a SELECT-type query on an SQLite database and returns the result as a Pandas DataFrame.

    Parameters:
    - query (str): SQL query to execute.
    - db_name (str): Path to the SQLite database file.
    - params (tuple, optional): Parameters to safely inject into the query (for parameterized queries).

    Returns:
    - pd.DataFrame: The result of the query.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    
    try:
        # Execute the query with or without parameters
        if params:
            result_set = pd.read_sql_query(query, conn, params=params)
        else:
            result_set = pd.read_sql_query(query, conn)
        
        print("Query executed successfully")
        return result_set

    except sqlite3.Error as e:  # Catch SQLite-specific errors
        print(f"SQLite Error: {e}")
        return None

    except Exception as e:  # Catch any other unforeseen errors
        print(f"Unexpected Error: {e}")
        return None

    finally:
        # Ensure the connection is closed
        conn.close()
        print("Connection closed")

####################################################################################################

#############################################
#############################################
#     Postgres - Neon
#############################################
#############################################

####################################################################################################

def get_db_engine(env='dev'):
    """
    Creates and returns a SQLAlchemy engine for the specified environment
    by loading credentials from an encrypted secrets file.

    Args:
        env (str): The environment to connect to ('dev' or 'prod'). 
                Defaults to 'dev'.

    Returns:
        sqlalchemy.engine.Engine: The SQLAlchemy engine object, or None if an error occurs.
    """
    try:
        # Construct the path to the encrypted secrets.json.encrypted file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..'))
        secrets_path = os.path.join(project_root, 'config', 'secrets.json')
        encrypted_secrets_path = f"{secrets_path}.encrypted"

        if not os.path.exists(encrypted_secrets_path):
            raise FileNotFoundError(f"Encrypted secrets file not found at: {encrypted_secrets_path}")

        secrets = load_secrets(encrypted_secrets_path)
        if not secrets:
            raise ValueError("Failed to load or decrypt secrets.")

        # Get the database URL for the specified environment
        db_url = secrets.get('postgres', {}).get(env, {}).get('url')
        
        if not db_url:
            raise ValueError(f"Database URL for env '{env}' not found in secrets file.")
            
        print(f"Successfully created database engine for '{env}' environment.")
        return create_engine(db_url)

    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error creating database engine: {e}")
        return None

####################################################################################################

def insert_dataframe_to_db(df, table_name, engine):
    """
    Inserts a pandas DataFrame into a specified database table.

    This function uses the high-performance df.to_sql method to efficiently
    upload data. It's suitable for any table and any DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be inserted.
        table_name (str): The name of the target database table.
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine to use for the connection.

    Returns:
        bool: True if the insertion was successful, False otherwise.
    """
    if df.empty:
        print(f"Warning: DataFrame for table '{table_name}' is empty. Nothing to insert.")
        return True # Not an error, just nothing to do.

    print(f"Attempting to insert {len(df)} rows into '{table_name}'...")
    try:
        # Use the 'with' statement to ensure the connection is properly handled
        with engine.connect() as connection:
            df.to_sql(
                name=table_name,
                con=connection,
                if_exists='append',  # Add data to the table. Don't replace it.
                index=False,         # Do not write the DataFrame's index as a column.
                chunksize=1000       # Upload in chunks for memory efficiency.
            )
        print(f"Successfully inserted {len(df)} rows into '{table_name}'.")
        return True
    except Exception as e:
        print(f"Error inserting DataFrame into '{table_name}': {e}")
        return False

####################################################################################################

#      .
#    .:;:.
#  .:;;;;;:.
#    ;;;;;
#    ;;;;;
#    ;;;;;
#    ;;;;;
#    ;:;;;
# NEW CODE

##################################
##################################

# OLD CODE FROM PREVIOUS PROJECT
#    ;;;;;
#    ;;;;;
#    ;;;;;
#    ;;;;;
#    ;;;;;
#  ..;;;;;..
#   ':::::'
#     ':`

####################################################################################################

# Function to read credentialls for mysql
def read_secrets_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            secrets = json.load(file)
            return secrets
    except Exception as e:
        print(f"Error reading secrets from JSON file: {e}")
        return None

####################################################################################################

####################################################################################################

# Function to connect to mysql db
# def connect_to_mysql(json_file_path):
#     # Read connection details from the JSON file
#     secrets = read_secrets_from_json(json_file_path)
    
#     if secrets is None:
#         return None

#     try:
#         mysql_secrets = secrets.get("mysql", {})
#         # Construct the database URL for SQLAlchemy
#         db_url = f"mysql+mysqlconnector://{mysql_secrets['user']}:{mysql_secrets['password']}@{mysql_secrets['host']}:{mysql_secrets['port']}/{mysql_secrets['database']}"
        
#         # Create a connection to the MySQL database using SQLAlchemy
#         engine = create_engine(db_url)
#         connection = engine.connect()

#         # Return the connection object
#         return connection
    
#     except mysql.connector.Error as e:
#         print(f"Error: {e}")

#     # Return None if the connection fails
#     return None

####################################################################################################

####################################################################################################

# Function to run a SELECT-type query in mysql and return the result set as a Pandas DataFrame
# def run_select_query(secrets_file_path, query):
#     # Connect to MySQL database
#     conn = connect_to_mysql(secrets_file_path)

#     # Check if the connection is successful before proceeding
#     if conn:
#         try:
#             result_set = pd.read_sql(query, conn)
#             print("Query executed successfully")
#             return result_set

#         except mysql.connector.Error as e:
#             print(f"Error: {e}")

#         finally:
#             # Close the connection
#             conn.close()
#             print("Connection closed")

#     return None

####################################################################################################

####################################################################################################
# """
# # Function to run an INSERT-type query in mysql and return the result set as a Pandas DataFrame
# # 
# #    ----- secrets_file_path -----
# #    Define the path to 'secrets.json'
# #    files_directory = os.path.join(os.path.dirname(__file__), 'Config')
# #    json_file_path = os.path.join(files_directory, 'secrets.json')
# #    ----- sql_insert -----
# #    sql_insert = f"INSERT INTO {table_name} (test_name) VALUES (%s)"
# #    ----- data_to_insert -----
# #    the data_to_insert parameter should be a list of dictionaries in the format expected by SQLAlchemy for insertion:
# data_to_insert_list = {
#     'value1': 'some_value',
#     'value2': 42,
#     'value3': 'another_value'
# }

# """
# def run_insert_query(secrets_file_path, sql_insert, data_to_insert_list):
#     # Connect to MySQL database
#     conn = connect_to_mysql(secrets_file_path)
    
#     # Use text() to create a textual SQL expression
#     sql_insert = text(sql_insert)

#     # Check if the connection is successful before proceeding
#     if conn:
#         try:
#             # Use the connection's execute method directly
#             conn.execute(sql_insert, data_to_insert_list)
#             # Commit the changes
#             conn.commit()
#             print("Data inserted successfully")

#         except Exception as e:
#             print(f"Error: {e}")
#             conn.rollback()

#         finally:
#             # Close the connection
#             conn.close()
        
#     return None

####################################################################################################

####################################################################################################

# Function to create a dictionary with all tables and columns in a schema
# def get_tables_columns(secrets_file_path):
#     secrets = read_secrets_from_json(secrets_file_path)
#     mysql_database = secrets['mysql']['database']

#     #   Get all tables in the schema
#     query_select_tables = f"select table_schema as database_name, table_name as table_name" \
#     " from information_schema.tables" \
#     " where table_type = 'BASE TABLE'" \
#     f" and table_schema = '{mysql_database}'" \
#     " order by database_name, table_name;"
#     df_tables = run_select_query(secrets_file_path, query_select_tables)
#     # Create a dictionary from the DataFrame with table_name as keys
#     dict_table_columns = df_tables.set_index('table_name')['database_name'].to_dict()
#     # print(dict_table_columns)

#     #   Get columns from all tables in the schema
#     for table_name in dict_table_columns.keys():
#         query_select_columns = "SELECT COLUMN_NAME" \
#     " FROM INFORMATION_SCHEMA.COLUMNS" \
#     f" WHERE TABLE_SCHEMA = '{mysql_database}' AND TABLE_NAME = '{table_name}';"
#         df_columns = run_select_query(secrets_file_path, query_select_columns)
#         # Update dict_table_columns with a list of column names for the current table
#         dict_table_columns[table_name] = df_columns['COLUMN_NAME'].tolist()
#     # pprint(dict_table_columns)

#     return dict_table_columns

####################################################################################################

####################################################################################################

# Function to INSERT data from a dataframe into a mysql db table
# def insert_dataframe_in_table(secrets_file_path, df_data, table_name):
#     # Get all columns for all tables in the db schema
#     dict_table_columns = get_tables_columns(secrets_file_path)

#     # Get columns for table before INSERT
#     list_columns = dict_table_columns.get(table_name, [])
#     # exclude_columns for specific tables
#     if table_name == 'fact_expense_tb': exclude_columns = ['expense_id', 'category_id']
#     elif table_name == 'dim_category_tb': exclude_columns = ['category_id']
#     elif table_name == 'dim_keyword_tb': exclude_columns = ['keyword_id']
#     elif table_name == 'aux_multiple_matches_tb': exclude_columns = ['match_id'] 
#     else: exclude_columns = []
#     # Construct the list of columns to include in the SQL INSERT statement
#     list_columns = [col for col in list_columns if col not in exclude_columns]
#     print('list_columns:')
#     print(list_columns)
#     try: 
#         column_mapping = config.column_mapping[table_name]
#         print(column_mapping)
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         print('Table column_mapping not defined in config.py')
#         column_mapping = {item: item for item in list_columns}

#     # Construct the SQL INSERT statement
#     if list_columns:
#         # sql_insert = f"INSERT INTO {table_name} ({', '.join(list_columns)}) VALUES ({', '.join(['%s'] * len(list_columns))})"
#         sql_insert = f"INSERT INTO {table_name} ({', '.join(list_columns)}) VALUES ({', '.join([':%s' % col for col in list_columns])})"
#         print('sql_insert:')
#         print(sql_insert)
#     else:
#         print(f"No columns found for table_name: {table_name}")
#     # Construct the DATA to INSERT list of dictionaries
#     data_to_insert_list = []
#     for _, row in df_data.iterrows():
#         # Extracting values based on the mapping and list_columns
#         values = {column_mapping[col]: row[column_mapping[col]] for col in list_columns}
#         # Creating a new dictionary and appending to the list
#         data_to_insert_dict = {}
#         data_to_insert_dict.update(values)
#         data_to_insert_list.append(data_to_insert_dict)

#     print('data_to_insert_list:')
#     print(data_to_insert_list)

#     run_insert_query(secrets_file_path, sql_insert, data_to_insert_list)

#     return None

