# seed_database.py
import pandas as pd
import os
import sys

# Add the root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from libraries.lib_db import get_db_engine, insert_dataframe_to_db

# --- CONFIGURATION ---
# Define which environment we are seeding. Change to 'dev' for the development DB.
TARGET_ENV = 'prod'

# Paths to your CSV data files
ASSETS_CSV_PATH = os.path.join(project_root, 'postgres', 'initial_WM_dim_assets.csv')
POSITIONS_CSV_PATH = os.path.join(project_root, 'postgres', 'initial_WM_positions.csv')

def main():
    """Main function to connect to the DB and run uploads."""
    print(f"--- Starting Database Seeding Process for '{TARGET_ENV}' environment ---")

    # 1. Get the database engine using our library function
    engine = get_db_engine(TARGET_ENV)
    if not engine:
        print("Halting process due to connection error.")
        return

    # 2. Load the CSV files into DataFrames
    try:
        df_assets = pd.read_csv(ASSETS_CSV_PATH)
        df_positions = pd.read_csv(POSITIONS_CSV_PATH)
        print("Successfully loaded CSV files into memory.")
    except FileNotFoundError as e:
        print(f"Error: Could not find a CSV file. {e}")
        return

    # 3. Insert the DataFrames into the database using our library function
    # The order is important due to the foreign key relationship.
    print("\n--- Uploading data to database ---")
    assets_success = insert_dataframe_to_db(df_assets, 'dim_assets', engine)

    # Only proceed to insert positions if assets were inserted successfully
    if assets_success:
        insert_dataframe_to_db(df_positions, 'positions', engine)

    print("\n--- Database Seeding Process Complete ---")

if __name__ == "__main__":
    main()