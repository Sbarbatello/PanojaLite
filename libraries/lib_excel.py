# lib_excel.py
import os
import pandas as pd
import numpy as np
import re
import streamlit as st
from io import BytesIO, StringIO

####################################################################################################

def convert_excel_to_csv(excel_file, rows_to_skip=0):  # Add rows_to_skip parameter
    """Converts an Excel file (xls or xlsx) to a CSV file, 
       keeping the same location and file name.

    Args:
        excel_file: Path to the Excel file.
        rows_to_skip: Number of rows to skip at the beginning of the file.
    Returns:
        The path to the created CSV file, or None if an error occurred.
    """
    try:
        df = pd.read_excel(excel_file, skiprows=rows_to_skip, header=None)  # Skip rows here

        # Dynamic header detection (improved)
        header_row = None
        for i in range(min(5, len(df))):  # Check first 5 rows (adjust if needed)
            if all(pd.notna(df.iloc[i])):
                header_row = i
                break

        if header_row is not None:
            df.columns = df.iloc[header_row]
            df = df.iloc[header_row + 1:].reset_index(drop=True)
        else:
            print(f"Warning: Could not find a valid header row in {excel_file}. Using default headers.")

        csv_file = os.path.splitext(excel_file)[0] + ".csv"  # Same name, .csv extension
        df.to_csv(csv_file, index=False)
        # print(f"Successfully converted {excel_file} to {csv_file}")
        return csv_file  # Return the CSV file path
    except Exception as e:
        print(f"Error converting {excel_file} to CSV: {e}")
        return None  # Return None to indicate failure

####################################################################################################

# Function to process the positions and return the cleaned DataFrame
def process_positions_old(asset_id, file_location, type_of_account):
    # filter by type_of_account
    if type_of_account == 'bankinter_legacy':
        # Load the data from the Excel file
        df = pd.read_excel(file_location)

        # Convert 'F. CONTABLE' column to datetime format (ensure the date column is in datetime format)
        df['F. CONTABLE'] = pd.to_datetime(df['F. CONTABLE'], errors='coerce')
        # Create a 'Month-Year' column to group by month (for internal processing, not for insertion)
        df['Month-Year'] = df['F. CONTABLE'].dt.to_period('M')
        # Calculate the absolute difference to the 1st of the month
        df['Days_to_1st'] = (df['F. CONTABLE'] - df['F. CONTABLE'].dt.to_period('M').dt.start_time).abs()
        # Rank the entries within each month by the closeness to the 1st of the month
        df['Rank'] = df.groupby('Month-Year')['Days_to_1st'].rank(method='min', ascending=True)

        # Filter the rows where rank is 1 (i.e., closest to the 1st of the month)
        closest_to_1st = df[df['Rank'] == 1]

        # Eliminate duplicates within the closest rows for each month by choosing the highest 'SALDO'
        closest_to_1st = closest_to_1st.loc[closest_to_1st.groupby('Month-Year')['SALDO'].idxmax()]

        # Insert the asset_id as the first column
        closest_to_1st.insert(0, 'asset_id', asset_id)

        # Rename columns
        closest_to_1st = closest_to_1st.rename(columns={'F. CONTABLE': 'valuation_date', 'SALDO': 'current_value'})

        # Extract the relevant columns in the desired order
        result = closest_to_1st[['asset_id', 'current_value', 'valuation_date']]

        return result

    elif type_of_account == 'lloyds':
        df = pd.read_csv(file_location)
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], format='%d/%m/%Y')
        df = df.sort_values('Transaction_Date')

        start_date = df['Transaction_Date'].min()
        end_date = df['Transaction_Date'].max()

        # Generate all months (simplified)
        all_months = pd.date_range(start=start_date, end=end_date, freq='MS')
        all_months_df = pd.DataFrame({'valuation_date': all_months})
        
        # Merge ALL months with the original data (using datetime directly)
        merged_df = pd.merge(all_months_df, df[['Transaction_Date', 'Balance']],
                            left_on='valuation_date', right_on='Transaction_Date', how='left')

        # Find closest prior date using np.searchsorted
        for index, row in merged_df.iterrows():
            if pd.isna(row['Balance']):
                valuation_date = row['valuation_date']

                # No conversion needed, use valuation_date directly
                transaction_dates = df['Transaction_Date'].tolist()  # Convert to list for faster search
                sorted_dates_index = np.searchsorted(transaction_dates, valuation_date)

                if sorted_dates_index == 0:
                    continue

                closest_prior_index = max(0, sorted_dates_index - 1)
                merged_df.loc[index, 'Balance'] = df.iloc[closest_prior_index]['Balance']

        # Add asset_id
        merged_df['asset_id'] = asset_id

        # Rename columns
        result_df = merged_df.rename(columns={'Balance': 'current_value'})

        # Convert to string ONLY when you need the string representation
        result_df['valuation_date'] = result_df['valuation_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return result_df
    
    elif type_of_account == 'bankinter':
        rows_to_skip = 3  # Number of rows to skip
        csv_location = convert_excel_to_csv(file_location, rows_to_skip)
        if csv_location: # Check if conversion was successful
            df = pd.read_csv(csv_location)
            # df manipulation:
            df['FECHA CONTABLE'] = pd.to_datetime(df['FECHA CONTABLE'], format='%d/%m/%Y', errors='coerce')  # Correct format is %d/%m/%Y
            df['Month-Year'] = df['FECHA CONTABLE'].dt.to_period('M')
            df['Days_to_1st'] = (df['FECHA CONTABLE'] - df['FECHA CONTABLE'].dt.to_period('M').dt.start_time).abs()
            df['Rank'] = df.groupby('Month-Year')['Days_to_1st'].rank(method='min', ascending=True)
            
            # Filter the rows where rank is 1 (i.e., closest to the 1st of the month)
            closest_to_1st = df[df['Rank'] == 1]
            # Eliminate duplicates within the closest rows for each month by choosing the highest 'SALDO'
            closest_to_1st = closest_to_1st.loc[closest_to_1st.groupby('Month-Year')['SALDO'].idxmax()]
            # Insert the asset_id as the first column
            closest_to_1st.insert(0, 'asset_id', asset_id)
            # Rename columns
            closest_to_1st = closest_to_1st.rename(columns={'FECHA CONTABLE': 'valuation_date', 'SALDO': 'current_value'})
            # Extract the relevant columns in the desired order
            result = closest_to_1st[['asset_id', 'current_value', 'valuation_date']]
            return result
        else:
            print("Conversion failed. Check the error messages.")

####################################################################################################

def convert_excel_to_csv_in_memory(excel_file, rows_to_skip=0):
    """Converts an Excel file (xls or xlsx) to a CSV string in memory.

    Args:
        excel_file: Either a file path (string) or a BytesIO object representing the Excel file.
        rows_to_skip: Number of rows to skip at the beginning of the file.

    Returns:
        A string containing the CSV data, or None if an error occurred.
    """
    try:
        if isinstance(excel_file, str):
            df = pd.read_excel(excel_file, skiprows=rows_to_skip, header=None)
        elif isinstance(excel_file, BytesIO):
            df = pd.read_excel(excel_file, skiprows=rows_to_skip, header=None)
        else:
            raise ValueError("Invalid input: Expected file path or BytesIO object")

        header_row = None
        for i in range(min(5, len(df))):
            if all(pd.notna(df.iloc[i])):
                header_row = i
                break

        if header_row is not None:
            df.columns = df.iloc[header_row]
            df = df.iloc[header_row + 1:].reset_index(drop=True)
        else:
            print(f"Warning: Could not find a valid header row in {excel_file}. Using default headers.")

        csv_buffer = StringIO()  # Create an in-memory text buffer
        df.to_csv(csv_buffer, index=False)  # Write DataFrame to buffer as CSV
        csv_string = csv_buffer.getvalue()  # Get the CSV string from the buffer
        return csv_string

    except Exception as e:
        print(f"Error converting Excel to CSV in memory: {e}")
        return None
    
####################################################################################################

def process_positions(asset_id, file_or_bytes, type_of_account):
    """Processes financial position data from either a file path or in-memory bytes.

    This function reads and processes data from various financial account types
    (bankinter_legacy, lloyds, bankinter) stored in Excel or CSV format.  It handles
    both file paths (strings) and in-memory file-like objects (BytesIO).

    Args:
        asset_id: The ID of the asset.
        file_or_bytes: Either a file path (string) to the data file or a
            BytesIO object containing the file content in memory.
        type_of_account: The type of financial account ('bankinter_legacy', 'lloyds',
            'bankinter').  This determines how the data is processed.

    Returns:
        A pandas DataFrame containing the processed financial position data, or
        None if an error occurs during processing.

    Raises:
        ValueError: If the `file_or_bytes` input is neither a string nor a
            BytesIO object.
    """
    if isinstance(file_or_bytes, str):  # Check if it's a file path (string) - ONCE
        file_input = file_or_bytes
    elif isinstance(file_or_bytes, BytesIO):  # Check if it's BytesIO - ONCE
        file_input = file_or_bytes
    else:
        raise ValueError("Invalid input: Expected file path or BytesIO object")

    try:  # Handle potential exceptions during file processing
        if type_of_account == 'bankinter_legacy':
            df = pd.read_excel(file_input)  # Read using file_input (string or BytesIO)
            df['F. CONTABLE'] = pd.to_datetime(df['F. CONTABLE'], errors='coerce')
            df['Month-Year'] = df['F. CONTABLE'].dt.to_period('M')
            df['Days_to_1st'] = (df['F. CONTABLE'] - df['F. CONTABLE'].dt.to_period('M').dt.start_time).abs()
            df['Rank'] = df.groupby('Month-Year')['Days_to_1st'].rank(method='min', ascending=True)

            closest_to_1st = df[df['Rank'] == 1]
            closest_to_1st = closest_to_1st.loc[closest_to_1st.groupby('Month-Year')['SALDO'].idxmax()]

            closest_to_1st.insert(0, 'asset_id', asset_id)
            closest_to_1st = closest_to_1st.rename(columns={'F. CONTABLE': 'valuation_date', 'SALDO': 'current_value'})
            result = closest_to_1st[['asset_id', 'current_value', 'valuation_date']]

            return result

        elif type_of_account == 'lloyds':
            df = pd.read_csv(file_input)  # Read using file_input (string or BytesIO)
            df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], format='%d/%m/%Y')
            df = df.sort_values('Transaction_Date')

            start_date = df['Transaction_Date'].min()
            end_date = df['Transaction_Date'].max()

            all_months = pd.date_range(start=start_date, end=end_date, freq='MS')
            all_months_df = pd.DataFrame({'valuation_date': all_months})

            merged_df = pd.merge(all_months_df, df[['Transaction_Date', 'Balance']],
                                left_on='valuation_date', right_on='Transaction_Date', how='left')

            for index, row in merged_df.iterrows():
                if pd.isna(row['Balance']):
                    valuation_date = row['valuation_date']

                    transaction_dates = df['Transaction_Date'].tolist()
                    sorted_dates_index = np.searchsorted(transaction_dates, valuation_date)

                    if sorted_dates_index == 0:
                        continue

                    closest_prior_index = max(0, sorted_dates_index - 1)
                    merged_df.loc[index, 'Balance'] = df.iloc[closest_prior_index]['Balance']

            merged_df['asset_id'] = asset_id
            result_df = merged_df.rename(columns={'Balance': 'current_value'})
            result_df['valuation_date'] = result_df['valuation_date'].dt.strftime('%Y-%m-%d %H:%M:%S')

            return result_df

        elif type_of_account == 'bankinter':
            csv_string = convert_excel_to_csv_in_memory(file_input, rows_to_skip=3)  # Get CSV string
            if csv_string:
                df = pd.read_csv(StringIO(csv_string))  # Read CSV from string
                df['FECHA CONTABLE'] = pd.to_datetime(df['FECHA CONTABLE'], format='%d/%m/%Y', errors='coerce')
                df['Month-Year'] = df['FECHA CONTABLE'].dt.to_period('M')
                df['Days_to_1st'] = (df['FECHA CONTABLE'] - df['FECHA CONTABLE'].dt.to_period('M').dt.start_time).abs()
                df['Rank'] = df.groupby('Month-Year')['Days_to_1st'].rank(method='min', ascending=True)

                closest_to_1st = df[df['Rank'] == 1]
                closest_to_1st = closest_to_1st.loc[closest_to_1st.groupby('Month-Year')['SALDO'].idxmax()]

                closest_to_1st.insert(0, 'asset_id', asset_id)
                closest_to_1st = closest_to_1st.rename(columns={'FECHA CONTABLE': 'valuation_date', 'SALDO': 'current_value'})
                result = closest_to_1st[['asset_id', 'current_value', 'valuation_date']]

                return result
            else:
                st.error("Error converting Excel to CSV in memory.")
                return None  # Or handle as you see fit

        return None  # Return None if no matching type_of_account is found.

    except Exception as e:  # Catch and handle any exception during processing
        print(f"Error processing file: {e}")
        st.error(f"An error occurred during file processing: {e}")
        return None

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

# Function to read Excel for "acc" files
def read_acc_xls(file_path):
    # Read the Excel file into a DataFrame
    df_contents = pd.read_excel(file_path)
    # Extract the first column title and content of cell [0, 0]
    acc_iban = re.search(r'IBAN:\s*([^\s]+)', df_contents.columns[0]).group(1)
    acc_holder = re.search(r'titular: (.+?) /', df_contents.iat[0, 0]).group(1)
    # Create a new DataFrame df_acc_transactions with specific columns (0, 2, 3, 4) and rows from row 3 onwards
    df_acc_transactions = df_contents.iloc[3:].copy()  # Copy rows from 3rd row to the bottom
    df_acc_transactions = df_acc_transactions.drop(df_acc_transactions.columns[1], axis=1, errors='ignore')
    # Rename columns with specified names
    df_acc_transactions.columns = ['transaction_date', 'transaction_description', 'transaction_amount', 'account_current_balance']
    # Reset indexes
    df_acc_transactions = df_acc_transactions.reset_index(drop=True)
    # Extract current balance from the last row and column E
    current_balance = df_acc_transactions.at[df_acc_transactions.index[-1], 'account_current_balance']
    # Extarct date from the last row
    last_date = df_acc_transactions.at[df_acc_transactions.index[-1], 'transaction_date']
    return acc_iban, acc_holder, df_acc_transactions, current_balance, last_date

####################################################################################################

# Function to read Excel for "car" files
def read_car_xls(file_path):
    # Read the Excel file into a DataFrame
    df_contents = pd.read_excel(file_path)
    # Create a new DataFrame with card transactions
    df_car_transactions = extract_date_cols(df_contents)
    # Reset indexes
    df_car_transactions = df_car_transactions.reset_index(drop=True)
    return df_car_transactions

####################################################################################################

# Function to extract transactions based on criteria
def extract_date_cols(dataframe):
    # Identify the starting row (where the date values start)
    # Iterate through the rows and find the first row with a valid datetime entry
    for i, date_str in enumerate(dataframe.iloc[:, 0]):
        if pd.notna(date_str):
            try:
                pd.to_datetime(date_str)
                start_row = i
                break
            except ValueError:
                continue

    # If no valid datetime entry is found, set start_row to 0
    if 'start_row' not in locals():
        start_row = None

    if start_row is None:
        return None  # No valid start row found

    # Identify the ending row (just before the first row with non-date data in the first column)
    end_row = dataframe.iloc[start_row:][dataframe.iloc[start_row:][dataframe.columns[0]].apply(lambda x: not str(x).strip().startswith('20'))].index.min()

    if end_row is None or end_row <= start_row:
        return None  # No valid end row found

    # Extract the required rows and rename the columns
    output_df = dataframe.loc[start_row:end_row-1, :]
    output_df.columns = ['transaction_date', 'transaction_description', 'transaction_amount']
    return output_df

####################################################################################################

# Function to generate dictionary of dataframes with account transactions from excel files
def generate_acc_transactions():
    # Define the path to the "Files" directory
    files_directory = os.path.join(os.path.dirname(__file__), '..', 'Files')
    # Initialize dictionary to store file details and transactions
    acc_data_dict = {}
    # Iterate over files in the "Files" directory
    for filename in os.listdir(files_directory):
        try:
            if filename.startswith('acc') and filename.endswith('.xls'):
                # Construct the full file path
                file_path = os.path.join(files_directory, filename)
                # Read Excel for "acc" files
                _, acc_holder, df_acc_transactions, _, _ = read_acc_xls(file_path)
                # put acc_holder and filename inside dataframe:
                df_acc_transactions['transaction_owner'] = acc_holder
                df_acc_transactions['filename'] = filename
                # print(df_acc_transactions['transaction_owner'])
                acc_data_dict[filename] = df_acc_transactions
        except Exception as e:
            print(f"An error occurred while reading {filename}: {str(e)}")
    return acc_data_dict

####################################################################################################

# Function to generate a dictionary of dataframes with card transactions from excel files
def generate_car_transactions():
    # Define the path to the "Files" directory
    files_directory = os.path.join(os.path.dirname(__file__), '..', 'Files')
    # Define the path to the "Config" directory
    config_directory = os.path.join(os.path.dirname(__file__), '..', 'Config')
    # Read file Config\cards.csv into a DataFrame
    # Construct the full file path
    cards_csv_path = os.path.join(config_directory, 'cards.csv')
    df_cards = pd.read_csv(cards_csv_path)
    # Initialize dictionary to store file details and transactions
    car_data_dict = {}
    # Iterate over files in the "Files" directory
    for filename in os.listdir(files_directory):
        try:
            if filename.startswith('car') and filename.endswith('.xls'):
                # Construct the full file path
                file_path = os.path.join(files_directory, filename)
                # Read Excel for "car" files
                df_car_transactions = read_car_xls(file_path)
                # extract the card_id
                card_id = int(filename[4:8])
                # Check if card_id_xx is in the DataFrame
                if card_id in df_cards['card_id'].values:
                    # Retrieve the corresponding card_holder value
                    card_holder = df_cards.loc[df_cards['card_id'] == card_id, 'card_holder'].values[0]
                else:
                    card_holder = 'Not found in cards.csv'
                df_car_transactions['transaction_owner'] = card_holder
                df_car_transactions['filename'] = filename
                # Store the file name and contents in the "car" dictionary
                car_data_dict[filename] = df_car_transactions
        except Exception as e:
            print(f"An error occurred while reading {filename}: {str(e)}")
    return car_data_dict

####################################################################################################

# Function to generate dictionary of dataframes with account balances from excel files
def generate_acc_balances():
    # Define the path to the "Files" directory
    files_directory = os.path.join(os.path.dirname(__file__), '..', 'Files')
    # Initialize dictionary to store file details and transactions
    acc_balance_dict = {}
    # Iterate over files in the "Files" directory
    for filename in os.listdir(files_directory):
        try:
            if filename.startswith('acc') and filename.endswith('.xls'):
                # Construct the full file path
                file_path = os.path.join(files_directory, filename)
                # Read Excel for "acc" files
                acc_iban, _, _, current_balance, last_date = read_acc_xls(file_path)
                # print(df_acc_transactions['transaction_owner'])
                acc_balance_dict[filename] = (last_date, acc_iban, current_balance)
        except Exception as e:
            print(f"An error occurred while reading {filename}: {str(e)}")
    return acc_balance_dict

####################################################################################################