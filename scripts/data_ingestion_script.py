# data_ingestion_script.py
import streamlit as st
import pandas as pd
import sys
import os
from io import BytesIO
from streamlit_extras.stylable_container import stylable_container
from datetime import datetime, timedelta

# Add the root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from libraries import lib_excel, lib_db, lib_streamlit, lib_api
from config import config

####################################################################################################

def get_asset_info():
    """Return static asset information."""
    return {
        "acc_bk_sm_5081": {
            "title": lib_streamlit.apply_labelling_assets('acc_bk_sm_5081'),
            "ingestion_type": "file",
            "asset_id": 1,
            "type_of_account": "bankinter"
        },
        "acc_bk_co_4264": {
            "title": lib_streamlit.apply_labelling_assets('acc_bk_co_4264'),
            "ingestion_type": "file",
            "asset_id": 2,
            "type_of_account": "bankinter"
        },
        "acc_bk_cc_4257": {
            "title": lib_streamlit.apply_labelling_assets('acc_bk_cc_4257'),
            "ingestion_type": "file",
            "asset_id": 3,
            "type_of_account": "bankinter"
        },
        "acc_lloyds": {
            "title": lib_streamlit.apply_labelling_assets('acc_lloyds'),
            "ingestion_type": "file",
            "asset_id": 4,
            "type_of_account": "lloyds"
        },
        "indexa_cc": {
            "title": lib_streamlit.apply_labelling_assets('indexa_cc'),
            "ingestion_type": "api",
            "asset_id": 5,
            "API_connector": "Indexa",
            "API_account": "7VX11611"
        },
        "indexa_sm": {
            "title": lib_streamlit.apply_labelling_assets('indexa_sm'),
            "ingestion_type": "api",
            "asset_id": 6,
            "API_connector": "Indexa",
            "API_account": "HGGPDGYF"
        },
        "allianz_inv_cc": {
            "title": lib_streamlit.apply_labelling_assets('allianz_inv_cc'),
            "ingestion_type": "manual",
            "asset_id": 7
        },
        "allianz_inv_sm": {
            "title": lib_streamlit.apply_labelling_assets('allianz_inv_sm'),
            "ingestion_type": "manual",
            "asset_id": 8
        },
        "fidelity_ap": {
            "title": lib_streamlit.apply_labelling_assets('fidelity_ap'),
            "ingestion_type": "manual",
            "asset_id": 9
        },
        "indexa_pension_cc": {
            "title": lib_streamlit.apply_labelling_assets('indexa_pension_cc'),
            "ingestion_type": "api",
            "asset_id": 10,
            "API_connector": "Indexa",
            "API_account": "8PIEQJYB"
        },
        "allianz_pension_sm": {
            "title": lib_streamlit.apply_labelling_assets('allianz_pension_sm'),
            "ingestion_type": "manual",
            "asset_id": 11
        },
        "mutua_pension": {
            "title": lib_streamlit.apply_labelling_assets('mutua_pension'),
            "ingestion_type": "manual",
            "asset_id": 12
        },
        "generali_pension": {
            "title": lib_streamlit.apply_labelling_assets('generali_pension'),
            "ingestion_type": "manual",
            "asset_id": 13
        },
        "azcona_43_house": {
            "title": lib_streamlit.apply_labelling_assets('azcona_43_house'),
            "ingestion_type": "manual",
            "asset_id": 14
        },
        "mini_car": {
            "title": lib_streamlit.apply_labelling_assets('mini_car'),
            "ingestion_type": "manual",
            "asset_id": 15
        },
        "ducati_bike": {
            "title": lib_streamlit.apply_labelling_assets('ducati_bike'),
            "ingestion_type": "manual",
            "asset_id": 16
        },
        "watch_collection": {
            "title": lib_streamlit.apply_labelling_assets('watch_collection'),
            "ingestion_type": "manual",
            "asset_id": 17
        },
        "azcona_43_parking": {
            "title": lib_streamlit.apply_labelling_assets('azcona_43_parking'),
            "ingestion_type": "manual",
            "asset_id": 18
        },
        "mortgage_azcona_43": {
            "title": lib_streamlit.apply_labelling_assets('mortgage_azcona_43'),
            "ingestion_type": "suggested_manual",
            "asset_id": 19
        }
    }

####################################################################################################

def load_data_and_refresh(asset_id, excel_file_content, type_of_account, db_engine):
    """
    Processes an uploaded file, filters for new data, and inserts it into the database.

    Args:
        asset_id (int): The ID of the asset.
        excel_file_content (bytes): The content of the uploaded file.
        type_of_account (str): The account type for processing logic.
        db_engine (sqlalchemy.engine.Engine): The database engine for the connection.
    """
    try:
        load_result = lib_excel.process_positions(asset_id, BytesIO(excel_file_content), type_of_account)
        
        if load_result is not None:
            asset_row = st.session_state['asset_data'][st.session_state['asset_data']['asset_id'] == asset_id]
            if not asset_row.empty and pd.notna(asset_row['latest_valuation_date'].iloc[0]):
                max_valuation_date = pd.Timestamp(asset_row['latest_valuation_date'].iloc[0])
                load_result['valuation_date'] = pd.to_datetime(load_result['valuation_date'])
                filtered_data = load_result[load_result['valuation_date'] > max_valuation_date]
            else:
                # If no existing data, all loaded data is new
                filtered_data = load_result

            if not filtered_data.empty:
                with st.spinner("Loading data into the database..."):

                    success = lib_db.insert_dataframe_to_db(filtered_data, 'positions', db_engine)
                    
                    if success:
                        # Refresh the session state data using the engine
                        lib_streamlit.get_recent_asset_data.clear()
                        st.session_state['asset_data'] = lib_streamlit.get_recent_asset_data(db_engine)
                        st.success("Data loaded successfully!")
                    else:
                        st.error("Failed to insert data into the database.")
            else:
                st.warning("No new data was loaded; all records are older than the latest valuation date.")
        else:
            st.error("Error processing the uploaded excel file.")
    except Exception as e:
        st.error(f"An unexpected error occurred during data refresh: {e}")

####################################################################################################

def calculate_missing_months(latest_date, current_date):
    """
    Calculates missing months between a latest date and the current date.

    This is a pure function that only performs date calculations and does not
    interact with the database.
    """
    if pd.isna(latest_date):
        # If there's no latest_date, the only missing month is the current one.
        return [current_date.replace(day=1)]
    
    # Ensure we are comparing date objects, not datetime objects
    if isinstance(latest_date, (datetime, pd.Timestamp)):
        latest_date = latest_date.date()
    if isinstance(current_date, (datetime, pd.Timestamp)):
        current_date = current_date.date()
    
    # Handle string conversion if necessary (though ideally we pass date objects)
    if isinstance(latest_date, str):
        latest_date = datetime.strptime(latest_date, '%Y-%m-%d').date()

    if latest_date.replace(day=1) >= current_date.replace(day=1):
        return []

    missing_months = []
    # Start from the month *after* the latest_date's month
    current_month = (latest_date.replace(day=1) + timedelta(days=32)).replace(day=1)

    while current_month <= current_date.replace(day=1):
        missing_months.append(current_month)
        # A more robust way to get to the next month
        current_month = (current_month + timedelta(days=32)).replace(day=1)

    return missing_months

####################################################################################################

def load_manual_data_and_refresh(df, asset_id, db_engine):
    """
    Filters a DataFrame of manual/API data and inserts it into the database.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to be inserted.
        asset_id (int): The ID of the asset.
        db_engine (sqlalchemy.engine.Engine): The database engine for the connection. # <-- CHANGE 2
    """
    try:
        if not df.empty:
            asset_row = st.session_state['asset_data'][st.session_state['asset_data']['asset_id'] == asset_id]
            if not asset_row.empty and pd.notna(asset_row['latest_valuation_date'].iloc[0]):
                max_valuation_date = pd.Timestamp(asset_row['latest_valuation_date'].iloc[0])
                df['valuation_date'] = pd.to_datetime(df['valuation_date'])
                filtered_data = df[df['valuation_date'] > max_valuation_date]
            else:
                filtered_data = df

            if not filtered_data.empty:
                with st.spinner("Loading data into the database..."):
                        
                    success = lib_db.insert_dataframe_to_db(filtered_data, 'positions', db_engine)
                    
                    if success:
                        # Refresh the session state data using the engine
                        lib_streamlit.get_recent_asset_data.clear()
                        st.session_state['asset_data'] = lib_streamlit.get_recent_asset_data(db_engine)
                        st.success("Data loaded successfully!")
                        st.rerun() # Rerun the script to reflect changes immediately
                    else:
                        st.error("Failed to insert data into the database.")
            else:
                st.warning("No new data was loaded; all records are older than the latest valuation date.")
        else:
            st.error("Dataframe is empty.")
    except Exception as e:
        st.error(f"An unexpected error occurred during manual data refresh: {e}")

####################################################################################################

def render_manual_ingestion_form(asset_id_key, data, asset_row, latest_valuation_date, db_engine):
    """
    Render the form for manual data ingestion for a specific asset.

    Args:
        asset_id_key (str): The unique key for the asset being processed.
        data (dict): Metadata about the asset (e.g., title, ingestion type).
        asset_row (pd.DataFrame): A single-row DataFrame containing the latest asset data.
        db_engine (sqlalchemy.engine.Engine): The database engine for the connection.

    Returns:
        None: This function renders the manual ingestion form directly in the Streamlit app.
    """
    current_date = datetime.now()
    missing_dates = calculate_missing_months(latest_valuation_date, current_date)

    with st.form(key=f'manual_ingestion_{asset_id_key}'):
        # Display the last balance
        if not asset_row.empty and 'latest_asset_value' in asset_row.columns:
            val = asset_row['latest_asset_value'].iloc[0]
            if pd.isna(val):
                latest_balance = 0.0 # Default to 0.0 if NaN
            else:
                latest_balance = val
        else:
            latest_balance = 0.0 # Default if asset_row is empty or column is missing

        if isinstance(latest_balance, (float, int)):
            st.markdown(f"Last balance in DB: {latest_balance:,.2f}")
        else:
            st.markdown(f"Last balance in DB: {latest_balance}")

        # Create input fields for missing dates
        input_data = {}
        for missing_date in missing_dates:
            input_data[missing_date] = st.text_input(
                label=f"Value for {missing_date.strftime('%Y-%m-%d')}",
                key=f"value_{asset_id_key}_{missing_date.strftime('%Y-%m-%d')}"
            )

        # Submit button
        if st.form_submit_button("Ingest data to DB"):
            df_data = []
            valid_inputs = True  # Flag to track input validity

            for date, value in input_data.items():
                if value:  # Check if a value was entered
                    try:
                        numeric_value = float(value)
                        df_data.append({
                            'asset_id': data['asset_id'],
                            'current_value': numeric_value,
                            'valuation_date': date.strftime('%Y-%m-%d %H:%M:%S')
                        })
                    except ValueError:
                        st.error(f"Invalid numeric value for {date.strftime('%Y-%m-%d')}: '{value}'")
                        valid_inputs = False  # Set flag to False
                        break  # Stop processing other inputs
                else:
                    st.warning(f"No value entered for {date.strftime('%Y-%m-%d')}.")
                    valid_inputs = False #Set flag to false
                    break #stop the proccessing

            if valid_inputs and df_data:
                df = pd.DataFrame(df_data)
                # Call a new function to handle manual data insertion
                load_manual_data_and_refresh(df, data["asset_id"], db_engine)
            elif not valid_inputs:
                pass
            else:
                st.warning("No valid data entered for ingestion.")

####################################################################################################

def render_file_ingestion_form(asset_id_key, data, asset_row, db_engine):
    """
    Render the form for file-based data ingestion for a specific asset.

    Args:
        asset_id_key (str): The unique key for the asset being processed.
        data (dict): Metadata about the asset (e.g., title, ingestion type).
        asset_row (pd.DataFrame): A single-row DataFrame containing the latest asset data.
        db_engine (sqlalchemy.engine.Engine): The database engine for the connection.

    Returns:
        None: This function renders the file ingestion form directly in the Streamlit app.
    """
    uploaded_file = st.file_uploader(f"Upload file for {asset_id_key}", key=f"uploader_{asset_id_key}")

    if uploaded_file is not None:
        try:
            excel_file_content = uploaded_file.getvalue()
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
            return  # Skip to the next file upload if an error occurs

        # Preview button
        if st.button(f"Preview data to load {asset_id_key}"):
            try:
                excel_file_content = uploaded_file.getvalue()
                preview_result = lib_excel.process_positions(
                    data["asset_id"], BytesIO(excel_file_content), data["type_of_account"]
                )  # Pass BytesIO object

                if preview_result is not None:
                    st.write("Data Preview:")
                    #Find the max_valuation_date for the correct asset.
                    asset_row = st.session_state['asset_data'][st.session_state['asset_data']['asset_id'] == data['asset_id']]
                    if not asset_row.empty:
                        max_valuation_date = asset_row['latest_valuation_date'].iloc[0]
                        lib_streamlit.style_preview_dataframe(preview_result, max_valuation_date)
                    else:
                        st.error(f"Asset ID not found in database")

                else:
                    st.error(f"Error processing data for preview: Check console for details.")

            except Exception as e:
                st.error(f"Error previewing file: {e}")

        # Load button
        if st.button(f"Load to DB {asset_id_key}", on_click=load_data_and_refresh, args=(data["asset_id"], uploaded_file.getvalue(), data["type_of_account"], db_engine)):
            if 'asset_data' in st.session_state:
                asset_row = st.session_state['asset_data'][st.session_state['asset_data']['asset_id'] == data['asset_id']]
                if asset_row.empty:
                    st.error(f"Asset ID {data['asset_id']} not found in the asset data.")
                else:
                    # The success message is now handled inside load_data_and_refresh, 
                    # but leaving a placeholder here is fine.
                    pass
            else:
                st.error(f"Asset data not available in session state.")

####################################################################################################

def render_api_ingestion_form(asset_id_key, data, asset_row, latest_valuation_date, db_engine, secrets_file_path):
    """
    Render the form for API-based data ingestion for a specific asset.

    Args:
        asset_id_key (str): The unique key for the asset being processed.
        data (dict): Metadata about the asset (e.g., title, ingestion type, API details).
        asset_row (pd.DataFrame): A single-row DataFrame containing the latest asset data.
        db_engine (sqlalchemy.engine.Engine): The database engine for the connection.
        secrets_file_path (str): Path to the secrets.json file for API authentication.

    Returns:
        None: This function renders the API ingestion form directly in the Streamlit app.
    """
    current_date = datetime.now()
    missing_dates = calculate_missing_months(latest_valuation_date, current_date)

    # Display Last balance in DB:
    if not asset_row.empty and 'latest_asset_value' in asset_row.columns:
        val = asset_row['latest_asset_value'].iloc[0]
        if pd.isna(val):
            latest_balance = 0.0 # Default to 0.0 if NaN
        else:
            latest_balance = val
    else:
        latest_balance = 0.0 # Default if asset_row is empty or column is missing

    if isinstance(latest_balance, (float, int)):
        st.markdown(f"Last balance in DB: {latest_balance:,.2f}")
    else:
        st.markdown(f"Last balance in DB: {latest_balance}")

    # Preview button
    if st.button("Preview data from API", key=f"preview_api_{asset_id_key}"):
        try:
            df_api_data = lib_api.process_indexa(data['asset_id'], secrets_file_path, data['API_account'])
            if df_api_data is not None:
                df_api_data['valuation_date'] = pd.to_datetime(df_api_data['valuation_date']).dt.date
                df_missing_data = df_api_data[df_api_data['valuation_date'].isin(missing_dates)]
                if not df_missing_data.empty:
                    st.write("Data Preview from API:")
                    st.dataframe(df_missing_data)
                    # Store the data and missing dates in session state
                    st.session_state[f"api_data_{asset_id_key}"] = df_api_data
                    st.session_state[f"api_missing_dates_{asset_id_key}"] = missing_dates
                else:
                    st.warning("No data found from API for the missing months.")
            else:
                st.error("Failed to retrieve data from API.")
        except Exception as e:
            st.error(f"Error previewing API data: {e}")

    # Load button
    if st.button("Ingest data from API to DB", key=f"ingest_api_{asset_id_key}"):
        try:
            # Retrieve data from session state
            if f"api_data_{asset_id_key}" in st.session_state and f"api_missing_dates_{asset_id_key}" in st.session_state:
                df_api_data = st.session_state[f"api_data_{asset_id_key}"]
                missing_dates = st.session_state[f"api_missing_dates_{asset_id_key}"]
                df_missing_data = df_api_data[df_api_data['valuation_date'].isin(missing_dates)]

                if not df_missing_data.empty:
                    # Format the valuation_date to 'YYYY-MM-DD HH:MM:SS'
                    df_missing_data['valuation_date'] = pd.to_datetime(df_missing_data['valuation_date']).dt.strftime('%Y-%m-%d %H:%M:%S')

                    load_manual_data_and_refresh(df_missing_data, data["asset_id"], db_engine)
                    # Clear session state
                    del st.session_state[f"api_data_{asset_id_key}"]
                    del st.session_state[f"api_missing_dates_{asset_id_key}"]
                else:
                    st.warning("No data found from API for the missing months.")
            else:
                st.error("Preview data first.")
        except Exception as e:
            st.error(f"Error ingesting API data: {e}")

####################################################################################################

def render_suggested_manual_ingestion_form(asset_id_key, data, asset_row, latest_valuation_date, db_engine):
    """
    Render the form for suggested manual data ingestion for a specific asset (e.g., mortgage).
    Suggests the next month's balance based on the last balance and a fixed monthly instalment.

    Args:
        asset_id_key (str): The unique key for the asset being processed.
        data (dict): Metadata about the asset (e.g., title, ingestion type).
        asset_row (pd.DataFrame): A single-row DataFrame containing the latest asset data.
        latest_valuation_date (datetime.date or pd.Timestamp): The latest valuation date for the asset.
        db_engine (sqlalchemy.engine.Engine): The database engine for the connection.

    Returns:
        None: This function renders the suggested manual ingestion form directly in the Streamlit app.
    """
    current_date = datetime.now()
    missing_dates = calculate_missing_months(latest_valuation_date, current_date)

    # Initialize session state for input values if not already present
    if f'input_values_{asset_id_key}' not in st.session_state:
        st.session_state[f'input_values_{asset_id_key}'] = {}

    with st.form(key=f'suggested_manual_ingestion_{asset_id_key}'):
        # Display the last balance
        if not asset_row.empty and 'latest_asset_value' in asset_row.columns:
            val = asset_row['latest_asset_value'].iloc[0]
            if pd.isna(val):
                latest_balance = 0.0 # Default to 0.0 if NaN
            else:
                latest_balance = val
        else:
            latest_balance = 0.0 # Default if asset_row is empty or column is missing
        
        # Display last uploaded balance date
        last_uploaded_date_str = latest_valuation_date.strftime('%Y-%m-%d') if pd.notna(latest_valuation_date) else "No Date"
        st.markdown(f"Last uploaded balance: {last_uploaded_date_str}")
        
        if isinstance(latest_balance, (float, int)):
            st.markdown(f"Last balance in DB: {latest_balance:,.2f}")
        else:
            st.markdown(f"Last balance in DB: {latest_balance}")

        input_data = {}
        current_calculated_balance = latest_balance 

        for missing_date in missing_dates:
            date_key = missing_date.strftime('%Y-%m-%d')
            widget_key = f"value_{asset_id_key}_{date_key}"

            suggested_balance = current_calculated_balance - config.mortgage_monthly_instalment

            if widget_key not in st.session_state[f'input_values_{asset_id_key}']:
                st.session_state[f'input_values_{asset_id_key}'][widget_key] = f"{suggested_balance:.2f}"

            formatted_instalment = f"{config.mortgage_monthly_instalment:,.2f}"

            input_value = st.text_input(
                label=f"Value for {date_key} (Instalment: {formatted_instalment} | Suggested: {suggested_balance:,.2f})",
                value=st.session_state[f'input_values_{asset_id_key}'][widget_key],
                key=widget_key,
            )
            input_data[missing_date] = input_value

            try:
                current_calculated_balance = float(input_value)
            except ValueError:
                current_calculated_balance = suggested_balance 
        
        if st.form_submit_button("Ingest data to DB"):
            df_data = []
            valid_inputs = True

            for date, value in input_data.items():
                if value:
                    try:
                        numeric_value = float(value)
                        df_data.append({
                            'asset_id': data['asset_id'],
                            'current_value': numeric_value,
                            'valuation_date': date.strftime('%Y-%m-%d %H:%M:%S')
                        })
                    except ValueError:
                        st.error(f"Invalid numeric value for {date.strftime('%Y-%m-%d')}: '{value}'")
                        valid_inputs = False
                        break
                else:
                    st.warning(f"No value entered for {date.strftime('%Y-%m-%d')}.")
                    valid_inputs = False
                    break

            if valid_inputs and df_data:
                df = pd.DataFrame(df_data)
                load_manual_data_and_refresh(df, data["asset_id"], db_engine)
                del st.session_state[f'input_values_{asset_id_key}']
            elif not valid_inputs:
                pass
            else:
                st.warning("No data entered for ingestion.")

####################################################################################################