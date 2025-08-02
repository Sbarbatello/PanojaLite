# streamlit_app.py

#############################################
#     INITIALISATION
#############################################

import sys
import os
import streamlit as st
import pandas as pd
import altair as alt
from datetime import timedelta, datetime
from streamlit_extras.stylable_container import stylable_container
import base64

# Add the root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from libraries import lib_db, lib_streamlit  # Import the database library dynamically
from scripts import data_ingestion_script
from config import config

# Define the path to 'secrets.json'
config_dir = os.path.join(project_root, 'config')
secrets_file_path = os.path.join(config_dir, 'secrets.json')

# Create an absolute path to the logo file for robust deployment
logo_path = os.path.join(project_root, 'streamlit', 'images', 'logo01.png')

# --- DATABASE CONNECTION ---
# Determine the active environment.
# Priority 1: Check for a Streamlit Secret (for cloud deployment).
# Priority 2: Fall back to the value in config.py (for local development).
active_env = st.secrets.get("ACTIVE_DB_ENV", config.ACTIVE_DB_ENV)

# Create the database engine once.
db_engine = lib_db.get_db_engine(active_env)

# If the connection fails, stop the app gracefully.
if db_engine is None:
    st.error(f"Fatal Error: Could not create a database engine for the '{active_env.upper()}' environment. Please check your configuration and secrets.")
    st.stop()

#############################################
#     FORMATTING
#############################################

# Set page config
st.set_page_config(page_title="PanojaLite", layout="wide") # Set to wide format

# Create columns for the title and the database name
title_col, db_name_col = st.columns([0.8, 0.3]) # Adjust ratios as needed

with title_col:
    st.title("PanojaLite Dashboard")

with db_name_col:
    # Set the color based on the active environment for visual feedback
    # Green for production (live data), Blue for development (safe test data)
    env_color = "#28a745" if active_env == "prod" else "#007bff"

    # Use st.markdown to display the environment name.
    # The 'unsafe_allow_html=True' lets us use custom styling.
    st.markdown(
        f"""
        <div style='text-align: right; font-size: 0.9em; color: #aaa; padding-top: 10px;'>
            Connected to: <b style='color:{env_color}; font-weight:bold;'>{active_env.upper()}</b>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Define a global mapping for currency symbols
currency_symbols = {
    "EUR": "€",
    "GBP": "£",
    "USD": "$"
}

#############################################
#     GLOBAL SIDEBAR (for page selection only)
#############################################

with st.sidebar:
    # Place the logo at the very top of the sidebar
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; margin-bottom: 10px;">
            <img src="data:image/png;base64,{base64.b64encode(open(logo_path, 'rb').read()).decode()}"  
                alt="Company Logo" 
                width="150">
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page_selector = st.sidebar.selectbox(
        "**Page selection**",
        ("Summary Dashboard", "Breakdown Dashboard", "Data Ingestion")
    )
    st.divider()

    st.header(page_selector)

    st.divider()

#############################################
#############################################
#     PAGE: Summary Dashboard
#############################################
#############################################

def summary_dash():

    #############################################
    #     SIDEBAR
    #############################################
    
    with st.sidebar:
        
        # This flag will control whether the mortgage is included in calculations
        include_mortgage_flag = st.checkbox(
            "Include Mortgage in Net Worth",
            value=True, # Default to true, as it's the more comprehensive view
            help="If checked, house value will be adjusted by mortgage balance (House - Mortgage). If unchecked, mortgage will be ignored."
        )
        
        st.divider()

        owner_selection = st.radio(
            "**Owner selection**",
            ("All", "Carlos", "Sara", "Common"),
            index=0 # Set "All" as default (index 0)
        )
            
        st.divider()

        asset_type_focus = st.radio(
            "**Asset Type Focus**",
            ("All Assets", "Non-Cash Assets", "Cash Accounts Only"),
            index=0 # Set "All Assets" as default (index 0)
        )
            
        st.divider()
        st.divider()

        target_currency = st.selectbox(
            "**Display Currency**",
            ("EUR", "GBP", "USD"), # Options for currency display
            index=0 # Default to EUR
        )

        st.divider()

    #############################################
    #     BODY
    #############################################

    # Load data (using cached function)
    df = lib_streamlit.get_asset_history(db_engine)

    # prepare_asset_data_for_analysis function will handle filtering based on include_mortgage_flag, owner_selection and asset_type_focus
    processed_df = lib_streamlit.prepare_asset_data_for_analysis(df, include_mortgage_flag, owner_selection, asset_type_focus, target_currency, secrets_file_path)

    if processed_df is not None:  # Check if data loading was successful
        asset_order = lib_streamlit.get_asset_order(db_engine)  # Get asset order
        if asset_order is not None:  # Check if asset_order retrieval was successful
            # Summary
            # Calculate Summary Metrics
            metrics = lib_streamlit.calculate_summary_metrics(processed_df)

            st.divider()

            # Get the correct currency symbol
            display_symbol = currency_symbols.get(target_currency, "€") # Default to Euro if not found

            # Display Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total value of assets", f"{display_symbol}{metrics['total_asset_value']:,.2f}")
                display_month = metrics['current_month_for_calc'].strftime('%B %Y')
                st.markdown(f"<p style='font-size: 0.8em; color: #888; margin-top: -15px;'>{display_month}</p>", unsafe_allow_html=True)
            with col2:
                st.metric("Monthly change", f"{display_symbol}{metrics['monthly_change_euro']:,.2f}", f"{metrics['monthly_change_percent']:+.2f}%")
            with col3:
                st.metric(
                "12-month change",
                f"{display_symbol}{metrics['annual_change_euro']:,.2f}" if metrics['annual_change_euro'] != 'N/A' else "N/A",
                f"{metrics['annual_change_percent']:+.2f}%" if metrics['annual_change_percent'] != 'N/A' else "N/A"
                )
                
            st.divider()

            # Display Top and Bottom Assets in Columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Top 3 Performing Assets (Month)")
                for row in metrics['top_3_monthly'].to_dict('records'):
                    change_percent = row['change_percent']
                    color = "green" if change_percent >= 0 else "red"
                    st.write(f"<div style='margin-bottom: 10px;'>{lib_streamlit.apply_labelling_assets(row['asset_name'])}<br><span style='font-size: 1.2em; color:{color}'>{change_percent:+.2f}%</span></div>", unsafe_allow_html=True)

                st.subheader("Top 3 Performing Assets (12-Month)")
                for row in metrics['top_3_annual'].to_dict('records'):
                    change_percent = row['change_percent']
                    color = "green" if change_percent >= 0 else "red"
                    st.write(f"<div style='margin-bottom: 10px;'>{lib_streamlit.apply_labelling_assets(row['asset_name'])}<br><span style='font-size: 1.2em; color:{color}'>{change_percent:+.2f}%</span></div>", unsafe_allow_html=True)


            with col2:
                st.subheader("Bottom 3 Performing Assets (Month)")
                for row in metrics['bottom_3_monthly'].to_dict('records'):
                    change_percent = row['change_percent']
                    color = "green" if change_percent >= 0 else "red"
                    st.write(f"<div style='margin-bottom: 10px;'>{lib_streamlit.apply_labelling_assets(row['asset_name'])}<br><span style='font-size: 1.2em; color:{color}'>{change_percent:+.2f}%</span></div>", unsafe_allow_html=True)

                st.subheader("Bottom 3 Performing Assets (12-Month)")
                for row in metrics['bottom_3_annual'].to_dict('records'):
                    change_percent = row['change_percent']
                    color = "green" if change_percent >= 0 else "red"
                    st.write(f"<div style='margin-bottom: 10px;'>{lib_streamlit.apply_labelling_assets(row['asset_name'])}<br><span style='font-size: 1.2em; color:{color}'>{change_percent:+.2f}%</span></div>", unsafe_allow_html=True)

            st.divider()

            # render area chart
            lib_streamlit.render_plotly_stacked_area_chart(processed_df, "Asset Value Evolution", height=800, asset_order=asset_order)

#############################################
#############################################
#     PAGE: Breakdown Dashboard
#############################################
#############################################

def breakdown_dash():

    #############################################
    #     HEADERS
    #############################################

    #############################################
    #     SIDEBAR
    #############################################
    
    with st.sidebar:
        
        # This flag will control whether the mortgage is included in calculations
        include_mortgage_flag = st.checkbox(
            "Include Mortgage in Net Worth",
            value=True, # Default to true, as it's the more comprehensive view
            help="If checked, house value will be adjusted by mortgage balance (House - Mortgage). If unchecked, mortgage will be ignored."
        )
        
        st.divider()

        owner_selection = st.radio(
            "**Owner selection**",
            ("All", "Carlos", "Sara", "Common"),
            index=0 # Set "All" as default (index 0)
        )
        
        st.divider()

        asset_type_focus = st.radio(
            "**Asset Type Focus**",
            ("All Assets", "Non-Cash Assets", "Cash Accounts Only"),
            index=0 # Set "All Assets" as default (index 0)
        )
            
        st.divider()
        st.divider()

        target_currency = st.selectbox(
            "**Display Currency**",
            ("EUR", "GBP", "USD"), # Options for currency display
            index=0 # Default to EUR
        )

        st.divider()

    #############################################
    #     BODY
    #############################################

    # Load data (using cached function)
    df = lib_streamlit.get_asset_history(db_engine)

    # prepare_asset_data_for_analysis function will handle filtering based on include_mortgage_flag, owner_selection and asset_type_focus
    processed_df = lib_streamlit.prepare_asset_data_for_analysis(df, include_mortgage_flag, owner_selection, asset_type_focus, target_currency, secrets_file_path)

    if processed_df is not None:  # Check if data loading was successful
            lib_streamlit.render_monthly_net_worth_waterfall(processed_df, "Monthly Net Worth Change", height=800)


#############################################
#############################################
#     PAGE: Data Ingestion
#############################################
#############################################

def data_ingestion():

    #############################################
    #     INITIALISATION
    #############################################
    # Check if asset data is available in session state
    if 'asset_data' not in st.session_state:
        st.session_state['asset_data'] = lib_streamlit.get_recent_asset_data(db_engine)

    #############################################
    #     SIDEBAR
    #############################################
    
    with st.sidebar:
        
        st.header("📥💰 Assets")

        if st.session_state['asset_data'] is not None:
            current_month = datetime.now().month

            # Iterate through each asset and display it in the sidebar
            for _, row in st.session_state['asset_data'].iterrows():
                asset_id = row['asset_id']
                asset_name = row['asset_name']
                latest_valuation_date = row['latest_valuation_date_date']  # Use the date-only column

                # Determine color code based on the valuation date
                if latest_valuation_date:
                    try:
                        if isinstance(latest_valuation_date, str):
                            date_obj = datetime.strptime(latest_valuation_date, '%Y-%m-%d')  # Correct parsing format
                        else:
                            date_obj = latest_valuation_date  # Assume it's already a date object

                        valuation_month = date_obj.month
                        color_code = "#38A169" if valuation_month == current_month else "#e37575"
                    except (ValueError, TypeError):
                        color_code = "#e37575"  # Handle parsing errors
                else:
                    color_code = "#e37575"  # Red if no date

                # Display each asset in a stylable container
                with stylable_container(
                    key=f"asset_sidebar_{asset_id}",
                    css_styles=f"""
                        {{
                            background-color: {color_code};
                            padding: 10px;
                            margin-bottom: 5px;
                            border-radius: 5px;
                            display: flex;
                            flex-direction: column;
                            min-height: 40px;
                        }}
                        .asset-name-date-container {{
                            display: flex;
                            flex-direction: column;
                            width: 100%;
                        }}
                        .asset-name {{
                            font-weight: bold;
                            font-size: 16px;
                            margin: 0;
                        }}
                        .asset-date {{
                            font-size: 14px;
                            margin: 0;
                            text-align: right;
                        }}
                    """
                ):
                    with st.container():
                        st.markdown("<div class='asset-name-date-container'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='asset-name'>Asset {asset_id}: {asset_name}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='asset-date'>{latest_valuation_date}</div>", unsafe_allow_html=True)  # Directly use the date
                        st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.error("Error loading asset data from database.")

    #############################################
    #     BODY
    #############################################

    # Asset information
    asset_info = data_ingestion_script.get_asset_info()
    # Sort assets by asset_id
    sorted_assets = sorted(asset_info.items(), key=lambda item: item[1]["asset_id"])

    col1, col2 = st.columns(2)
    # Main loop for displaying assets
    for i, (asset_id_key, data) in enumerate(sorted_assets):
        current_col = col1 if i % 2 == 0 else col2  # Determine column dynamically
        with current_col:
            # Calculate color code for the container border
            asset_row = st.session_state['asset_data'][st.session_state['asset_data']['asset_id'] == data['asset_id']]
            if not asset_row.empty:
                latest_valuation_date = asset_row['latest_valuation_date_date'].iloc[0]
                if latest_valuation_date:
                    try:
                        if isinstance(latest_valuation_date, str):
                            date_obj = datetime.strptime(latest_valuation_date, '%Y-%m-%d')
                        else:
                            date_obj = latest_valuation_date
                        valuation_month = date_obj.month
                        color_code = "#38A169" if valuation_month == datetime.now().month else "#e37575"
                    except (ValueError, TypeError):
                        color_code = "#e37575"  # Handle parsing errors
                else:
                    color_code = "#e37575"  # Red if no date
            else:
                color_code = "#e37575" #Red if no asset row.

            with stylable_container(
                key=f"asset_container_{i}",  # Unique key for each container
                css_styles=f"""
                    {{
                        border: 1px solid {color_code};
                        padding: 10px;
                        margin-bottom: 10px;
                        border-radius: 5px;
                    }}
                    /* Target the inner input element of the file uploader and make it 100% width */
                    .st-file-uploader input[type="file"] {{
                        width: 100%;
                    }}

                    /* Style for the status message container */
                    .status-message-container {{
                        padding: 5px;
                        margin-top: 5px;
                        border: 1px solid #ccc; /* Optional border */
                        border-radius: 3px;
                    }}
                    div.last-uploaded-balance {{ /* More specific selector */
                        font-size: 18px;
                        margin: 10px 0;
                        font-weight: bold;
                        color: {color_code};
                    }}
                """,
            ):
                with st.container():
                    st.subheader(f"Asset {data['asset_id']}: {data['title']}")

                    # Find the latest_valuation_date for the current asset
                    asset_row = st.session_state['asset_data'][st.session_state['asset_data']['asset_id'] == data['asset_id']]
                    if not asset_row.empty:
                        last_uploaded_date = asset_row['latest_valuation_date'].iloc[0].strftime('%Y-%m-%d') if pd.notna(asset_row['latest_valuation_date'].iloc[0]) else "No Date"
                        st.markdown(f"<div class='last-uploaded-balance' style='color: {color_code};'>Last uploaded balance: {last_uploaded_date}</div>", unsafe_allow_html=True)

                ##################################################

                # Pass last_uploaded_date to the appropriate rendering function
                if data["ingestion_type"] == "manual":
                    data_ingestion_script.render_manual_ingestion_form(asset_id_key, data, asset_row, latest_valuation_date, db_engine)
                elif data["ingestion_type"] == "file":
                    data_ingestion_script.render_file_ingestion_form(asset_id_key, data, asset_row, db_engine)
                elif data["ingestion_type"] == "api":
                    data_ingestion_script.render_api_ingestion_form(asset_id_key, data, asset_row, latest_valuation_date, db_engine, secrets_file_path)
                elif data["ingestion_type"] == "suggested_manual":
                    data_ingestion_script.render_suggested_manual_ingestion_form(asset_id_key, data, asset_row, latest_valuation_date, db_engine)


#############################################
#     PAGE MAPPING
#############################################

page_functions = {
"Summary Dashboard": summary_dash,
"Breakdown Dashboard": breakdown_dash,
"Data Ingestion": data_ingestion,
}

#############################################
#     PAGE/FUNCTION SELECTOR AND HEADER
#############################################

if page_selector in page_functions:
    page_functions[page_selector]()
else:
    st.write("Page not found")

#############################################
#     FOOTER
#############################################    
st.divider()
foot1, foot2, foot3 = st.columns([1,1,1])

with foot1:
    st.markdown("Version 1.0")
with foot2:
    st.markdown("Carlos Canas")
with foot3:
    st.markdown("August 2025")