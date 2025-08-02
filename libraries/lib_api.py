# lib_api.py
import requests
from libraries.lib_db import read_secrets_from_json
import pandas as pd
from libraries.lib_config import load_secrets
from datetime import datetime, timedelta
import streamlit as st

####################################################################################################

def get_indexa_balance_performance(json_file_path, account):
    # Read connection details from the JSON file
    # Load and print secrets from the encrypted file
    encrypted_secrets_file = f"{json_file_path}.encrypted"
    secrets = load_secrets(encrypted_secrets_file)

    if secrets is None:
        return None

    # Retrieve the token for the specified account
    indexa_token = secrets.get('indexa', {}).get(account, {}).get('token')
    if not indexa_token:
        return f"Error: Token for Indexa account {account} not found."
    
    # Construct the database URL for Indexa's API
    url_base = "https://api.indexacapital.com"
    url_account = f"{url_base}/accounts/{account}/performance"      
    
    # Construct the headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Cache-Control": "no-cache",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "X-Auth-Token": indexa_token
    }
    
    # Connect to API
    response = requests.get(url_account, headers=headers)
    response_data = response.json()
        
    if response.status_code == 200:
        # Extract total_amounts from the response data
        total_amounts = response_data.get('return', {}).get('total_amounts', {})
        return total_amounts
    else:
        error_message = f"Error: Unable to fetch data. Status Code: {response.status_code}"
        return error_message

####################################################################################################

def process_indexa(asset_id, json_file_path, account):
    try:
        indexa_balances = get_indexa_balance_performance(json_file_path, account)
    except Exception as e:
        print(f"Error fetching data for account {account}: {e}")
        return None  

    if not isinstance(indexa_balances, dict):
        print(f"Unexpected response format: {indexa_balances}")
        return None

    # Convert dictionary to DataFrame
    df_indexa = pd.DataFrame([
        {"asset_id": asset_id, "valuation_date": date, "current_value": amount}
        for date, amount in indexa_balances.items()
    ])

    # Filter to keep only records where the date corresponds to the first day of the month
    df_indexa_months = df_indexa[df_indexa["valuation_date"].str.endswith("01")]
    # Ensure 'valuation_date' is in datetime format, then convert to the desired string format
    df_indexa_months['valuation_date'] = pd.to_datetime(df_indexa_months['valuation_date']).dt.strftime('%Y-%m-%d %H:%M:%S')

    return df_indexa_months

####################################################################################################

####################################################################################################
####################################################################################################
# Cached functions
####################################################################################################
####################################################################################################

@st.cache_data(ttl=timedelta(days=1)) # Cache for 24 hours
def get_exchange_rates(secrets_file_path, base_currency="EUR"):
    """
    Fetches and caches exchange rates from ExchangeRate-API.com.

    Args:
        secrets_file_path (str): Path to the secrets.json file (used to find the encrypted version).
        base_currency (str): The base currency for the exchange rates (e.g., "EUR", "USD", "GBP").

    Returns:
        dict: A dictionary of exchange rates relative to the base_currency,
            or None if an error occurs.
            Example: {'EUR': 1.0, 'USD': 1.08, 'GBP': 0.85}
    """
    # Load secrets from the encrypted file
    encrypted_secrets_file = f"{secrets_file_path}.encrypted"
    secrets = load_secrets(encrypted_secrets_file)

    if secrets is None:
        st.error("Error: Could not load secrets for exchange rates.")
        return None

    api_key = secrets.get('exchangerate-api', {}).get('api_key')

    if not api_key:
        st.error("Error: ExchangeRate-API key not found in secrets.")
        return None

    # ExchangeRate-API.com endpoint
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}"

    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        if data.get('result') == 'success':
            conversion_rates = data.get('conversion_rates')
            if conversion_rates:
                # The API returns rates relative to the base_currency.
                # We want to return a dict where EUR is the implicit base for all rates.
                # So, if base_currency is EUR, we return conversion_rates directly.
                # If base_currency is not EUR, we convert all rates to be relative to EUR.

                if base_currency == "EUR":
                    return conversion_rates
                else:
                    # To convert all rates to be relative to EUR:
                    # rate_X_to_EUR = 1 / rate_EUR_to_X
                    # rate_X_to_Y = rate_X_to_EUR * rate_EUR_to_Y
                    # Or more simply: rate_X_to_Y = rate_Base_to_Y / rate_Base_to_X
                    
                    # Get the rate of EUR relative to the current base_currency
                    eur_rate_from_base = conversion_rates.get("EUR")
                    if eur_rate_from_base:
                        # Create a new dictionary where all rates are relative to EUR
                        eur_based_rates = {
                            currency: rate / eur_rate_from_base
                            for currency, rate in conversion_rates.items()
                        }
                        return eur_based_rates
                    else:
                        st.error(f"Error: EUR rate not found in API response when base currency is {base_currency}.")
                        return None
            else:
                st.error("Error: 'conversion_rates' not found in API response.")
                return None
        else:
            st.error(f"Error from ExchangeRate-API: {data.get('error-type', 'Unknown error')}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Network or API request error: {e}")
        return None
    except ValueError as e:
        st.error(f"Error parsing API response (not valid JSON): {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching exchange rates: {e}")
        return None

####################################################################################################
####################################################################################################
