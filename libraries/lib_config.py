# lib_config.py
from cryptography.fernet import Fernet
import streamlit as st
import os
import json

# Define the file location for the key
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
file_location = os.path.join(project_root, 'config', 'secret.key')

####################################################################################################

def get_encryption_key(key_file=file_location):
    """
    Gets the encryption key.

    It first checks for a 'SECRET_KEY' in Streamlit's secrets manager (for cloud deployment).
    If not found, it falls back to reading the key from the local file system.

    Args:
        key_file (str): The local file path to the secret.key file.

    Returns:
        bytes: The encryption key as a bytes object.
    """
    # Priority 1: Check for a secret in the Streamlit Cloud environment
    if "SECRET_KEY" in st.secrets:
        print("Loading encryption key from Streamlit Cloud environment.")
        # Keys must be bytes, so we encode the string from secrets
        return st.secrets["SECRET_KEY"].encode()
    else:
        # Priority 2: Fallback to reading from the local file
        print("Loading encryption key from local file.")
        try:
            with open(key_file, 'rb') as key_in:
                return key_in.read()
        except FileNotFoundError:
            print(f"ERROR: Local key file not found at {key_file}. Please generate one.")
            return None

####################################################################################################

# Generate and save a key (only do this once, then reuse the key)
def generate_key(key_file=file_location):
    key = Fernet.generate_key()
    with open(key_file, 'wb') as key_out:
        key_out.write(key)
    print(f"Key saved to {key_file}")

####################################################################################################

# Encrypt the file
def encrypt_file(file_path, key_file=file_location):
    """Encrypts the file using the environment-aware key."""
    key = get_encryption_key(key_file)

    if key is None:
        print("ERROR: Cannot encrypt file without an encryption key.")
        return # Stop if we couldn't get a key

    fernet = Fernet(key)

    # Read the original file
    with open(file_path, 'rb') as file_in:
        original_data = file_in.read()
    
    # Encrypt the data
    encrypted_file = f"{file_path}.encrypted"
    encrypted_data = fernet.encrypt(original_data)
    
    # Save the encrypted file
    with open(encrypted_file, 'wb') as file_out:
        file_out.write(encrypted_data)
    print(f"File encrypted and saved as {encrypted_file}")

####################################################################################################

# Decrypt the file
def decrypt_file(encrypted_file_path, key_file=file_location):
    """Decrypts the file using the environment-aware key."""
    key = get_encryption_key(key_file)

    if key is None:
        print("ERROR: Cannot decrypt file without an encryption key.")
        return None # Stop if we couldn't get a key

    fernet = Fernet(key)

    # Read the encrypted file
    with open(encrypted_file_path, 'rb') as file_in:
        encrypted_data = file_in.read()

    # Decrypt the data
    decrypted_data = fernet.decrypt(encrypted_data)
    return decrypted_data

####################################################################################################

# Load the decrypted secrets.json as a dictionary
def load_secrets(encrypted_file_path, key_file=file_location):
    decrypted_data = decrypt_file(encrypted_file_path, key_file)
    secrets = json.loads(decrypted_data)
    return secrets

    # # Example usage
    # secrets_file = os.path.join(project_root, 'config', 'secrets.json')
    # encrypted_secrets_file = f"{secrets_file}.encrypted"
    # secrets = lib_config.load_secrets(encrypted_secrets_file)
    # print(secrets)

####################################################################################################