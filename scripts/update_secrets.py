# update_secrets.py
import sys
import os
import json
from cryptography.fernet import Fernet

# Add the root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)
from libraries import lib_config

# Define the path to 'secrets.json'
config_dir = os.path.join(project_root, 'config')
secrets_file_path = os.path.join(config_dir, 'secrets.json')
encrypted_secrets_file_path = f"{secrets_file_path}.encrypted"

print(f"Attempting to encrypt: {secrets_file_path}")

try:
    # 4. Re-encrypt the secrets.json file
    lib_config.encrypt_file(secrets_file_path)
    print("Secrets re-encrypted successfully.")

except Exception as e:
    print(f"An error occurred during secrets update: {e}")
    print("Please ensure 'config/secret.key' exists and is valid, and that 'config/secrets.json.encrypted' is present.")

