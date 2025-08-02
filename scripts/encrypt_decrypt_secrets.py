# encrypt_decrypt_secrets.py
import sys
import os
import json

# Add the root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from libraries import lib_config

# --- Setup Paths ---
config_dir = os.path.join(project_root, 'config')
unencrypted_secrets_path = os.path.join(config_dir, 'secrets.json')
encrypted_secrets_path = f"{unencrypted_secrets_path}.encrypted"

# --- Option 1: Decrypt ---
def decrypt_secrets_file():
    """
    Decrypts the encrypted secrets file and saves it as a new, unencrypted secrets.json.
    """
    print(f"\n--- Decrypting '{encrypted_secrets_path}' ---")

    # Safety check: ensure encrypted file exists
    if not os.path.exists(encrypted_secrets_path):
        print(f"ERROR: Encrypted file not found at '{encrypted_secrets_path}'. Cannot decrypt.")
        return

    # Safety check: ensure unencrypted file does NOT exist to prevent overwriting
    if os.path.exists(unencrypted_secrets_path):
        print(f"ERROR: An unencrypted '{unencrypted_secrets_path}' already exists.")
        print("Please remove or rename it before running decryption.")
        return

    try:
        secrets = lib_config.load_secrets(encrypted_secrets_path)
        with open(unencrypted_secrets_path, 'w') as f:
            json.dump(secrets, f, indent=4)
        print(f"Success! Decrypted content saved to: '{unencrypted_secrets_path}'")
        print("\nWARNING: This file contains unencrypted sensitive information.")
    except Exception as e:
        print(f"An error occurred during decryption: {e}")

# --- Option 2: Encrypt ---
def encrypt_secrets_file():
    """
    Encrypts an existing unencrypted secrets.json file, overwriting the encrypted version.
    """
    print(f"\n--- Encrypting '{unencrypted_secrets_path}' ---")

    # Safety check: ensure unencrypted file exists to be encrypted
    if not os.path.exists(unencrypted_secrets_path):
        print(f"ERROR: Unencrypted file not found at '{unencrypted_secrets_path}'. Cannot encrypt.")
        return

    try:
        lib_config.encrypt_file(unencrypted_secrets_path)
        print(f"Success! Encrypted file updated: '{encrypted_secrets_path}'")
        
        # Ask for cleanup
        confirm = input("Do you want to delete the unencrypted 'secrets.json' file now? (y/n): ").lower().strip()
        if confirm == 'y':
            os.remove(unencrypted_secrets_path)
            print(f"Cleaned up and removed unencrypted file: '{unencrypted_secrets_path}'")
        else:
            print(f"Cleanup skipped. Please manually delete '{unencrypted_secrets_path}' later.")
            
    except Exception as e:
        print(f"An error occurred during encryption: {e}")

# --- Main Orchestration Function ---
def main():
    """
    Main function to orchestrate the interactive secrets management process.
    """
    print("Secrets Management Utility")
    print("--------------------------")
    print("1. Decrypt 'secrets.json.encrypted' to view/edit 'secrets.json'")
    print("2. Encrypt 'secrets.json' to update 'secrets.json.encrypted'")

    choice = input("Please choose an option (1 or 2): ").strip()

    if choice == '1':
        decrypt_secrets_file()
    elif choice == '2':
        encrypt_secrets_file()
    else:
        print("Invalid choice. Please run the script again and enter 1 or 2.")

    print("\nProcess complete.")

if __name__ == "__main__":
    main()