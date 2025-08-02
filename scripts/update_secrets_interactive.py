# update_secrets_interactive.py
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
# This is the temporary, unencrypted file we will create
unencrypted_secrets_path = os.path.join(config_dir, 'secrets.json')
# This is the source and final destination for encrypted data
encrypted_secrets_path = f"{unencrypted_secrets_path}.encrypted"

# --- Step 1: Display Current Secrets ---
def display_current_secrets():
    """
    Decrypts and displays the current secrets from the encrypted file.
    Returns the secrets as a dictionary.
    """
    print("--- Current Encrypted Secrets ---")
    secrets = {}
    try:
        if os.path.exists(encrypted_secrets_path):
            secrets = lib_config.load_secrets(encrypted_secrets_path)
            print(json.dumps(secrets, indent=4))
        else:
            print("No existing encrypted secrets file found. A new one will be created.")
    except Exception as e:
        print(f"Could not read or decrypt secrets file. Error: {e}")
        print("Proceeding to create a new secrets file from your input.")
    finally:
        print("---------------------------------")
        return secrets

# --- Step 2: Get New Secrets from User ---
def get_new_secrets_from_user():
    """
    Prompts the user to paste new JSON content and returns it as a dictionary.
    """
    print("\nPaste the new JSON content to add or update.")
    print("This should be a valid JSON object, e.g., {\"new_key\": {\"sub_key\": \"value\"}}")
    print("On Windows, press Ctrl+Z then Enter to finish. On Mac/Linux, press Ctrl+D.")
    print("Paste content now:")
    try:
        new_content_str = sys.stdin.read()
        if not new_content_str.strip():
            print("\nNo input received. Exiting.")
            sys.exit(0)
        new_secrets_to_add = json.loads(new_content_str)
        if not isinstance(new_secrets_to_add, dict):
            raise ValueError("Pasted content is not a valid JSON object (dictionary).")
        return new_secrets_to_add
    except json.JSONDecodeError:
        print("\nError: The pasted content is not valid JSON. Please try again.")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)

# --- Step 3: Check for Existing Unencrypted File ---
def check_for_existing_unencrypted_file():
    """
    Checks if an unencrypted secrets file already exists and exits if it does.
    """
    if os.path.exists(unencrypted_secrets_path):
        print(f"\nERROR: An unencrypted '{unencrypted_secrets_path}' already exists.")
        print("Please remove or rename it before running this script to avoid overwriting.")
        sys.exit(1)

# --- Step 4: Write and Encrypt Secrets ---
def write_and_encrypt_secrets(merged_secrets):
    """
    Writes the merged secrets to a temporary file and encrypts it.
    Does NOT clean up the temporary file.
    """
    try:
        # Write the merged secrets to the temporary unencrypted file
        with open(unencrypted_secrets_path, 'w') as f:
            json.dump(merged_secrets, f, indent=4)
        print(f"Temporarily created '{unencrypted_secrets_path}' with updated content.")

        # Encrypt the file, overwriting the old encrypted file
        lib_config.encrypt_file(unencrypted_secrets_path)
        print(f"Successfully created new encrypted file: '{encrypted_secrets_path}'")
        return True # Indicate success
    except Exception as e:
        print(f"An error occurred during writing/encryption: {e}")
        return False # Indicate failure

# --- Step 5: Clean Up Unencrypted File ---
def cleanup_unencrypted_file():
    """
    Asks for user confirmation and then deletes the unencrypted secrets file.
    """
    if os.path.exists(unencrypted_secrets_path):
        print("\nAn unencrypted 'secrets.json' file was created for this process.")
        confirm = input("Do you want to delete this file now? (y/n): ").lower().strip()
        if confirm == 'y':
            os.remove(unencrypted_secrets_path)
            print(f"Cleaned up and removed temporary file: '{unencrypted_secrets_path}'")
        else:
            print(f"Cleanup skipped. Please manually delete '{unencrypted_secrets_path}' later.")

######################################################################
def main():
    """
    Main function to orchestrate the interactive secrets update process.
    """
    # Step 1: Display current secrets and get them
    current_secrets = display_current_secrets()

    # Step 2: Get new secrets from the user
    new_secrets = get_new_secrets_from_user()

    # Step 3: Safety check for existing unencrypted file
    check_for_existing_unencrypted_file()

    # Merge the old and new secrets
    current_secrets.update(new_secrets)
    print("\nSecrets merged successfully.")

    # Step 4: Write the new secrets and encrypt
    success = write_and_encrypt_secrets(current_secrets)

    # Step 5: Clean up the temporary file if the previous step was successful
    if success:
        cleanup_unencrypted_file()

    print("\nSecrets update process complete.")

if __name__ == "__main__":
    main()