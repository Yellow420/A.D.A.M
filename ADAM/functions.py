#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import os
from ADAM.respond import bot_speak

def list_available_commands(bot_profile, user_profile, BASE_DIR = "user_data"):
    mods_folder = 'Mods'  # Adjust this path based on your actual folder structure
    user_id = user_profile['id']
    user_profile_path = os.path.join(BASE_DIR, user_id)
    available_commands = {}
    # Check if the Mods folder exists
    if os.path.exists(mods_folder) and os.path.isdir(mods_folder):
        # Iterate over files in the Mods folder
        for mod_name in os.listdir(mods_folder):
            mod_path = os.path.join(mods_folder, mod_name)
            # Check if it's a directory and contains description.txt
            if os.path.isdir(mod_path) and 'description.txt' in os.listdir(mod_path):
                description_file = os.path.join(mod_path, 'description.txt')
                # Read the description from the file
                with open(description_file, 'r') as file:
                    description = file.read().strip()
                # Add the command and description to the available_commands dictionary
                available_commands[mod_name] = {'description': description}
    # Check if the user's profile folder exists
    if os.path.exists(user_profile_path) and os.path.isdir(user_profile_path):
        # Iterate over files in the user's profile folder
        for mod_name in os.listdir(user_profile_path):
            mod_path = os.path.join(user_profile_path, mod_name)
            # Check if it's a directory and contains description.txt
            if os.path.isdir(mod_path) and 'description.txt' in os.listdir(mod_path):
                description_file = os.path.join(mod_path, 'description.txt')
                # Read the description from the file
                with open(description_file, 'r') as file:
                    description = file.read().strip()
                # Add the command and description to the available_commands dictionary
                available_commands[mod_name] = {'description': description}
    if not available_commands:
        response = "I currently have no available commands."
        bot_speak(bot_profile, response)
        return
    response = "Here's what I can do:\n"
    for command, details in available_commands.items():
        response += f"- {command.capitalize()}: {details['description']}\n"
    bot_speak(bot_profile, response)
def toggle_lockdown(bot_profile):
    lockdown_file_path = './bot_profiles/lockdown'
    # Check if the lockdown file exists
    if os.path.exists(lockdown_file_path):
        # If it exists, remove it (unlock)
        os.remove(lockdown_file_path)
        response =("Lockdown deactivated.")
    else:
        # If it doesn't exist, create it (lock)
        open(lockdown_file_path, 'a').close()
        response =("Lockdown activated.")
    bot_speak(bot_profile, response)
def lock_bot(bot_profile):
    lock_file_path = os.path.join(f'./bot_profiles/{bot_profile["name"]}', 'lock')
    open(lock_file_path, 'a').close()  # Create an empty file named 'lock'
def unlock_bot(bot_profile):
    lock_file_path = os.path.join(f'./bot_profiles/{bot_profile["name"]}', 'lock')
    if os.path.exists(lock_file_path):
        os.remove(lock_file_path)  # Remove the 'lock' file if it exists
