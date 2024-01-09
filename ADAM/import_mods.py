#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import ast
import os
import importlib.util
from ADAM.Temporal import get_voice_input
from ADAM.respond import bot_speak
def extract_imports(content):
    tree = ast.parse(content)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            imports.append(f"from {node.module} import {', '.join(alias.name for alias in node.names)}")
    return '\n'.join(imports)
def merge_imports(mod_path, bot_profile, destination_path):
    try:
        # Read the content of the source IMPORTS.py
        with open(mod_path, 'r') as source_file:
            source_content = source_file.read()
        # Read the content of the destination IMPORTS.py
        with open(destination_path, 'r') as dest_file:
            dest_content = dest_file.read()
        # Extract imports, COMMANDS, and MAP from the source content
        source_imports = extract_imports(source_content)
        source_commands = source_content.split("COMMANDS = ")[1].split("\n\n")[0]
        source_map = source_content.split("MAP = ")[1]
        # Extract existing imports from the destination content
        dest_imports = extract_imports(dest_content)
        # Combine the imports from both files
        merged_imports = f"{source_imports}\n{dest_imports}"
        # Extract existing COMMANDS and MAP from the destination content
        dest_commands = dest_content.split("COMMANDS = ")[1].split("\n\n")[0]
        dest_map = dest_content.split("MAP = ")[1]
        # Combine the COMMANDS and MAP from both files
        merged_commands = f"{dest_commands}\n{source_commands}"
        merged_map = f"{dest_map}\n{source_map}"
        # Combine all parts and write the merged content back to the destination IMPORTS.py
        merged_content = f"{merged_imports}COMMANDS = {merged_commands}\n\nMAP = {merged_map}"
        with open(destination_path, 'w') as dest_file:
            dest_file.write(merged_content)
        # Merge descriptions.txt
        source_description_path = os.path.join(mod_path, 'description.txt')
        dest_description_path = os.path.join(os.path.dirname(destination_path), 'descriptions.txt')
        merge_descriptions(source_description_path, dest_description_path)
        a_prompt = ("Imports, COMMANDS, and MAP merged successfully.")
        bot_speak(bot_profile, a_prompt)
    except Exception as e:
        b_prompt = (f"Error merging imports, COMMANDS, and MAP: {e}")
        bot_speak(bot_profile, b_prompt)
def merge_descriptions(source_path, dest_path):
    try:
        # Read the content of the source description.txt
        with open(source_path, 'r') as source_file:
            source_description = source_file.read()
        # Read the content of the destination descriptions.txt
        dest_content = ""
        if os.path.exists(dest_path):
            with open(dest_path, 'r') as dest_file:
                dest_content = dest_file.read()
        # Append the source description to the destination content
        merged_content = f"{dest_content}\n\n{source_description}"
        # Write the merged content back to the destination descriptions.txt
        with open(dest_path, 'w') as dest_file:
            dest_file.write(merged_content)
    except Exception as e:
        print(f"Error merging descriptions: {e}")
def import_new_mod(mod_path, bot_profile, user_profile):
    # Read the description from description.txt
    description_path = os.path.join(mod_path, 'description.txt')
    with open(description_path, 'r') as desc_file:
        description = desc_file.read()
    # Ask the user if the mod is for everyone
    prompt = f"{description}\nIs this mod for everyone? (yes/no): "
    while True:
        user_input = get_voice_input(bot_profile, prompt)
        if "yes" in user_input.lower():
            is_for_everyone = True
            break
        elif "no" in user_input.lower():
            is_for_everyone = False
            break
        else:
            b_prompt =("Invalid response. Please answer with 'yes' or 'no'.")
            bot_speak(bot_profile, b_prompt)
    # Determine the target IMPORTS.py based on user preference
    destination_path = 'IMPORTS.py' if is_for_everyone else os.path.join(user_profile['id'], 'USER_IMPORTS.py')
    merge_imports(mod_path, bot_profile, destination_path)
    a_prompt = (f"Mod has been imported{' globally' if is_for_everyone else ' for the user'}.")
    bot_speak(bot_profile, a_prompt)
def install_mod(bot_profile, user_profile):
    prompt = "What mod would you like to install?"
    while True:
        # Get user input through voice
        user_input = get_voice_input(bot_profile, prompt)
        # Check if any folder names in Mods folder match the user input
        mods_folder_path = "./Mods"
        matching_folders = [folder for folder in os.listdir(mods_folder_path) if folder.lower() in user_input.lower()]
        if matching_folders:
            # If there is a match, install the mod
            mod_path = os.path.join(mods_folder_path, matching_folders[0])
            import_new_mod(mod_path, bot_profile, user_profile)
            break
        else:
            # If no match, ask if the user wants to try again
            try_again_prompt = "No matching mod found. Would you like to try again? (yes/no)"
            try_again_input = get_voice_input(bot_profile, try_again_prompt)
            if "no" in try_again_input.lower():
                a_prompt =("No mod installed. Exiting.")
                bot_speak(bot_profile, a_prompt)
                break
            elif "yes" in try_again_input.lower():
                continue
            else:
                b_prompt =("Invalid response. Exiting.")
                bot_speak(bot_profile, b_prompt)
                break
def import_user_imports(user_profile, BASE_DIR = "user_data"):
    user_id = user_profile['id']
    user_profile_path = os.path.join(BASE_DIR, user_id)
    imports_path = os.path.join(user_profile_path, 'IMPORTS.py')
    spec = importlib.util.spec_from_file_location("user_imports", imports_path)
    user_imports = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_imports)
    return user_imports.CUSTOM_COMMANDS, user_imports.CUSTOM_MAP