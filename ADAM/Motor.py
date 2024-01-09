#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import os
from ADAM.import_mods import import_user_imports
from ADAM.respond import bot_speak
from ADAM.response import command_response


def execute_commands(audio_text, user_profile, bot_profile):
    from Mods.IMPORTS import COMMANDS, MAP
    command_executed_tags = []  # List to accumulate executed command tags
    for command_dict in COMMANDS:
        commands = command_dict['commands']
        action = command_dict['action']
        print(f'Checking against commands: {commands}')
        if any(cmd in audio_text for cmd in commands):
            print(f'Matched command: {commands}, triggering action: {action.__name__}')
            args = MAP.get(action, ())
            # Accumulate the executed command tags
            command_executed_tags.append(f"[{commands[0]}]")
            # Communicate the action being performed to the user
            final_input = f"the user said '{audio_text}' in your own words, tell them you will perform these actions {', '.join(command_executed_tags)}"
            response = command_response(final_input, user_profile, bot_profile)
            bot_speak(bot_profile, response)
            action(*args)
        # Check CUSTOM_COMMANDS from the user profile
        user_commands, user_map = import_user_imports(user_profile)

        for command_dict in user_commands:
            commands = command_dict['commands']
            action = command_dict['action']
            print(f'Checking against custom commands: {commands}')
            if any(cmd in audio_text for cmd in commands):
                print(f'Matched command: {commands}, triggering action: {action.__name__}')
                args = user_map.get(action, ())
                # Accumulate the executed command tags
                command_executed_tags.append(f"[{commands[0]}]")
                # Communicate the action being performed to the user
                final_input = f"the user said '{audio_text}' in your own words, tell them you will perform these actions {', '.join(command_executed_tags)}"
                response = command_response(final_input, user_profile, bot_profile)
                bot_speak(bot_profile, response)
                action(*args)
    # If no basic command was executed, proceed with the shortcut check
    shortcuts_dir = './Mods/COMMANDS'
    if not command_or_shortcut_executed(audio_text, shortcuts_dir):
        # Check if the input contains a folder path or name
        matched_paths = check_for_folder_or_file(shortcuts_dir, audio_text)
        if matched_paths:
            # Communicate the action being performed to the user
            command_executed_tags.append(f"Opening/Executing: {', '.join(matched_paths)}")
            final_input = f"the user said '{audio_text}' in your own words, tell them you will perform these actions {', '.join(command_executed_tags)}"
            response = command_response(final_input, user_profile, bot_profile)
            bot_speak(bot_profile, response)
            return True
    return bool(command_executed_tags)
def command_or_shortcut_executed(audio_text, current_dir):
    # Ensure the SHORTCUTS directory exists
    os.makedirs(current_dir, exist_ok=True)
    # List to accumulate matched entries
    matched_entries = []
    # Recursively check each file/folder in the current directory
    for entry in os.listdir(current_dir):
        entry_path = os.path.join(current_dir, entry)
        entry_name_without_extension = os.path.splitext(entry)[0].lower()
        if entry_name_without_extension in audio_text.lower():
            # Accumulate the matched entry
            matched_entries.append(entry_path)
    # Open the directories or execute the files for all matched entries
    for matched_entry in matched_entries:
        os.startfile(matched_entry)
        print(f"Executed or opened: {matched_entry}")
    return bool(matched_entries)  # Return True if at least one entry was matched
def check_for_folder_or_file(base_dir, audio_text):
    # List to accumulate matched paths
    matched_paths = []
    # Recursively check for folder or file names mentioned in the audio text within the base directory
    for entry in os.listdir(base_dir):
        entry_path = os.path.join(base_dir, entry)
        # Check if entry is in the audio text
        if entry.lower() in audio_text.lower():
            # Accumulate the matched path
            matched_paths.append(entry_path)
        # If it's a directory, search inside it recursively
        if os.path.isdir(entry_path):
            matched_path = check_for_folder_or_file(entry_path, audio_text)
            # Accumulate the matched path if found inside the directory
            if matched_path:
                matched_paths.append(matched_path)
    return matched_paths  # Return the list of matched paths