#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import os
import subprocess
import re
from ADAM.Temporal import get_voice_input
from ADAM.Hippocampus import get_bot
from ADAM.respond import bot_speak
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_python_script_polycoder(text: str) -> str:
    model_id = "VHellendoorn/PolyCoder-2.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

def generate_python_script_starcoder(text: str) -> str:
    model_id = "EleutherAI/starcoder-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

def generate_python_script_falcon(text: str) -> str:
    model_id = "microsoft/CodeXGLUE-fine-tuned_Falcon"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

def generate_python_script_codebert(text: str) -> str:
    model_id = "microsoft/CodeBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input_ids)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded_output

def generate_scripts(text: str) -> dict:
    results = {
        "Polycoder": generate_python_script_polycoder(text),
        "Starcoder": generate_python_script_starcoder(text),
        "Falcon": generate_python_script_falcon(text),
        "CodeBERT": generate_python_script_codebert(text),
    }
    return results

def listen_for_wake_word(wake_word):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print(f"Listening for the wake word '{wake_word}'...")
        try:
            audio_data = recognizer.listen(source, timeout=10)
            user_input = recognizer.recognize_google(audio_data).lower()
            print(f"Detected: {user_input}")

            if wake_word in user_input:
                return True
            else:
                print("Wake word not detected.")
                return False

        except sr.UnknownValueError:
            print("Speech recognition could not understand audio.")
            return False
        except sr.RequestError as e:
            print(f"Error with the speech recognition service: {e}")
            return False


def extract_script(text):
    # Define the pattern to match text between "***" delimiters
    pattern = r"\*\*\*(.*?)\*\*\*"

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Assuming the first match is the explanation and the second match is the script
    explanation = matches[0] if len(matches) > 0 else ""
    script = matches[1] if len(matches) > 1 else ""

    # If script is not found, return only the explanation
    if not script:
        return explanation.strip()

    return explanation.strip(), script.strip()


def update_script(bot_profile, user_input, mod_folder_path):
    from ADAM.response import response
    context = bot_profile['history']
    knowledge = generate_scripts(user_input)
    instruction = f"You are Scriptor your only purpose is to write Python scripts the code will be extracted directly from you're response you should structure you're response like this (explaination ***python script***) divide the script from everything else with *** so that it can be extracted. This is the request {user_input}"

    # Get the list of .py files in the mod_folder_path
    python_files = [f for f in os.listdir(mod_folder_path) if f.endswith('.py')]

    # If there are .py files, find the most recently modified one
    if python_files:
        most_recent_file = max(python_files, key=lambda f: os.path.getmtime(os.path.join(mod_folder_path, f)))
        file_path = os.path.join(mod_folder_path, most_recent_file)

        # Backup the existing file
        backup_folder = os.path.join(mod_folder_path, "backup")
        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)

        # Find the highest version number
        existing_backup_files = [f for f in os.listdir(backup_folder) if f.startswith(most_recent_file)]
        highest_version = max([int(f.split("_")[1]) for f in existing_backup_files], default=0)

        # Create the backup with the next version number
        backup_version = highest_version + 1
        backup_filename = f"{most_recent_file}_backup_{backup_version}.py"
        backup_path = os.path.join(backup_folder, backup_filename)

        # Copy the content of the existing file to the backup
        with open(file_path, 'r') as existing_file, open(backup_path, 'w') as backup_file:
            backup_file.write(existing_file.read())

        # Read the content of the most recently modified file
        with open(file_path, 'r') as file:
            dialog = file.read()

        # Extract script from the response
        text = response(instruction, context, knowledge, dialog)
        explanation, script = extract_script(text)

        # If no script is returned, save the response to a log file
        if not script:
            log_folder = os.path.join(mod_folder_path, "bot_log")
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)

            # Find the highest log number
            existing_log_files = [f for f in os.listdir(log_folder) if f.startswith("log")]
            highest_log_number = max([int(f.split("_")[1].split(".")[0]) for f in existing_log_files], default=0)

            # Create the log file with the next log number
            log_number = highest_log_number + 1
            log_filename = f"log_{log_number}.txt"
            log_path = os.path.join(log_folder, log_filename)

            # Save the response to the log file
            with open(log_path, 'w') as log_file:
                log_file.write(explanation)

            # Open the new log file
            os.system(log_path)
            bot_speak(bot_profile, explanation)
        else:
            # Overwrite the existing file with the new script
            with open(file_path, 'w') as file:
                file.write(script)

            # Open the new .py file
            os.system(file_path)
            # save the response to a log file
            log_folder = os.path.join(mod_folder_path, "bot_log")
            if not os.path.exists(log_folder):
                os.makedirs(log_folder)

            # Find the highest log number
            existing_log_files = [f for f in os.listdir(log_folder) if f.startswith("log")]
            highest_log_number = max([int(f.split("_")[1].split(".")[0]) for f in existing_log_files], default=0)

            # Create the log file with the next log number
            log_number = highest_log_number + 1
            log_filename = f"log_{log_number}.txt"
            log_path = os.path.join(log_folder, log_filename)

            # Save the response to the log file
            with open(log_path, 'w') as log_file:
                log_file.write(explanation)

            # Open the new log file
            os.system(log_path)
            bot_speak(bot_profile, explanation)
    else:
        explanation = "No Python files found in the specified folder."
        bot_speak(bot_profile, explanation)



def mod_builder():
    wake_word = "scriptor"
    bot_profile = get_bot(wake_word)
    # Step 1: Ask the user for the mod name
    mod_name = get_voice_input(bot_profile, "Hello I am Scriptor, what would you like to name you're Mod?")

    # Step 2: Check if the mod folder already exists; create it if not
    mod_folder_path = os.path.join('Mods', mod_name)
    os.makedirs(mod_folder_path, exist_ok=True)

    # Step 3: Create or load the main.py file in the mod folder
    create_or_load_main_py(bot_profile, mod_folder_path)

    # Step 4: Initiate a loop for helping the user create the mod
    user_input = ""
    while user_input.lower() not in ["exit mod builder", "close mod builder", "we are done", "we are finished"]:
        # Listen for the wake word inside the loop
        if listen_for_wake_word(wake_word):
            # Step 3: Continue with the mod builder logic
            user_input = get_voice_input(bot_profile, "How can I help?")
            update_script(bot_profile, user_input, mod_folder_path)
        else:
            print("Waiting for the wake word...")

    # Step 5: Create the IMPORTS.py file in the mod folder
    create_imports_py(bot_profile, mod_folder_path)

    # Step 7: Create the descriptions.txt for the new mod
    create_descriptions_txt(bot_profile, mod_folder_path)

    prompt = (f"Mod '{mod_name}' has been created successfully.")
    bot_speak(bot_profile, prompt)
def create_or_load_main_py(bot_profile, mod_folder_path):
    main_py_path = os.path.join(mod_folder_path, 'main.py')

    if not os.path.exists(main_py_path):
        # If main.py doesn't exist, create an empty file
        open(main_py_path, 'w').close()
    else:
        # If main.py already exists, open it with the default script editor
        try:
            # Use the system's default program for opening Python scripts
            subprocess.run(['open', main_py_path], check=True)
        except subprocess.CalledProcessError:
            prompt = ("Failed to open main.py. Please open it manually.")
            bot_speak(bot_profile, prompt)
def get_functions_and_arguments(main_py_path):
    """
    Extract functions and their arguments from main.py.
    """
    functions_and_arguments = {}

    with open(main_py_path, 'r') as main_py:
        lines = main_py.readlines()

        # Regular expression pattern to find function definitions
        pattern = re.compile(r"def (\w+)\(([^)]*)\):")

        for line in lines:
            match = pattern.match(line)
            if match:
                function_name = match.group(1)
                arguments = [arg.strip() for arg in match.group(2).split(',')] if match.group(2) else []
                functions_and_arguments[function_name] = arguments

    return functions_and_arguments

def create_imports_py(bot_profile, mod_folder_path):
    imports_py_path = os.path.join(mod_folder_path, 'IMPORTS.py')

    # If IMPORTS.py already exists, ask the user if they want to overwrite it
    if os.path.exists(imports_py_path):
        overwrite = get_voice_input(bot_profile, f"'IMPORTS.py' already exists. Do you want to overwrite it? ")
        if overwrite.lower() != 'yes':
            prompt = ("Import creation canceled.")
            bot_speak(bot_profile, prompt)
            return

    # Get function names and arguments from main.py in the mod folder
    main_py_path = os.path.join(mod_folder_path, 'main.py')
    if os.path.exists(main_py_path):
        functions_and_arguments = get_functions_and_arguments(main_py_path)
    else:
        prompt = ("Error: 'main.py' not found. Make sure it exists in the mod folder.")
        bot_speak(bot_profile, prompt)
        return

    # Create or overwrite IMPORTS.py with default content
    with open(imports_py_path, 'w') as imports_py:
        imports_py.write("# Your mod-specific imports go here\n\n")

        # Iterate over functions and write the import statements
        for function_name, _ in functions_and_arguments.items():
            imports_py.write(f"from ADAM.Mods.{os.path.basename(mod_folder_path)}.main import {function_name}\n")

        imports_py.write("COMMANDS = [\n")

        # Iterate over functions and ask for command phrases
        for function_name, arguments in functions_and_arguments.items():
            commands = get_voice_input(bot_profile, f"What command phrase(s) should trigger {function_name} function?").split(',')
            add_more_commands = get_voice_input(bot_profile, f"Do you want to add more command phrases for {function_name}?").lower() == 'yes'

            while add_more_commands:
                additional_command = get_voice_input(bot_profile, "Enter additional command phrase:")
                commands.append(additional_command)
                add_more_commands = get_voice_input(bot_profile, f"Do you want to add more command phrases for {function_name}?").lower() == 'yes'

            # Write the command entry to IMPORTS.py
            imports_py.write(f'    {{"commands": {commands}, "action": {function_name}, "args": {arguments}}},\n')

        imports_py.write("]\n\n")
        imports_py.write("MAP = {\n")

        # Write the mapping entries to IMPORTS.py
        for function_name, arguments in functions_and_arguments.items():
            imports_py.write(f'    {function_name}: {arguments},\n')

        imports_py.write("}")

def create_descriptions_txt(bot_profile, mod_folder_path):
    descriptions_txt_path = os.path.join(mod_folder_path, 'descriptions.txt')

    # If descriptions.txt already exists, ask the user if they want to overwrite it
    if os.path.exists(descriptions_txt_path):
        overwrite = get_voice_input(bot_profile, f"'descriptions.txt' already exists. Do you want to overwrite it? ")
        if overwrite.lower() != 'yes':
            prompt = ("Description creation canceled.")
            bot_speak(bot_profile, prompt)
            return

    # Create or overwrite descriptions.txt with user-specified content
    prompt = "Let's add descriptions for each command in your mod. You can say 'done' when you're finished."
    bot_speak(bot_profile, prompt)

    # Keep asking the user for command phrases and descriptions until they say 'done'
    while True:
        command_phrase = get_voice_input(bot_profile, "Say a command phrase or say done if you are finished")
        if command_phrase.lower() == 'done':
            break

        description = get_voice_input(bot_profile, f"What is the description for the command '{command_phrase}':")

        # Format and append the user's input to descriptions.txt
        with open(descriptions_txt_path, 'a') as descriptions_txt:
            descriptions_txt.write(f"{command_phrase}: {description}\n")

    prompt = "Descriptions have been added to 'descriptions.txt'."
    bot_speak(bot_profile, prompt)