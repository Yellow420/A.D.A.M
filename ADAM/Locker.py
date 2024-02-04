# Author: Chance Brownfield
# Email: ChanceBrownfield@protonmail.com
import os
from ADAM.respond import bot_speak
def is_locked(bot_profile, user_profiles):
    # Path to the segmented_audio folder
    segmented_audio_dir = "segmented_audio"

    # Iterate through all .txt files in the segmented_audio folder
    for file_name in os.listdir(segmented_audio_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(segmented_audio_dir, file_name)

            # Read the content of the .txt file
            with open(file_path, "r") as txt_file:
                content = txt_file.read()

            # Check if the bot's name is mentioned in the content
            if bot_profile['name'] in content:
                # Extract speaker and timestamp from the file name
                speaker, timestamp = os.path.splitext(file_name)[0].split('_')

                # Find the matching user_profile by speaker name
                matching_user = next((user for user in user_profiles if user['name'] == speaker), None)

                if matching_user:
                    # Check if the user's ID matches the bot's creator ID
                    if matching_user['id'] == bot_profile['creator']:
                        print("I only answer to my creator.")
                        f_prompt = "I only answer to my creator."
                        bot_speak(bot_profile, f_prompt)
                        return True  # Bot is locked

    # Check for lock or lockdown files
    lock_file_path = os.path.join(f'./bot_profiles/{bot_profile["name"]}', 'lock')
    lockdown_file_path = os.path.join(f'./bot_profiles', 'lockdown')

    if os.path.exists(lock_file_path) or os.path.exists(lockdown_file_path):
        print("I only answer to my creator.")
        f_prompt = "I only answer to my creator."
        bot_speak(bot_profile, f_prompt)
        return True  # Bot is locked

    # If no lock conditions are met, return False (not locked)
    return False