# Author: Chance Brownfield
# Email: ChanceBrownfield@protonmail.com
import os
import speech_recognition as sr
import json
import time
from ADAM.Brain import Brain
from ADAM.Hippocampus import create_new_user_profile, create_new_bot, get_bot
from ADAM.idle import idle
from ADAM.AudioRecognition import Audio_Recognition as ar
from ADAM.Locker import is_locked
asr = ar()

# Main loop
while True:
    print("Entering main loop")
    wake_words_file = os.path.join('bot_profiles', 'wake_words.json')

    # Timer for idle chat
    idle_timer = None

    if os.path.exists(wake_words_file):
        with open(wake_words_file, 'r') as file:
            wake_words = json.load(file)
            print(f"wake_words: {wake_words}")

            # Start the timer for idle chat
            idle_timer = time.time()
    else:
        wake_words = []
        print("No wake words found.")

    if not wake_words:
        print("About to create a new user profile")
        user_id, user_profile = create_new_user_profile()
        print("Created new user profile, about to create a new bot")
        create_new_bot(user_profile)
    else:
        print("Wake words found, proceeding with existing user profile and bot.")

    if wake_words:
        print("wake_words is not empty, skipping user and bot creation")
        # Check for idle.txt and the timer
        if os.path.exists('idle.txt') and idle_timer is not None:
            with open('idle.txt', 'r') as idle_file:
                idle_time = int(idle_file.read().strip())
                if idle_time < time.time() - idle_timer:
                    # Call the idle function
                    print("Calling idle function...")
                    idle()
                    idle_timer = time.time()  # Reset the timer
                    continue  # Go back to listening for a wake word
        print("wake_words is not empty, skipping user and bot creation")
        try:
            print(f"Listening for {wake_words}...")
            # Record audio using AMSSR
            dialogue, user_profiles = asr.AMSSR()
            print(f"AMSSR Result: {dialogue}")

            if not dialogue:
                print("No speech segments were found.")
                continue  # Skip to the next iteration of the main loop if no speech segments were found

            # Identify the wake word
            detected_wake_word = None
            for wake_word in wake_words:
                if wake_word in dialogue.lower():
                    detected_wake_word = wake_word
                    print(f"Wake word {wake_word} detected.")
                    break

            if detected_wake_word:
                # Match the bot profile to the wake word
                bot_profile = get_bot(detected_wake_word)

                # Check if the bot is locked
                if is_locked(bot_profile, user_profiles):
                    continue  # Go back to listening for a wake word

                Brain(user_profiles, bot_profile, dialogue)

                # Second loop for additional input
                start_time = time.time()
                while time.time() - start_time < 30:
                    print("Entering second loop for additional input")
                    try:
                        print(f"Listening for additional input...")
                        # Record additional audio using AMSSR
                        additional_dialogue, _ = asr.AMSSR()
                        print(f"AMSSR Additional Input Result: {additional_dialogue}")

                        if not additional_dialogue:
                            print("No speech segments were found.")
                            continue  # Skip to the next iteration of the second loop if no speech segments were found

                        # Identify the wake word for additional input
                        detected_additional_wake_word = None
                        for wake_word in wake_words:
                            if wake_word in additional_dialogue.lower():
                                detected_additional_wake_word = wake_word
                                print(f"Wake word {wake_word} detected in additional input.")
                                break

                        if detected_additional_wake_word:
                            # Match the bot profile to the wake word for additional input
                            bot_profile = get_bot(detected_additional_wake_word)

                            # Check if the bot is locked
                            if is_locked(bot_profile, user_profiles):
                                break  # Exit the second loop and go back to the first loop

                            Brain(user_profiles, bot_profile, additional_dialogue)
                        else:
                            # No wake word detected in additional input, continue the conversation with the last used bot profile
                            Brain(user_profiles, bot_profile, additional_dialogue)
                    except sr.WaitTimeoutError:
                        print("No new additional input received within the timeout.")
                        break  # Exit the second loop and go back to the first loop
            else:
                print("No wake word detected in the audio.")
                continue

        except sr.RequestError:
            print("Sorry, I'm having trouble processing your request.")
        except sr.WaitTimeoutError:
            print("No new input received within the timeout.")
