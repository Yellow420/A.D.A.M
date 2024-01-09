#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import os
import speech_recognition as sr
import json
import time
from ADAM.Brain import Brain
from ADAM.Hippocampus import create_new_user_profile, create_new_bot, get_bot, verify_user
from ADAM.Temporal import get_user
from ADAM.clean_audio import clean_audio
from ADAM.respond import bot_speak
from ADAM.idle import idle
recognizer = sr.Recognizer()

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
        print("Created new user profile, about to create new bot")
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
        with sr.Microphone() as source:
            print(f"Listening for {wake_words}...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                user_audio = recognizer.listen(source, timeout=5)
                # Save the audio file where the wake word was detected
                audio_filename = "wake_word_audio.wav"
                with open(audio_filename, "wb") as f:
                    f.write(user_audio.get_wav_data())

                # Call process_audio to handle audio processing
                processed_audio_file = clean_audio(audio_filename)

                if processed_audio_file is None:
                    print("No segments containing speech were found.")
                    continue  # Skip to the next iteration of the main loop if no speech segments were found

                try:
                    print("google")
                    # Use 'after_vad.wav' directly with the Google Web Speech API
                    with sr.AudioFile('post_proc.wav') as source:
                        audio_data = recognizer.record(source)
                    audio_text = recognizer.recognize_google(audio_data)
                    print(f"You said: {audio_text}")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except Exception as e:
                    print(f"Error: {e}")
                    continue  # Skip to the next iteration of the main loop

                # Identify the user by voice
                print("Identifying user by voice...")
                user_id, user_profile = get_user(processed_audio_file)

                # If the user is not recognized, call the verify_user function
                if user_id == "unknown":
                    user_id, user_profile = verify_user()

                # Identify the wake word
                detected_wake_word = None
                for wake_word in wake_words:
                    if wake_word in audio_text.lower():
                        detected_wake_word = wake_word
                        print(f"Wake word {wake_word} detected.")
                        break

                if detected_wake_word:
                    # Match the bot profile to the wake word
                    bot_profile = get_bot(detected_wake_word)

                    lock_file_path = os.path.join(f'./bot_profiles/{bot_profile["name"]}',
                                                  'lock')
                    lockdown_file_path = os.path.join(f'./bot_profiles', 'lockdown')
                    if os.path.exists(lock_file_path) or os.path.exists(lockdown_file_path):
                        print("I only answer to my creator.")
                        f_prompt =("I only answer to my creator.")
                        bot_speak(bot_profile, f_prompt)
                        continue  # Go back to listening for a wake word

                else:
                    print("No wake word detected in the audio.")
                    continue

                Brain(user_profile, bot_profile, processed_audio_file)

                # Second loop for additional input
                start_time = time.time()
                while time.time() - start_time < 30:
                    print("Entering second loop for additional input")
                    with sr.Microphone() as source:
                        print(f"Listening for additional input...")
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        try:
                            user_audio = recognizer.listen(source, timeout=5)
                            # Save the audio file for additional input
                            audio_filename = "additional_input_audio.wav"
                            with open(audio_filename, "wb") as f:
                                f.write(user_audio.get_wav_data())

                            # Call process_audio for additional audio processing
                            processed_audio_file = clean_audio(audio_filename)

                            if processed_audio_file is None:
                                print("No segments containing speech were found.")
                                continue  # Skip to the next iteration of the second loop if no speech segments were found

                            try:
                                with sr.AudioFile(processed_audio_file) as source:
                                    audio_data = recognizer.record(source)
                                additional_audio_text = recognizer.recognize_google(audio_data)
                                print(f"You said: {additional_audio_text}")
                            except sr.RequestError as e:
                                print(f"Could not request results from Google Speech Recognition service; {e}")
                            except sr.UnknownValueError:
                                print("Google Speech Recognition could not understand audio")
                            except Exception as e:
                                print(f"Error: {e}")
                                continue  # Skip to the next iteration of the second loop

                            # Identify the user by voice for additional input
                            print("Identifying user by voice in the second loop...")
                            user_id, user_profile = get_user(processed_audio_file)

                            # If the user is not recognized, call the verify_user function
                            if user_id == "unknown":
                                user_id, user_profile = verify_user()

                            # Identify the wake word for additional input
                            detected_additional_wake_word = None
                            for wake_word in wake_words:
                                if wake_word in additional_audio_text.lower():
                                    detected_additional_wake_word = wake_word
                                    print(f"Wake word {wake_word} detected in additional input.")
                                    break

                            if detected_additional_wake_word:
                                # Match the bot profile to the wake word for additional input
                                bot_profile = get_bot(detected_additional_wake_word)
                                lock_file_path = os.path.join(f'./bot_profiles/{bot_profile["name"]}',
                                                              'lock')
                                lockdown_file_path = os.path.join(f'./bot_profiles', 'lockdown')
                                if os.path.exists(lock_file_path) or os.path.exists(lockdown_file_path):
                                    # Additional input is for a different bot profile
                                    print("I only answer to my creator.")
                                    break  # Exit the second loop and go back to the first loop
                                else:
                                    # Additional input is for the initial bot profile
                                    print("Additional input is for the initial bot profile.")
                                    Brain(user_profile, bot_profile, processed_audio_file)
                            else:
                                # No wake word detected in additional input, continue the conversation with the last used bot profile
                                Brain(user_profile, bot_profile, processed_audio_file)
                        except sr.WaitTimeoutError:
                            print("No new additional input received within the timeout.")
                            break  # Exit the second loop and go back to the first loop
            except sr.RequestError:
                print("Sorry, I'm having trouble processing your request.")
            except sr.WaitTimeoutError:
                print("No new input received within the timeout.")