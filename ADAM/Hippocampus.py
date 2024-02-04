# Author: Chance Brownfield
# Email: ChanceBrownfield@protonmail.com
from ADAM.clean_audio import clean_audio
from ADAM.Temporal import remember_user, identification_test, get_user, get_voice_input, get_voice
from ADAM.respond import select_voice, bot_speak, create_avatar, display_image_box
from ADAM.is_personal import is_personal, save_personal_info
from ADAM.AMSSR import Audio_Recognition as ar
import os
import sys
import json
import speech_recognition as sr
import pyttsx3
import pyaudio
import pygame
import wave
import time
import random
SAMPLE_RATE = 16000
BASE_DIR = "user_data"
ar = ar()
segmented_audio_dir = "segmented_audio"
def find_user_texts(user_profile):
    user_name = user_profile['name']
    concatenated_text = ""

    for file_name in os.listdir(segmented_audio_dir):
        if file_name.startswith(user_name):
            base_name, extension = os.path.splitext(file_name)
            separator_index = base_name.rfind('_')

            if separator_index != -1:
                user_name_from_file = base_name[:separator_index]
                text_path = os.path.join(segmented_audio_dir, f"{user_name_from_file}.txt")

                if os.path.exists(text_path):
                    with open(text_path, 'r') as text_file:
                        concatenated_text += text_file.read()

    if concatenated_text:
        return concatenated_text

    return None  # Return None if no matching text files are found

def bio_check(user_profile, bot_profile):
    user_id = user_profile['id']
    # Define the path to the autobiography file for the user in the bot's profile directory
    autobiography_path = os.path.join('bot_profiles', bot_profile['name'], f"{user_id}_autobiography.txt")

    # Check if the autobiography file already exists
    if not os.path.exists(autobiography_path):
        # If not, get the user's digital handshake
        digital_handshake = user_profile.get('bio', '')

        # Create the bot's profile directory for the user if it doesn't exist
        os.makedirs(os.path.dirname(autobiography_path), exist_ok=True)

        # Save the digital handshake as the user's bio for the bot
        with open(autobiography_path, 'w') as file:
            file.write(digital_handshake)
        print(f"User bio for {user_profile['name']} created in bot profile {bot_profile['name']}.")

    else:
        print(f"User bio for {user_profile['name']} already exists in bot profile {bot_profile['name']}.")


def update_bio(user_profile, bot_profile):
    bio_check(user_profile, bot_profile)
    text = find_user_texts(user_profile)
    # Check if the text is classified as personal
    if is_personal(text):
        # If it's personal, save the personal info
        save_personal_info(text, user_profile, bot_profile)
        print("Personal info saved successfully.")
    else:
        print("Text is not classified as personal.")

# Function to create a new user profile
def create_new_user_profile():
    engine = pyttsx3.init()

    # Name confirmation loop
    user_name = ""
    while not user_name:
        engine.say("Please state your name.")
        engine.runAndWait()

        # Use ASR method for name recognition
        user_name = ar.ASR(None, 5)

        engine.say(f"You said {user_name}, is that correct?")
        engine.runAndWait()

        confirmation_text = ar.ASR()

        if 'no' in confirmation_text.lower():
            user_name = ""
        elif 'yes' in confirmation_text.lower():
            break
        else:
            engine.say("I didn't catch that. Let's try again.")
            engine.runAndWait()
            user_name = ""

    timestamp = int(time.time())
    user_id = f"{user_name}_{timestamp}"

    user_dir = os.path.join(BASE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    user_profile = {
        "id": user_id,
        "name": user_name,
        "commands": {},
        "history": []
    }

    # User bio confirmation loop
    bio = ""
    while not bio:
        engine.say(
            "Please describe yourself. This is how every assistant will be introduced to you. Everything else they will learn along the way as they get to know you.")
        engine.runAndWait()

        bio = ar.ASR()

        engine.say(f"You described yourself as: {bio}, is that correct?")
        engine.runAndWait()

        # Use ASR method for confirmation
        confirmation_audio = ar.ASR()

        if 'no' in confirmation_audio.lower():
            bio = ""
        elif 'yes' in confirmation_audio.lower():
            break  # Exit the loop if the description is confirmed
        else:
            engine.say("I didn't catch that. Let's try again.")
            engine.runAndWait()
            bio = ""

    profile_file = os.path.join(user_dir, "profile.json")
    with open(profile_file, "w") as profile_json:
        json.dump(user_profile, profile_json)


    # Record the ABC song for voice training
    engine.say("Please sing the ABC song for voice training.")
    engine.runAndWait()

    print("Recording voice sample for training...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(0, int(SAMPLE_RATE / 1024 * 60)):
        data = stream.read(1024)
        frames.append(data)

    # Ensure the training_data directory exists
    os.makedirs('training_data', exist_ok=True)

    # Specify the path to the training_data directory in the sample_filename
    raw_sample_filename = os.path.join('training_data', f"user_{user_id}_ABC_raw.wav")

    wf = wave.open(raw_sample_filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    # Process the audio file before training
    processed_sample_filename = clean_audio(raw_sample_filename)
    if processed_sample_filename is None:
        print("There was an error processing the audio. Aborting model training.")
        sys.exit(1)  # Exit the program with a non-zero status to indicate an error

    # Train the model on the processed audio file using SpeechBrain
    remember_user(user_id, processed_sample_filename)

    while True:
        engine.say("Please say 'hello Adam how are you?' to test the voice recognition.")
        engine.runAndWait()
        with sr.Microphone() as source:
            print("Listening for test phrase...")
            test_audio = ar.listen_and_record(source)
        test_audio_filename = os.path.join(user_dir, "test_audio.wav")
        with open(test_audio_filename, "wb") as f:
            f.write(test_audio.get_wav_data())

        processed_test_audio_filename = clean_audio(test_audio_filename)  # Processing audio
        if processed_test_audio_filename is None:
            engine.say("There was an error processing the audio. Let's try again.")
            engine.runAndWait()
            continue  # Skip to the next iteration of the loop if no processed audio was obtained

        predicted_user_id = identification_test(processed_test_audio_filename)
        if predicted_user_id == user_id:
            engine.say("Voice recognition model trained and saved successfully.")
            engine.runAndWait()
            break
        else:
            engine.say("There was an error. Let's try recording the ABC song again.")
            engine.runAndWait()
            # ... (Record the ABC song and train the model again)

    return user_id, user_profile

def get_bot(wake_word):
    bot_name = wake_word  # Since the wake word is the bot's name
    bot_profile_path = os.path.join('./bot_profiles', bot_name, 'profile.json')
    if os.path.exists(bot_profile_path):
        with open(bot_profile_path, 'r') as file:
            bot_profile = json.load(file)
        return bot_profile
    else:
        print(f"No profile found for bot: {bot_name}")
        return None
def create_new_bot(user_profile):
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()
    user_id = user_profile['id']

    # Create a directory for bot profiles if it doesn't exist
    if not os.path.exists('bot_profiles'):
        os.makedirs('bot_profiles')

    confirmed = False
    while not confirmed:
        bot_name, bot_bio = "", ""
        while not bot_name or os.path.exists(f"bot_profiles/{bot_name}_profile.json"):
            with sr.Microphone() as source:
                engine.say("What would you like to name your bot?")
                engine.runAndWait()
                recognizer.adjust_for_ambient_noise(source)
                while True:  # Loop until a valid name is obtained
                    try:
                        audio = recognizer.listen(source, timeout=10)
                        bot_name = recognizer.recognize_google(audio).strip()
                        if os.path.exists(f"bot_profiles/{bot_name}_profile.json"):
                            engine.say(f"A bot named {bot_name} already exists. Please provide a different name.")
                            engine.runAndWait()
                        else:
                            break
                    except sr.WaitTimeoutError:
                        engine.say("Listening timed out. Please try speaking again.")
                        engine.runAndWait()
                    except (sr.RequestError, sr.UnknownValueError) as e:
                        engine.say(f"Error: {e}. Please try again.")
                        engine.runAndWait()

        with sr.Microphone() as source:
            engine.say(f"Describe {bot_name}'s personality:")
            engine.runAndWait()
            recognizer.adjust_for_ambient_noise(source)
            while True:  # Loop until a valid description is obtained
                try:
                    audio = recognizer.listen(source, timeout=10)
                    bot_bio = recognizer.recognize_google(audio).strip()
                    break
                except sr.WaitTimeoutError:
                    engine.say("Listening timed out. Please try speaking again.")
                    engine.runAndWait()
                except (sr.RequestError, sr.UnknownValueError) as e:
                    engine.say(f"Error: {e}. Please try again.")
                    engine.runAndWait()

        # Confirm the bot name and bio with the user
        engine.say(f"You've named your bot {bot_name} and described its personality as {bot_bio}. Is that correct?")
        engine.runAndWait()

        with sr.Microphone() as source:
            engine.say("Is this information correct? (yes/no)")
            engine.runAndWait()
            recognizer.adjust_for_ambient_noise(source)
            while True:  # Loop until a valid confirmation is obtained
                try:
                    audio = recognizer.listen(source, timeout=10)
                    confirmation = recognizer.recognize_google(audio).strip().lower()
                    if 'yes' in confirmation:
                        confirmed = True
                        break
                    elif 'no' in confirmation:
                        engine.say("Let's try again.")
                        engine.runAndWait()
                        break
                    else:
                        engine.say("Please respond with 'yes' or 'no'.")
                        engine.runAndWait()
                except sr.WaitTimeoutError:
                    engine.say("Listening timed out. Please try speaking again.")
                    engine.runAndWait()
                except (sr.RequestError, sr.UnknownValueError) as e:
                    engine.say(f"Error: {e}. Please try again.")
                    engine.runAndWait()
        # After confirming the bot's name and bio, now we select the voice
    voicelib, voiceset = select_voice()
    emotion_history_dir = os.path.join('bot_profiles', bot_name, 'emotion_histories')
    os.makedirs(emotion_history_dir, exist_ok=True)

    bot_profile = {
        "id": bot_name,
        "name": bot_name,
        "bio": bot_bio,
        "emotion_histories": emotion_history_dir,
        "history": [],
        "creator": user_id,
        "voicelib": voicelib,  # Add the selected voice library key
        "voiceset": voiceset  # Add the selected voice set key
    }

    # Create the bot's directory if it doesn't exist
    bot_dir = os.path.join('bot_profiles', bot_name)
    os.makedirs(bot_dir, exist_ok=True)

    # Save the bot profile to a file within the bot's directory
    bot_profile_file_path = os.path.join(bot_dir, "profile.json")
    with open(bot_profile_file_path, 'w') as file:
        json.dump(bot_profile, file, indent=4)

    # Load existing wake words from file, or start with an empty list if the file doesn't exist
    wake_words_file = os.path.join('bot_profiles', 'wake_words.json')
    if os.path.exists(wake_words_file):
        with open(wake_words_file, 'r') as file:
            wake_words = json.load(file)
    else:
        wake_words = []

    # Add the new bot name to the wake words list
    wake_words.append(bot_name.lower())

    # Save the updated wake words list back to the file
    with open(wake_words_file, 'w') as file:
        json.dump(wake_words, file, indent=4)

        # loop to create an avatar for new bot
        while True:
            engine.say("Please describe what the new bot should look like.")
            engine.runAndWait()

            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = recognizer.listen(source, timeout=10)
                    bot_description = recognizer.recognize_google(audio).strip()
                    engine.say("Let me create an avatar based on your description.")
                    engine.runAndWait()

                    # Create avatar based on user's description
                    avatar_image = create_avatar(bot_description)

                    # Display the avatar image
                    display_image_box(avatar_image, pygame.display.get_surface())

                    # Ask the user if the avatar looks right
                    engine.say("Does this avatar look right? (yes/no)")
                    engine.runAndWait()

                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source)
                        try:
                            audio = recognizer.listen(source, timeout=10)
                            confirmation = recognizer.recognize_google(audio).strip().lower()
                            if 'yes' in confirmation:
                                # Save the image in the "Avatars" folder using the bot name as the file name
                                avatar_folder = os.path.join('Avatars', bot_name)
                                os.makedirs(avatar_folder, exist_ok=True)
                                avatar_file_path = os.path.join(avatar_folder, f"{bot_name}.png")
                                avatar_image.save(avatar_file_path)

                                engine.say("Avatar saved successfully!")
                                engine.runAndWait()
                                break  # Exit the loop as the user confirmed the avatar
                            elif 'no' in confirmation:
                                engine.say("Let's try again.")
                                engine.runAndWait()
                            else:
                                engine.say("Please respond with 'yes' or 'no'.")
                                engine.runAndWait()
                        except sr.WaitTimeoutError:
                            engine.say("Listening timed out. Please try speaking again.")
                            engine.runAndWait()
                        except (sr.RequestError, sr.UnknownValueError) as e:
                            engine.say(f"Error: {e}. Please try again.")
                            engine.runAndWait()
                except sr.WaitTimeoutError:
                    engine.say("Listening timed out. Please try speaking again.")
                    engine.runAndWait()
                except (sr.RequestError, sr.UnknownValueError) as e:
                    engine.say(f"Error: {e}. Please try again.")
                    engine.runAndWait()

        print(f"{bot_name} has been created successfully!")
        engine.say(f"{bot_name} has been created successfully!")
        engine.runAndWait()  # Ensure the text is spoken before the function returns
        return bot_profile

def update_user_profile(user_profile):
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

    # Ask user if they want to update their name
    while True:
        engine.say("Would you like to update your name? Say yes or no.")
        engine.runAndWait()
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
        response = recognizer.recognize_google(audio).lower()
        if response in ["yes", "no"]:
            break
        else:
            engine.say("I didn't catch that. Please say yes or no.")
            engine.runAndWait()

    # If yes, get the new name and confirm
    if response == "yes":
        while True:
            engine.say("Please state your new name.")
            engine.runAndWait()
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
            new_name = recognizer.recognize_google(audio)
            engine.say(f"Your new name is {new_name}, is that correct? Say yes or no.")
            engine.runAndWait()
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
            response = recognizer.recognize_google(audio).lower()
            if response == "yes":
                user_profile['name'] = new_name  # update the name in user_profile
                break

    # Ask user if they want to update their digital handshake
    while True:
        engine.say("Would you like to update your digital handshake? Say yes or no.")
        engine.runAndWait()
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
        response = recognizer.recognize_google(audio).lower()
        if response in ["yes", "no"]:
            break
        else:
            engine.say("I didn't catch that. Please say yes or no.")
            engine.runAndWait()

    # If yes, get the new digital handshake and confirm
    if response == "yes":
        while True:
            engine.say("Please describe your physical appearance including gender, race, and any defining features.")
            engine.runAndWait()
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
            new_handshake = recognizer.recognize_google(audio)
            engine.say(f"Your new digital handshake is: {new_handshake}, is that correct? Say yes or no.")
            engine.runAndWait()
            with sr.Microphone() as source:
                audio = recognizer.listen(source)
            response = recognizer.recognize_google(audio).lower()
            if response == "yes":
                user_profile['digital_handshake'] = new_handshake  # update the digital handshake in user_profile
                break

    # Save the updated user profile
    user_dir = os.path.join(BASE_DIR, user_profile['id'])
    profile_file = os.path.join(user_dir, "profile.json")
    with open(profile_file, "w") as profile_json:
        json.dump(user_profile, profile_json)

    engine.say("User profile updated.")
    engine.runAndWait()

def is_locked_down():
    lockdown_file_path = './bot_profiles/lockdown'
    return os.path.exists(lockdown_file_path)

# Function to verify the user
def verify_user():
    wake_words_file = os.path.join('bot_profiles', 'wake_words.json')
    with open(wake_words_file, 'r') as file:
        wake_words = json.load(file)
        print(f"wake_words: {wake_words}")
        # Randomly pick a wake word from the list
        random_wake_word = random.choice(wake_words)
        print(f"Matching bot profile for wake word: {random_wake_word}")
        bot_profile = get_bot(random_wake_word)
    user_id = "unknown"
    user_profile = {}

    while user_id == "unknown":
        a_prompt = "I'm sorry, do I know you?"
        user_input, processed_audio_filename = get_voice(bot_profile, a_prompt)

        if 'yes' in user_input:
            # Randomly pick a wake word from the list
            random_wake_word = random.choice(wake_words)

            print(f"Matching bot profile for wake word: {random_wake_word}")
            bot_profile = get_bot(random_wake_word)

            print("Identifying user by voice...")
            user_id, user_profile = get_user(processed_audio_filename)

            if user_id == "unknown":
                c_prompt = "Alright, let's start over. Please say your name."
                bot_speak(bot_profile, c_prompt)
                lockdown_file_path = os.path.join(f'./bot_profiles', 'lockdown')
                if not os.path.exists(lockdown_file_path):
                    user_id, user_profile = create_new_user_profile()
                    d_prompt = f"Nice to meet you, {user_profile['name']}!"
                    bot_speak(bot_profile, d_prompt)
            else:
                d_prompt = f"Welcome back, {user_profile['name']}!"
                bot_speak(bot_profile, d_prompt)
        else:
            e_prompt = "Alright, let's start over. Please say your name."
            bot_speak(bot_profile, e_prompt)
            if not is_locked_down():
                user_id, user_profile = create_new_user_profile()
                f_prompt = f"Nice to meet you, {user_profile['name']}!"
                bot_speak(bot_profile, f_prompt)

    return user_id, user_profile