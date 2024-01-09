#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
from ADAM.pictotext import pic_to_text
from ADAM.texttopic import text_to_pic
from ADAM.response import response
from ADAM.Limbic import limbic
from ADAM.Pineal import gather_web_info, gather_system_info
from ADAM.Optics import optics
from ADAM.Hippocampus import update_bio
from ADAM.Motor import execute_commands
from ADAM.respond import respond,generate_transcription
import datetime
import os

def Brain(user_profile, bot_profile, audio_file_path):
    final_input = generate_transcription(audio_file_path)
    if execute_commands(final_input, user_profile, bot_profile):
       return
    else:
        update_bio(final_input, user_profile, bot_profile)
        # Get the current date and time
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        # Extract bot and user details
        bot_name = bot_profile['name']
        bot_personality = bot_profile['bio']
        user_name = user_profile['name']
        # Get the path to the autobiography file
        autobiography_path = os.path.join('bot_profiles', bot_profile['name'],
                                          f"{user_profile['id']}_autobiography.txt")
        # Load the user's bio from the autobiography file
        with open(autobiography_path, 'r') as file:
            autobiography_text = file.read()
        screenshot, snapshot = optics(output_folder=".")
        sentiment, emotion, emotion2, feelings = limbic(user_profile, bot_profile, final_input, audio_file_path)
        info = gather_web_info(final_input)
        system_info = gather_system_info()
        user_history = user_profile['history']
        bot_history = bot_profile['history']
        context = f"user history:{user_history} .bot history: {bot_history}?"
        instruction = (
            f"it is {current_time} on {current_date}. You are an advanced artificial life form named {bot_name}. "
            f"Your personality is {bot_personality}. "
            f"You are talking to {user_name}. "
            f"only respond as {bot_name} do not break character. "
            f"You are an all powerful artificially intelligent constantly evolving digital life form. "
            f"{user_name} cannot see this prompt please as {bot_name} respond to this from {user_name}:"
        )
        knowledge = (
            f"it is {current_time} on {current_date}. You are an advanced artificial life form named {bot_name}. "
            f"this is what you see {snapshot} this is what your system is currently displaying {screenshot}. "
            f"Your personality is {bot_personality}. "
            f"You are talking to {user_name}. "
            f"{user_name} is {autobiography_text}. "
            f"{user_name} is currently expressing {sentiment}, {emotion}, {emotion2}. "
            f"Your feelings towards {user_name} are {feelings}. "
            f"This is your system information it might be useful {user_name} can NOT see this: {system_info} " 
            f"You gathered this from the web with your brain it might be useful {user_name} can NOT see this: {info} "
        )
        text = response(instruction, context, knowledge, final_input)
        pic = text_to_pic(text)
        botpic = pic_to_text(pic)
        user_history_entry = {
            'input': f"instruction: {instruction}. knowledge: {knowledge}.dialog: {final_input}",
            'response': f"text response: {text}. generated image: {botpic}",
            'bot': bot_name,
            'date': current_date,
            'time': current_time
        }
        user_profile['history'].append(user_history_entry)  # Append the new history entry to the history list
        bot_history_entry = {
            'input': f"instruction: {instruction}. knowledge: {knowledge}.dialog: {final_input}",
            'response': f"text response: {text}. generated image: {botpic}",
            'user': user_name,
            'date': current_date,
            'time': current_time
        }
        bot_profile['history'].append(bot_history_entry)  # Append the new history entry to the history list
        respond(final_input, text, bot_profile, user_profile, pic)
