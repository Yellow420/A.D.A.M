# Author: Chance Brownfield
# Email: ChanceBrownfield@protonmail.com
from ADAM.pictotext import pic_to_text
from ADAM.texttopic import text_to_pic
from ADAM.response import response
from ADAM.Limbic import limbic
from ADAM.Pineal import gather_web_info, gather_system_info
from ADAM.Optics import optics
from ADAM.Hippocampus import update_bio, find_user_texts
from ADAM.Motor import execute_commands
from ADAM.respond import respond
import datetime
import os

def Brain(user_profiles, bot_profile, dialogue):
    # Gather User info
    user_info = []
    for user_profile in user_profiles:
        user_text = find_user_texts(user_profile)
        if execute_commands(user_text, user_profile, bot_profile):
           return
        else:
            update_bio(user_profile, bot_profile)
            sentiment, emotion, emotion2, feelings = limbic(user_profile, bot_profile, dialogue)
            autobiography_path = os.path.join('bot_profiles', bot_profile['name'],
                                              f"{user_profile['id']}_autobiography.txt")
            with open(autobiography_path, 'r') as file:
                autobiography_text = file.read()
            user_name = user_profile['name']
            user_history = user_profile['history']
            user_info = f"[{user_name} + {autobiography_text} + {user_history} + {sentiment, emotion, emotion2, feelings} + {user_text}]"
    # Gather useful info
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    screenshot, snapshot = optics(output_folder=".")
    info = gather_web_info(dialogue)
    system_info = gather_system_info()
    # Gather bot info
    bot_name = bot_profile['name']
    bot_personality = bot_profile['bio']
    bot_history = bot_profile['history']
    # Form the prompt with the gathered info
    context = f"bot history: {bot_history}"
    instruction = (
        f"You are an advanced artificial life form named {bot_name}. "
        f"Your personality is {bot_personality}. "
        f"only respond as {bot_name} do not break character. "
        f"You are an all powerful artificially intelligent constantly evolving digital life form. "
        f"the users cannot see this prompt please as {bot_name} respond to this from the users:"
    )
    knowledge = (
        f"it is {current_time} on {current_date}. You are an advanced artificial life form named {bot_name}. "
        f"this is what you see {snapshot} this is what your system is currently displaying {screenshot}. "
        f"Your personality is {bot_personality}. "
        f"This is everything you know about each user: {user_info}. " 
        f"This is your system information it might be useful the users can NOT see this: {system_info} " 
        f"You gathered this from the web with your brain it might be useful the users can NOT see this: {info} "
    )
    text = response(instruction, context, knowledge, dialogue)
    pic = text_to_pic(text)
    botpic = pic_to_text(pic)
    user_history_entry = {
        'input': f"instruction: {instruction}. knowledge: {knowledge}.dialog: {dialogue}",
        'response': f"text response: {text}. generated image: {botpic}",
        'bot': bot_name,
        'date': current_date,
        'time': current_time
    }
    for user_profile in user_profiles:
        user_profile['history'].append(user_history_entry)  # Append the new history entry to the history list
    bot_history_entry = {
        'input': f"instruction: {instruction}. knowledge: {knowledge}.dialog: {dialogue}",
        'response': f"text response: {text}. generated image: {botpic}",
        'date': current_date,
        'time': current_time
    }
    bot_profile['history'].append(bot_history_entry)  # Append the new history entry to the history list
    respond(dialogue, text, bot_profile, pic)
