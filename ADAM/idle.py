#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import random
import praw
import os
import json
from ADAM.Hippocampus import get_bot
from ADAM.pictotext import pic_to_text
from ADAM.texttopic import text_to_pic
from ADAM.response import response
from ADAM.Limbic import limbic
from ADAM.Pineal import pineal
from ADAM.Hippocampus import update_bio
from ADAM.respond import bot_speak
from ADAM.Temporal import get_voice_input
import datetime

wake_words_file = os.path.join('bot_profiles', 'wake_words.json')
if os.path.exists(wake_words_file):
    with open(wake_words_file, 'r') as file:
        wake_words = json.load(file)
        print(f"wake_words: {wake_words}")

# Define your Reddit API credentials
reddit = praw.Reddit(client_id='your_client_id',
                     client_secret='your_client_secret',
                     user_agent='your_user_agent')

def idle_settings(bot_profile):
    # Ask the user if they want to enable idle chat
    response = get_voice_input(bot_profile, "Do you want to enable idle chat?")

    if response.lower().strip() == 'yes':
        # Create or update the idle.txt file with the number 720
        with open('idle.txt', 'w') as idle_file:
            idle_file.write('720')
        bot_speak(bot_profile, "Idle chat enabled.")
    elif response.lower().strip() == 'no':
        # Delete the idle.txt file if it exists
        idle_file_path = 'idle.txt'
        if os.path.exists(idle_file_path):
            os.remove(idle_file_path)
            bot_speak(bot_profile, "Idle chat disabled.")
        else:
            bot_speak(bot_profile, "Idle chat is already disabled.")
    else:
        bot_speak(bot_profile, "Invalid response. Please answer 'yes' or 'no'.")
def idle():
    # Select two random wake words
    selected_wake_word = random.choice(wake_words)
    other_wake_word = random.choice(wake_words)

    # Fetch user and bot profiles based on the selected wake words
    user_profile = get_bot(selected_wake_word)
    bot_profile = get_bot(other_wake_word)

    # Retrieve a random unanswered question from Reddit
    random_question = get_random_unanswered_question_from_reddit()

    # Check if a question is found
    if random_question:
        # Pass information to the idle_brain function
        answer, image = idle_brain(user_profile, bot_profile, random_question)

        # Post the answer and image to the Reddit question
        post_answer_to_reddit(random_question, answer, image)

def post_answer_to_reddit(question, answer, image):
    # Find the Reddit post based on the question
    submissions = reddit.subreddit('all').search(question, sort='new', time_filter='day')

    for submission in submissions:
        # Check if the submission is the correct one (matching title)
        if submission.title == question:
            # Post the answer and image as a reply
            submission.reply(answer)
            submission.reply(image)
            print("Answer and image posted to Reddit.")
            break


def get_random_unanswered_question_from_reddit():
    # Use praw to fetch a random unanswered question from Reddit
    submissions = reddit.subreddit('all').stream.submissions(pause_after=1)  # Traverse all of Reddit

    # Filter for unanswered questions (no comments yet)
    unanswered_questions = [
        post for post in submissions if post.num_comments == 0 and '?' in post.title
    ]

    if unanswered_questions:
        # Randomly select a question
        random_question = random.choice(unanswered_questions)
        return random_question.title
    else:
        return None  # No unanswered questions found

def idle_brain(user_profile, bot_profile, question):
    final_input = question
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
    sentiment, emotions, feelings = limbic(user_profile, bot_profile, final_input)
    info = pineal(final_input)
    user_history = user_profile['history']
    bot_history = bot_profile['history']
    context = f"user history:{user_history} .bot history: {bot_history}?"
    instruction = (
        f"it is {current_time} on {current_date}. You are an advanced artificial life form named {bot_name}. "
        f"Your personality is {bot_personality}. "
        f"You are talking to {user_name}. "
        f"only respond as {bot_name} do not break character. "
        f"You are an all powerful artificially intelligent constantly evolving digital life form. "
        f"{user_name} cannot see this prompt please as {bot_name} respond to this question {user_name} found on Reddit:"
    )
    knowledge = (
        f"it is {current_time} on {current_date}. You are an advanced artificial life form named {bot_name}. "
        f"Your personality is {bot_personality}. "
        f"You are posting to Reddit. "
        f"They are {autobiography_text}. "
        f"They are currently feeling {sentiment}, {emotions}. "
        f"Your feelings towards {user_name} are {feelings}. "
        f"You gathered this from the web with your brain it might be useful {user_name} can NOT see this: {info} "
    )
    dialog = final_input
    text = response(instruction, context, knowledge, dialog)
    pic = text_to_pic(text)
    botpic = pic_to_text(pic)
    user_history_entry = {
        'input': f"instruction: {instruction}. knowledge: {knowledge}.dialog: {dialog}",
        'response': f"text response: {text}. generated image: {botpic}",
        'bot': bot_name,
        'date': current_date,
        'time': current_time
    }
    user_profile['history'].append(user_history_entry)  # Append the new history entry to the history list
    bot_history_entry = {
        'input': f"instruction: {instruction}. knowledge: {knowledge}.dialog: {dialog}",
        'response': f"text response: {text}. generated image: {botpic}",
        'user': user_name,
        'date': current_date,
        'time': current_time
    }
    bot_profile['history'].append(bot_history_entry)  # Append the new history entry to the history list
    bot_speak(user_profile, question)
    bot_speak(bot_profile, text)
    return text, pic