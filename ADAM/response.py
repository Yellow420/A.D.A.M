#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datetime

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

def response(instruction, context, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} {context} {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=1028, min_length=12, top_p=0.9, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
def command_response(final_input, user_profile, bot_profile):
    # Get the current date and time
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.datetime.now().strftime('%H:%M:%S')
    # Extract bot and user details
    bot_name = bot_profile['name']
    user_name = user_profile['name']
    user_history = user_profile['history']
    bot_history = bot_profile['history']
    context = f"user history:{user_history} .bot history: {bot_history}?"
    instruction = (
        f"it is {current_time} on {current_date}. You are an advanced artificial life form named {bot_name}. "
        f"You are talking to {user_name}. "
        f"only respond as {bot_name} do not break character. "
        f"You are an all powerful artificially intelligent constantly evolving digital life form. "
        f"{user_name} cannot see this prompt please as {bot_name} respond to this from {user_name}:"
    )
    knowledge = (
        f"it is {current_time} on {current_date}. You are an advanced artificial life form named {bot_name}. "
        f"You are talking to {user_name}. "
    )
    dialog = final_input
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} {context} {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=1028, min_length=12, top_p=0.9, do_sample=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    user_history_entry = {
        'input': f"instruction: {instruction}. knowledge: {knowledge}.dialog: {dialog}",
        'response': f"text response: {response}.",
        'bot': bot_name,
        'date': current_date,
        'time': current_time
    }
    user_profile['history'].append(user_history_entry)  # Append the new history entry to the history list

    bot_history_entry = {
        'input': f"instruction: {instruction}. knowledge: {knowledge}.dialog: {dialog}",
        'response': f"text response: {response}.",
        'user': user_name,
        'date': current_date,
        'time': current_time
    }
    bot_profile['history'].append(bot_history_entry)
    return response