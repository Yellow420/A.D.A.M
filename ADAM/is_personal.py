#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import fasttext
import os
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
gpttokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
gptmodel = GPT2LMHeadModel.from_pretrained("distilgpt2")


def is_personal(text):
    # Load the trained FastText model
    model = fasttext.load_model('models/intent_model.bin')

    # Predict the label for the given text
    prediction = model.predict(text)

    # Extract the predicted label from the tuple
    predicted_label = prediction[0][0]

    # Check if the predicted label is 'personal'
    return predicted_label == '__label__personal'

def generate_autobiography(user_profile, bot_profile):
    # Extract user_id from user_profile
    user_id = user_profile['id']
    digital_handshake = user_profile.get('digital_handshake', '')

    # Get the path to the user's personal info file in the current bot's directory
    personal_info_path = os.path.join('bot_profiles', bot_profile['name'], f'{user_id}_personal_info.json')

    # Load personal info
    with open(personal_info_path, 'r') as file:
        personal_info = json.load(file)

    # Generate prompt for AI21
    prompt = f"Generate an autobiography for {user_profile['name']} based on the following personal info:\n"
    prompt += f"{digital_handshake}\n"
    for key, value in personal_info.items():
        prompt += f"{key}: {value}\n"

        # Tokenize input and generate output
    input_ids = gpttokenizer.encode(prompt, return_tensors="pt")
    output = gptmodel.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode the generated output
    autobiography_text = gpttokenizer.decode(output[0], skip_special_tokens=True).strip()

    # Save the generated autobiography to the bot's directory
    autobiography_path = os.path.join('bot_profiles', bot_profile['name'], f'{user_id}_autobiography.txt')
    with open(autobiography_path, 'w') as file:
        file.write(autobiography_text)

    print(f"Autobiography for {user_profile['name']} has been generated and saved to {bot_profile['name']}'s directory.")


def save_personal_info(text, user_profile, bot_profile):
    # Path to the bot's directory
    bot_directory = f'./bot_profiles/{bot_profile["name"]}'

    # Each user will have a separate file within the bot's directory
    user_info_file_path = os.path.join(bot_directory, f'{user_profile["id"]}_personal_info.json')

    # Load existing user info if file exists, otherwise start with an empty dict
    if os.path.exists(user_info_file_path):
        with open(user_info_file_path, 'r') as file:
            user_info = json.load(file)
    else:
        user_info = {}

    # Append the new command to the user's info
    user_info.setdefault('personal_info', []).append(text)

    # Save the updated user info back to file
    with open(user_info_file_path, 'w') as file:
        json.dump(user_info, file, indent=4)

    # Call generate_autobiography after saving the personal info
    generate_autobiography(user_profile, bot_profile)