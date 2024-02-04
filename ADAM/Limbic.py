#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import os
import json
from collections import Counter
import torch.nn.functional as F
import torch
from ADAM.Emodel.models import HubertForSpeechClassification, EmoGpt
from transformers import ElectraTokenizer, ElectraForSequenceClassification, Wav2Vec2FeatureExtractor, AutoConfig, AutoProcessor
from pydub import AudioSegment
import numpy as np
segmented_audio_dir = "segmented_audio"

#Load Models
electratokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
electramodel = ElectraForSequenceClassification.from_pretrained('google/electra-small-discriminator')
hubertmodel = HubertForSpeechClassification.from_pretrained("Rajaram1996/Hubert_emotion")
hubertfeature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
emo_gpt_model = EmoGpt()
sampling_rate = 16000
hubertconfig = AutoConfig.from_pretrained("Rajaram1996/Hubert_emotion")

def speech_file_to_array(path, sampling_rate):
        sound = AudioSegment.from_file(path)
        sound = sound.set_frame_rate(sampling_rate)
        sound_array = np.array(sound.get_array_of_samples())
        return sound_array
def text_emotion(txt):
    print(f"\n{txt}\n")

    # Tokenize the input for Electra
    inputs = electratokenizer(txt, return_tensors="pt")

    # Use Electra for sentiment and emotion analysis
    with torch.no_grad():
        outputs = electramodel(**inputs)

    # Extract sentiment from Electra's output
    sentiment_score = torch.argmax(outputs.logits).item()

    # Assuming you have a mapping from score to sentiment (e.g., 0 for negative, 1 for neutral, 2 for positive)
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment = sentiment_mapping[sentiment_score]

    # Examples for emotion analysis
    emotion_examples = [
        {"text": "What a beautiful day!", "label": "Positive, Happy"},
        {"text": "The news was disheartening.", "label": "Negative, Sad"},
        {"text": "I can't wait for the concert!", "label": "Positive, Excited"},
        {"text": "The decision was unacceptable!", "label": "Negative, Angry"},
        {"text": "The situation is quite concerning.", "label": "Negative, Worried"},
        {"text": "This discovery is amazing!", "label": "Positive, Excited"},
        {"text": "The party was a blast.", "label": "Positive, Joyful"},
        {"text": "The waiting is nerve-wracking.", "label": "Negative, Anxious"},
        {"text": "I don't have a strong opinion on that.", "label": "Neutral, Indifferent"},
        {"text": "There's just too much going on right now.", "label": "Negative, Overwhelmed"},
        {"text": "Old songs bring back memories.", "label": "Positive, Nostalgic"},
        {"text": "A day at the spa was just what I needed.", "label": "Positive, Relaxed"},
        {"text": "The new project sparks hope.", "label": "Positive, Hopeful"},
        {"text": "The constant interruptions are annoying.", "label": "Negative, Frustrated"},
        {"text": "There's absolutely nothing to do.", "label": "Negative, Bored"},
        {"text": "The mystery novel was captivating.", "label": "Positive, Curious"},
        {"text": "The mistake was humiliating.", "label": "Negative, Ashamed"},
        {"text": "Winning the award was an honor.", "label": "Positive, Proud"},
        {"text": "The accident was regrettable.", "label": "Negative, Guilty"},
        {"text": "The surprise party was a shock.", "label": "Positive, Surprised"},
        {"text": "The movie didn't live up to expectations.", "label": "Negative, Disappointed"},
        {"text": "The vacation was a peaceful retreat.", "label": "Positive, Content"},
        {"text": "The new schedule is so much better.", "label": "Positive, Satisfied"},
        {"text": "I'm overjoyed with the news!", "label": "Positive, Happy"},
        {"text": "I find the situation quite depressing.", "label": "Negative, Sad"},
        {"text": "Looking forward to the event!", "label": "Positive, Excited"},
        {"text": "I can't believe the audacity!", "label": "Negative, Angry"},
        {"text": "This issue is really bothering me.", "label": "Negative, Worried"},
        {"text": "Discovering new places is thrilling!", "label": "Positive, Excited"},
        {"text": "Had a great time at the event.", "label": "Positive, Joyful"},
        {"text": "The wait for the results is stressing me out.", "label": "Negative, Anxious"},
        {"text": "I don't really care about that.", "label": "Neutral, Indifferent"},
        {"text": "I'm swamped with work.", "label": "Negative, Overwhelmed"},
        {"text": "Old movies are quite nostalgic.", "label": "Positive, Nostalgic"},
        {"text": "A quiet evening is very relaxing.", "label": "Positive, Relaxed"},
        {"text": "The new developments are encouraging.", "label": "Positive, Hopeful"},
        {"text": "The interruptions are getting on my nerves.", "label": "Negative, Frustrated"},
        {"text": "I've nothing interesting to do.", "label": "Negative, Bored"},
        {"text": "The book was very intriguing.", "label": "Positive, Curious"},
        {"text": "The incident was very embarrassing.", "label": "Negative, Ashamed"},
        {"text": "I'm proud of my achievements.", "label": "Positive, Proud"},
        {"text": "I regret my actions.", "label": "Negative, Guilty"},
        {"text": "The surprise visit was astonishing.", "label": "Positive, Surprised"},
        {"text": "The show was a letdown.", "label": "Negative, Disappointed"},
        {"text": "The retreat was serene.", "label": "Positive, Content"},
        {"text": "The changes are satisfactory.", "label": "Positive, Satisfied"},
        {"text": "I'm annoyed I can't find my wallet.", "label": "Negative, Frustrated"},
        {"text": "The lecture was insightful.", "label": "Positive, Interested"},
        {"text": "This is baffling!", "label": "Negative, Confused"},
        {"text": "The joke was funny.", "label": "Positive, Amused"},
        {"text": "This is a frightening situation.", "label": "Negative, Fearful"},
        {"text": "The meal was delightful.", "label": "Positive, Pleased"},
        {"text": "I'm infuriated with the traffic.", "label": "Negative, Irritated"},
        {"text": "I can't find my keys anywhere.", "label": "Negative, Frustrated"},
        {"text": "The discussion was quite enlightening.", "label": "Positive, Interested"},
        {"text": "I don't understand this at all.", "label": "Negative, Confused"},
        {"text": "The dog's reaction was hilarious.", "label": "Positive, Amused"},
        {"text": "This situation is really scary.", "label": "Negative, Fearful"},
        {"text": "The cake was delicious.", "label": "Positive, Pleased"},
        {"text": "The traffic is terrible.", "label": "Negative, Irritated"}
    ]

    # Map emotion labels to scores
    emotion_mapping = {}
    for example in emotion_examples:
        text = example["text"]
        labels = example["label"].split(", ")
        for label in labels:
            emotion_mapping[label] = text

    # Tokenize emotion examples
    emotion_inputs = electratokenizer(list(emotion_mapping.values()), return_tensors="pt", padding=True, truncation=True)

    # Use Electra for emotion analysis
    with torch.no_grad():
        emotion_outputs = electramodel(**emotion_inputs)

    # Extract emotion from Electra's output (this is a simplified example)
    emotion_scores = torch.argmax(emotion_outputs.logits, dim=1)

    # Map emotion scores to labels
    emotions = [key for key, value in emotion_mapping.items() if value == emotion_scores.item()]

    print(f"Sentiment: {sentiment}")
    print(f"Emotions: {emotions}")

    return sentiment, emotions

def audio_emotion(audio_file):
     # Preprocess audio file for Hubert
    sound_array = speech_file_to_array(audio_file, sampling_rate)
    inputs = hubertfeature_extractor(sound_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to("cpu").float() for key in inputs}

    # Make predictions using Hubert
    with torch.no_grad():
        logits = hubertmodel(**inputs).logits

    # Process Hubert's output
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{
        "emo": hubertconfig.id2label[i],
        "score": round(score * 100, 1)}
        for i, score in enumerate(scores)
    ]
    
    # Filter and sort the predictions
    result = [row for row in sorted(outputs, key=lambda x: x["score"], reverse=True) if row['score'] != '0.0%'][:2]

    return result

def update_emotional_history(bot_profile, user_profile, sentiment, emotion, emotion2):
    emotion_histories_dir = os.path.join('bot_profiles', bot_profile['name'], 'emotion_histories')
    os.makedirs(emotion_histories_dir, exist_ok=True)

    emotion_history_file = os.path.join(emotion_histories_dir, f"{user_profile['id']}.json")
    if os.path.exists(emotion_history_file):
        with open(emotion_history_file, 'r') as file:
            user_emotional_history = json.load(file)
    else:
        user_emotional_history = {
            'sentiments': {},
            'emotion1': {},
            'emotion2': {}
        }

    # Trim any extra characters like '.' from sentiment and emotion variables
    sentiment = sentiment.strip('.')
    emotion = emotion.strip('.')
    emotion2 = emotion2.strip('.')

    # Increment sentiment, emotion, and emotion2 counts
    user_emotional_history['sentiments'][sentiment] = user_emotional_history['sentiments'].get(sentiment, 0) + 1
    user_emotional_history['emotion1'][emotion] = user_emotional_history['emotion1'].get(emotion, 0) + 1
    user_emotional_history['emotion2'][emotion2] = user_emotional_history['emotion2'].get(emotion2, 0) + 1

    with open(emotion_history_file, 'w') as file:
        json.dump(user_emotional_history, file)

    return emotion_history_file

def determine_bot_feelings(dialogue, common_emotions,common_sentiment, sentiment, emotion, emotion2, bot_profile, user_profile):
    user_name = user_profile['name']
    bot_name = bot_profile['name']
    user_bio = user_profile['bio']
    bot_bio = bot_profile['bio']
    # Create a prompt
    prompt = (
        f"{dialogue}. "
        f"{user_name} is {user_bio} and {bot_name} is {bot_bio}. "
        f"{user_name} usually acts {common_emotions} and {common_sentiment} towards {bot_name}. "
        f"{user_name} is currently being {sentiment}, the words they used expressed {emotion}, but the sound of their voice expressed {emotion2}. " 
        f"How would {bot_name} likely feel about {user_name} and what would {bot_name}'s current emotions be?"
    )

    # Use EmoGpt to predict bot feelings and emotion
    bot_feelings = emo_gpt_model.generate(prompt)

    return bot_feelings

def handle_emotion(dialogue, bot_profile, user_profile, sentiment, emotion, emotion2):
    emotion_history_file=update_emotional_history(bot_profile, user_profile, sentiment, emotion, emotion2)


    with open(emotion_history_file, 'r') as file:
        user_emotional_history = json.load(file)

    sentiments = user_emotional_history.get('sentiments', {})
    emotions = user_emotional_history.get('emotions', {})

    if sentiments:
        common_sentiment = max(sentiments, key=sentiments.get)
    else:
        common_sentiment = None  # or some default value

    emotion_counter = Counter(emotions)
    common_emotions = emotion_counter.most_common(3)
    feelings=determine_bot_feelings(dialogue, common_emotions, common_sentiment, sentiment, emotion, emotion2, bot_profile, user_profile)

    return feelings


def find_matching_files(user_name):
    for file_name in os.listdir(segmented_audio_dir):
        if file_name.startswith(user_name):
            base_name, extension = os.path.splitext(file_name)
            separator_index = base_name.rfind('_')

            if separator_index != -1:
                user_name_from_file = base_name[:separator_index]
                text_path = os.path.join(segmented_audio_dir, f"{user_name_from_file}.txt")
                audio_path = os.path.join(segmented_audio_dir, f"{user_name_from_file}.wav")

                if os.path.exists(text_path) and os.path.exists(audio_path):
                    return audio_path, text_path

    return None, None  # Return None if matching files are not found

def limbic(user_profile, bot_profile, dialogue):
    # Find the matching audio and text files in the segmented_audio_dir
    user_audio_path, user_text_path = find_matching_files(user_profile['name'])

    if user_audio_path is None or user_text_path is None:
        return None  # Handle case where matching files are not found

    # Load text from the matching text file
    with open(user_text_path, 'r') as text_file:
        user_text = text_file.read()

    # Perform sentiment and emotion analysis on the loaded text
    sentiment, emotion = text_emotion(user_text)

    # Analyze audio emotion
    emotion2 = audio_emotion(user_audio_path)

    # Handle emotions and update emotional history
    feelings = handle_emotion(dialogue, bot_profile, user_profile, sentiment, emotion, emotion2)

    return sentiment, emotion, emotion2, feelings
