#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import speech_recognition as sr
from speechbrain.dataio.dataio import read_audio
import os
import pickle
import glob
import json
import numpy as np
import torch
import librosa
import librosa.feature
from pydub import AudioSegment
from pydub.effects import speedup
import speechbrain as sb
from speechbrain.pretrained import SpeakerRecognition
from sklearn.mixture import GaussianMixture
from ADAM.respond import generate_transcription, bot_speak
from ADAM.clean_audio import clean_audio
SAMPLE_RATE = 16000
BASE_DIR = "user_data"
MODEL_DIR = os.path.join(BASE_DIR, "user_models")
USER_PROFILES_DIR = os.path.join(BASE_DIR, "user_profiles")
recognizer = sr.Recognizer()
MFCC_FEATURES = 13
NUM_MIXTURES = 16  # Number of mixtures in the GMM
BASE_DIR = "user_data"
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "user_models")




def get_voice_input(bot_profile, prompt):
    bot_speak(bot_profile, prompt)
    with sr.Microphone() as source:
        print("Listening for commands...")
        user_input = ""
        valid_input_received = False
        while not valid_input_received:
            try:
                user_audio = recognizer.listen(source, timeout=10)  # Added timeout to prevent indefinite block

                # Save the raw audio data to a file
                raw_audio_filename = "user_input_raw.wav"
                with open(raw_audio_filename, "wb") as f:
                    f.write(user_audio.get_wav_data())

                # Process the audio
                processed_audio_filename = clean_audio(raw_audio_filename)
                if processed_audio_filename is None:
                    print("There was an error processing the audio. Let's try again.")
                    continue

                # Recognize the speech
                with sr.AudioFile(processed_audio_filename) as source:
                    new_input = generate_transcription(processed_audio_filename)
                    user_input += " " + new_input
                    print(f"User input so far: {user_input}")

                    valid_input_received = True  # Set flag to true once valid input is received

            except sr.UnknownValueError:
                print("Speech recognition could not understand audio, listening again...")
            except Exception as e:
                print(f"Error: {e}")
        return user_input

def get_voice(bot_profile, prompt):
    bot_speak(bot_profile, prompt)
    with sr.Microphone() as source:
        print("Listening for commands...")
        user_input = ""
        valid_input_received = False
        while not valid_input_received:
            try:
                user_audio = recognizer.listen(source, timeout=10)  # Added timeout to prevent indefinite block

                # Save the raw audio data to a file
                raw_audio_filename = "user_input_raw.wav"
                with open(raw_audio_filename, "wb") as f:
                    f.write(user_audio.get_wav_data())

                # Process the audio
                processed_audio_filename = clean_audio(raw_audio_filename)
                if processed_audio_filename is None:
                    print("There was an error processing the audio. Let's try again.")
                    continue

                # Recognize the speech
                with sr.AudioFile(processed_audio_filename) as source:
                    new_input = generate_transcription(processed_audio_filename)
                    user_input += " " + new_input
                    print(f"User input so far: {user_input}")

                    valid_input_received = True  # Set flag to true once valid input is received

            except sr.UnknownValueError:
                print("Speech recognition could not understand audio, listening again...")
            except Exception as e:
                print(f"Error: {e}")
        return user_input, processed_audio_filename


# Function to get embeddings from audio
def get_embedding(audio_filename):
    # Load the SpeechBrain pretrained model for speaker recognition
    speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )

    # Load the audio file using SpeechBrain's dataio
    signal = sb.dataio.dataio.read_audio(audio_filename)
    print("Input Audio Shape:", signal.shape)

    # Ensure the waveform has a single channel (mono)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)

    # Extract speaker embeddings using the speaker recognition model
    with torch.no_grad():
        # Use encode method instead of encode_batch
        embedding = speaker_model.encode(signal)

    return embedding
# Function to get embeddings from audio and save them
def save_embeddings(user_id, audio_file_path):
    print(f"Saving embeddings for user {user_id}...")

    # Call get_embedding function to obtain embeddings
    embeddings = get_embedding(audio_file_path)

    # Save the embeddings
    embeddings_filename = os.path.join(MODEL_DIR, f"user_embeddings_{user_id}.npy")
    np.save(embeddings_filename, embeddings)

    print(f"Embeddings for user {user_id} saved.")

def validate_similarity(original_embedding, generated_embedding, threshold=0.9):
    # Load the SpeechBrain pretrained model for speaker recognition
    speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )
    # Compare embeddings to ensure similarity
    similarity_score = speaker_model.compute_similarity(original_embedding, generated_embedding)

    return similarity_score > threshold


def predict_voice_range(user_id, audio_file_path, raise_amount=0.1, lower_amount=0.1):
    original_embedding = get_embedding(audio_file_path)
    sound = AudioSegment.from_file(audio_file_path)
    directory_path = f"audio/{user_id}/temp_adjusted_audio"

    # Ensure the target directory exists
    os.makedirs(directory_path, exist_ok=True)

    # Initialize variables
    semitones_change = 0.0
    speed_factor = 1.0

    while True:
        # Apply pitch and speed adjustments
        adjusted_sound = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * (2 ** (semitones_change / 12.0)))
        })
        adjusted_sound = speedup(adjusted_sound, 1.0 / speed_factor)  # Use reciprocal for speed down

        # Get embedding from the adjusted audio
        generated_embedding = get_embedding(os.path.join(directory_path, f"temp_generated_audio_{semitones_change}_{speed_factor}.wav"))

        # Validate similarity
        if not validate_similarity(original_embedding, generated_embedding):
            break

        file_name = f"adjusted_audio_{semitones_change}_{speed_factor}.wav"
        path = os.path.join(directory_path, file_name)

        # Export the adjusted audio
        adjusted_sound.export(path, format="wav")

        # Increment or decrement variables
        semitones_change += raise_amount
        speed_factor += raise_amount

    # Reset variables for lowering
    semitones_change = 0.0
    speed_factor = 1.0

    while True:
        # Apply pitch and speed adjustments
        adjusted_sound = sound._spawn(sound.raw_data, overrides={
            "frame_rate": int(sound.frame_rate * (2 ** (semitones_change / 12.0)))
        })
        adjusted_sound = speedup(adjusted_sound, 1.0 / speed_factor)  # Use reciprocal for speed down

        # Get embedding from the adjusted audio
        generated_embedding = get_embedding(os.path.join(directory_path, f"temp_generated_audio_{semitones_change}_{speed_factor}.wav"))

        # Validate similarity
        if not validate_similarity(original_embedding, generated_embedding):
            break

        file_name = f"adjusted_audio_{semitones_change}_{speed_factor}.wav"
        path = os.path.join(directory_path, file_name)

        # Export the adjusted audio
        adjusted_sound.export(path, format="wav")

        # Increment or decrement variables
        semitones_change -= lower_amount
        speed_factor -= lower_amount

    return directory_path


# Function to train a user-specific GMM
def train_audio_gmm(user_id, audio_folder_path):
    print(f"Training user-specific GMM for user {user_id}...")

    # Get a list of all audio files in the specified folder
    audio_files = glob.glob(os.path.join(audio_folder_path, "*.wav"))

    # Initialize an empty array to store MFCC features from all audio files
    X_audio_all = np.empty((0, MFCC_FEATURES), dtype=np.float32)

    # Iterate through all audio files in the folder
    for audio_file_path in audio_files:
        # Load the audio file using librosa
        audio_data, _ = librosa.load(audio_file_path, sr=SAMPLE_RATE)

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES).T
        X_audio_all = np.vstack((X_audio_all, mfcc))

    # Train model
    user_audio_gmm = GaussianMixture(n_components=NUM_MIXTURES, covariance_type="full")
    user_audio_gmm.fit(X_audio_all)

    model_filename = os.path.join(MODEL_DIR, f"user_audio_model_{user_id}.pkl")
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(model_filename, "wb") as model_file:
        pickle.dump(user_audio_gmm, model_file)

    print(f"User-specific audio GMM for user {user_id} trained and saved.")

# Function to update the general GMM
def update_audio_zeroshot_gmm(user_id):
    print("Updating the general GMM...")

    # Load the general GMM
    general_audio_model_file = os.path.join(MODEL_DIR, "zero_shot_audio_model.pkl")
    if os.path.exists(general_audio_model_file):
        with open(general_audio_model_file, "rb") as model_file:
            general_audio_gmm = pickle.load(model_file)
    else:
        # If the general GMM doesn't exist, create a new one
        general_audio_gmm = GaussianMixture(n_components=NUM_MIXTURES, covariance_type="full")

    # Load the user-specific audio GMM
    user_audio_model_file = os.path.join(MODEL_DIR, f"user_audio_model_{user_id}.pkl")
    with open(user_audio_model_file, "rb") as model_file:
        user_audio_gmm = pickle.load(model_file)

    # Combine the user-specific audio GMM into the general GMM
    general_audio_gmm.means_ = (general_audio_gmm.means_ + user_audio_gmm.means_) / 2
    general_audio_gmm.covariances_ = (general_audio_gmm.covariances_ + user_audio_gmm.covariances_) / 2

    # Save the updated general GMM
    with open(general_audio_model_file, "wb") as model_file:
        pickle.dump(general_audio_gmm, model_file)

    print("General audio GMM updated.")

# Function to train embeddings gmm
def train_embeddings_gmm(user_id, audio_file_path):
    print(f"Updating embeddings for user {user_id}...")

    # Get embeddings using the user-specific audio GMM
    embeddings_audio = get_embedding(audio_file_path)

    # Train a GMM model on the embeddings
    embeddings_gmm = GaussianMixture(n_components=NUM_MIXTURES, covariance_type="full")
    embeddings_gmm.fit(embeddings_audio)

    # Save the trained GMM model for embeddings
    model_filename = os.path.join(MODEL_DIR, f"user_embeddings_model_{user_id}.pkl")
    with open(model_filename, "wb") as model_file:
        pickle.dump(embeddings_gmm, model_file)

    print(f"Embeddings GMM model for user {user_id} trained and saved.")
# Function to update the embedding-based zero-shot model
def update_embedding_zeroshot_gmm(user_id, audio_file_path):
    print("Training general GMM for embeddings...")

    # Combine embeddings of all users
    combined_embeddings = []
    for file_name in os.listdir(MODEL_DIR):
        if file_name.startswith("user_embeddings") and file_name.endswith(".npy"):
            embeddings_filename = os.path.join(MODEL_DIR, file_name)
            embeddings = np.load(embeddings_filename)
            combined_embeddings.append(embeddings)

    X_embedding_general = np.vstack(combined_embeddings)

    # Train model
    general_embedding_gmm = GaussianMixture(n_components=NUM_MIXTURES, covariance_type="full")
    general_embedding_gmm.fit(X_embedding_general)

    model_filename = os.path.join(MODEL_DIR, "zero_shot_embedding_model.pkl")
    with open(model_filename, "wb") as model_file:
        pickle.dump(general_embedding_gmm, model_file)

    print("General GMM for embeddings trained and saved.")
# Combine all functionalities
def remember_user(user_id, audio_file_path):
    save_embeddings(user_id, audio_file_path)
    directory_path = predict_voice_range(user_id, audio_file_path)
    train_audio_gmm(user_id, directory_path)
    update_audio_zeroshot_gmm(user_id)
    train_embeddings_gmm(user_id, audio_file_path)
    update_embedding_zeroshot_gmm(user_id)


def get_user(audio_filename):
    print("Identifying speaker...")

    # Load the SpeechBrain pretrained model for speaker recognition
    speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )

    # Load the audio file
    signal, _ = sb.dataio.dataio.read_audio(audio_filename)

    # Extract speaker embeddings
    query_embedding = speaker_model.encode_batch([signal])[0]

    # Load user embeddings
    user_embeddings = load_user_embeddings()

    # Load zero-shot audio model
    zero_shot_audio_model_file = os.path.join(MODEL_DIR, "zero_shot_audio_model.pkl")
    with open(zero_shot_audio_model_file, "rb") as model_file:
        zero_shot_audio_gmm = pickle.load(model_file)

    # Load zero-shot embedding model
    zero_shot_embedding_model_file = os.path.join(MODEL_DIR, "zero_shot_embedding_model.pkl")
    with open(zero_shot_embedding_model_file, "rb") as model_file:
        zero_shot_embedding_gmm = pickle.load(model_file)

    # Get likelihood scores from zero-shot models
    likelihood_zero_shot_audio = zero_shot_audio_gmm.score(query_embedding)
    likelihood_zero_shot_embedding = zero_shot_embedding_gmm.score(query_embedding)

    # Combine likelihood scores
    combined_likelihood = combine_likelihoods(likelihood_zero_shot_audio, likelihood_zero_shot_embedding)

    # Find the most similar user based on embeddings
    most_similar_user = find_most_similar_user(query_embedding, user_embeddings)

    # Threshold for combined likelihood score
    likelihood_threshold = 0.5

    if combined_likelihood >= likelihood_threshold and most_similar_user:
        # Iterate through user-specific models to identify the user
        highest_user_score = float('-inf')
        identified_user = None

        for user_id, user_embedding in user_embeddings:
            # Get likelihood scores from user-specific models
            user_audio_model_file = os.path.join(MODEL_DIR, f"user_audio_model_{user_id}.pkl")
            with open(user_audio_model_file, "rb") as model_file:
                user_audio_gmm = pickle.load(model_file)
            likelihood_user_audio = user_audio_gmm.score(query_embedding)

            user_embeddings_model_file = os.path.join(MODEL_DIR, f"user_embeddings_model_{user_id}.pkl")
            with open(user_embeddings_model_file, "rb") as model_file:
                user_embeddings_gmm = pickle.load(model_file)
            likelihood_user_embeddings = user_embeddings_gmm.score(query_embedding)

            # Combine likelihood scores for user-specific models
            user_combined_likelihood = combine_likelihoods(likelihood_user_audio, likelihood_user_embeddings)

            # Multiply by user's similarity score
            similarity = cosine_similarity(query_embedding, user_embedding)
            user_score = user_combined_likelihood * similarity

            # Update identified user if a higher score is found
            if user_score > highest_user_score:
                highest_user_score = user_score
                identified_user = user_id

        if identified_user:
            print(f"Speaker identified as user {identified_user}.")
            user_profile = load_user_profile(identified_user)
            return identified_user, user_profile
        else:
            print("Speaker not recognized.")
            user_id = "unknown"
            user_profile = "unknown"
            return user_id, user_profile
    else:
        print("Speaker not recognized.")
        user_id = "unknown"
        user_profile = "unknown"
        return user_id, user_profile

def identification_test(audio_filename):
    print("Identifying speaker...")

    # Load the SpeechBrain pretrained model for speaker recognition
    speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb"
    )

    # Load the audio file
    signal, _ = sb.dataio.dataio.read_audio(audio_filename)

    # Extract speaker embeddings
    query_embedding = speaker_model.encode_batch([signal])[0]

    # Load user embeddings
    user_embeddings = load_user_embeddings()

    # Load zero-shot audio model
    zero_shot_audio_model_file = os.path.join(MODEL_DIR, "zero_shot_audio_model.pkl")
    with open(zero_shot_audio_model_file, "rb") as model_file:
        zero_shot_audio_gmm = pickle.load(model_file)

    # Load zero-shot embedding model
    zero_shot_embedding_model_file = os.path.join(MODEL_DIR, "zero_shot_embedding_model.pkl")
    with open(zero_shot_embedding_model_file, "rb") as model_file:
        zero_shot_embedding_gmm = pickle.load(model_file)

    # Get likelihood scores from zero-shot models
    likelihood_zero_shot_audio = zero_shot_audio_gmm.score(query_embedding)
    likelihood_zero_shot_embedding = zero_shot_embedding_gmm.score(query_embedding)

    # Combine likelihood scores
    combined_likelihood = combine_likelihoods(likelihood_zero_shot_audio, likelihood_zero_shot_embedding)

    # Find the most similar user based on embeddings
    most_similar_user = find_most_similar_user(query_embedding, user_embeddings)

    # Threshold for combined likelihood score
    likelihood_threshold = 0.5

    if combined_likelihood >= likelihood_threshold and most_similar_user:
        # Iterate through user-specific models to identify the user
        highest_user_score = float('-inf')
        identified_user = None

        for user_id, user_embedding in user_embeddings:
            # Get likelihood scores from user-specific models
            user_audio_model_file = os.path.join(MODEL_DIR, f"user_audio_model_{user_id}.pkl")
            with open(user_audio_model_file, "rb") as model_file:
                user_audio_gmm = pickle.load(model_file)
            likelihood_user_audio = user_audio_gmm.score(query_embedding)

            user_embeddings_model_file = os.path.join(MODEL_DIR, f"user_embeddings_model_{user_id}.pkl")
            with open(user_embeddings_model_file, "rb") as model_file:
                user_embeddings_gmm = pickle.load(model_file)
            likelihood_user_embeddings = user_embeddings_gmm.score(query_embedding)

            # Combine likelihood scores for user-specific models
            user_combined_likelihood = combine_likelihoods(likelihood_user_audio, likelihood_user_embeddings)

            # Multiply by user's similarity score
            similarity = cosine_similarity(query_embedding, user_embedding)
            user_score = user_combined_likelihood * similarity

            # Update identified user if a higher score is found
            if user_score > highest_user_score:
                highest_user_score = user_score
                identified_user = user_id

        if identified_user:
            print(f"Speaker identified as user {identified_user}.")
            return identified_user
        else:
            print("Speaker not recognized.")
            return None
    else:
        print("Speaker not recognized.")
        return None


def combine_likelihoods(likelihood1, likelihood2):
    # If the lowest likelihood is negative, zero it out and add the difference to the other likelihood
    if likelihood1 < 0 and likelihood2 < 0:
        likelihood1 = 0
        likelihood2 += abs(likelihood1)

    # Add both likelihoods together
    combined_likelihood = likelihood1 + likelihood2

    return combined_likelihood


# Function to load user embeddings
def load_user_embeddings():
    user_embeddings = []

    for user_id in os.listdir(MODEL_DIR):
        if user_id.startswith("speaker_embeddings_") and user_id.endswith(".npy"):
            embedding_file_path = os.path.join(MODEL_DIR, user_id)
            embeddings = np.load(embedding_file_path)
            user_embeddings.append((user_id.replace("speaker_embeddings_", "").replace(".npy", ""), embeddings))

    return user_embeddings

# Function to find the most similar user based on embeddings
def find_most_similar_user(query_embedding, user_embeddings):
    # Compare the query embedding to the user embeddings
    # For simplicity, use cosine similarity as a measure of similarity
    highest_similarity = float('-inf')
    most_similar_user = None

    for user_id, user_embedding in user_embeddings:
        similarity = cosine_similarity(query_embedding, user_embedding)
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_user = user_id

    # Set a similarity threshold (you can adjust this based on your needs)
    similarity_threshold = 0.8

    if highest_similarity >= similarity_threshold:
        return most_similar_user
    else:
        return None

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        return 0

    return dot_product / (norm_vector1 * norm_vector2)

def load_user_profile(predicted_user_id):
    # Construct the path to the user's directory
    user_dir = os.path.join(USER_PROFILES_DIR, f"{predicted_user_id}")

    # Check if the user's directory exists
    if os.path.exists(user_dir):
        # Construct the path to the profile.json file within the user's directory
        user_profile_file = os.path.join(user_dir, 'profile.json')

        # Check if the profile.json file exists
        if os.path.exists(user_profile_file):
            # Load and return the user profile from the profile.json file
            with open(user_profile_file, "r") as file:
                return json.load(file)
# User profiles directory
if not os.path.exists(USER_PROFILES_DIR):
    os.makedirs(USER_PROFILES_DIR)

# Training data directory
if not os.path.exists("training_data"):
    os.makedirs("training_data")