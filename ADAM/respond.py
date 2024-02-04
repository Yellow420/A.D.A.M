# Author: Chance Brownfield
# Email: ChanceBrownfield@protonmail.com
import tensorflow as tf
import tensorflow_hub as tfhub
from torchvision import transforms
from PIL import ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from moviepy.editor import *
import numpy as np
import math
import random
from glob import glob
import cv2
from langdetect import detect
import pygame
import os
import json
import wave
import pandas as pd
import speech_recognition as sr
import pyttsx3
import gtts
import pyaudio
import sounddevice as sd
import torch
import torchaudio
import speechbrain as sb
from speechbrain.pretrained import EncoderDecoderASR, Tacotron2, HIFIGAN
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.dataio.batch import PaddedBatch
import datetime
import yaml
from torch.utils.data import DataLoader
from PIL import Image


voice_language_mapping = ({
    'English': {
        'gTTS': 'en',
        'pyttsx3': [
            'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_DAVID_11.0',
            'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-GB_HAZEL_11.0',
            'HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Voices\\Tokens\\TTS_MS_EN-US_ZIRA_11.0'
        ],
        'SpeechBrain': ['Tacotron2 + HIFIGAN']
    },
    'spanish': {'gTTS': 'es'},
    'french': {'gTTS': 'fr'},
    'afrikaans': {'gTTS': 'af'},
    'arabic': {'gTTS': 'ar'},
    'bulgarian': {'gTTS': 'bg'},
    'bengali': {'gTTS': 'bn'},
    'bosnian': {'gTTS': 'bs'},
    'catalan': {'gTTS': 'ca'},
    'czech': {'gTTS': 'cs'},
    'danish': {'gTTS': 'da'},
    'german': {'gTTS': 'de'},
    'greek': {'gTTS': 'el'},
    'estonian': {'gTTS': 'et'},
    'finnish': {'gTTS': 'fi'},
    'gujarati': {'gTTS': 'gu'},
    'croatian': {'gTTS': 'hr'},
    'hungarian': {'gTTS': 'hu'},
    'indonesian': {'gTTS': 'id'},
    'icelandic': {'gTTS': 'is'},
    'italian': {'gTTS': 'it'},
    'hebrew': {'gTTS': 'iw'},
    'japanese': {'gTTS': 'ja'},
    'javanese': {'gTTS': 'jw'},
    'khmer': {'gTTS': 'km'},
    'kannada': {'gTTS': 'kn'},
    'korean': {'gTTS': 'ko'},
    'latin': {'gTTS': 'la'},
    'latvian': {'gTTS': 'lv'},
    'malayalam': {'gTTS': 'ml'},
    'marathi': {'gTTS': 'mr'},
    'malay': {'gTTS': 'ms'},
    'myanmar': {'gTTS': 'my'},
    'nepali': {'gTTS': 'ne'},
    'dutch': {'gTTS': 'nl'},
    'norwegian': {'gTTS': 'no'},
    'polish': {'gTTS': 'pl'},
    'portuguese': {'gTTS': 'pt'},
    'romanian': {'gTTS': 'ro'},
    'russian': {'gTTS': 'ru'},
    'sinhala': {'gTTS': 'si'},
    'slovak': {'gTTS': 'sk'},
    'albanian': {'gTTS': 'sq'},
    'serbian': {'gTTS': 'sr'},
    'sundanese': {'gTTS': 'su'},
    'swedish': {'gTTS': 'sv'},
    'swahili': {'gTTS': 'sw'},
    'tamil': {'gTTS': 'ta'},
    'telugu': {'gTTS': 'te'},
    'thai': {'gTTS': 'th'},
    'filipino': {'gTTS': 'tl'},
    'turkish': {'gTTS': 'tr'},
    'ukrainian': {'gTTS': 'uk'},
    'urdu': {'gTTS': 'ur'},
    'vietnamese': {'gTTS': 'vi'},
    'chinese_simplified': {'gTTS': 'zh-CN'},
    'chinese_traditional': {'gTTS': 'zh-TW'},
    'chinese_mandarin': {'gTTS': 'zh'},
})

def select_voice():
    recognizer = sr.Recognizer()
    engine = pyttsx3.init()

    while True:  # keep asking until a valid language is entered or detected
        language_input = input("Enter your language (e.g., English, Spanish, French) or speak your language and I'll try to detect it: ").strip().lower()
        if language_input:
            detected_language = detect(language_input)
            if detected_language in voice_language_mapping:
                language = detected_language
                break
            else:
                print(f"I'm sorry, I couldn't identify the language. Please try again.")
        else:
            print("Please enter or speak a language.")

    language_voices = voice_language_mapping[language]

    while True:  # keep trying voices until the user says 'yes'
        for voicelib, voices in language_voices.items():
            for voiceset in voices:
                if voicelib == 'pyttsx3':
                    engine.setProperty('voice', voiceset)
                    engine.say("Do you like this voice?")
                    engine.runAndWait()
                elif voicelib == 'gTTS':
                    tts = gtts.gTTS("Do you like this voice?", lang=voice_language_mapping[language]['gTTS'])
                    tts.save('voice_sample.mp3')
                    os.system("mpg123 voice_sample.mp3")
                elif voicelib == 'SpeechBrain':
                    # load the pre-trained TTS model as per SpeechBrain documentation
                    tts_model = sb.models.TTS.from_hparams(source="speechbrain/tts-waveglow-ljspeech", savedir="tmpdir_tts")
                    mel_output, mel_length, alignment = tts_model.encode_text("Do you like this voice?")
                    waveforms = tts_model.decode_batch(mel_output)
                    # play the audio
                    sd.play(waveforms.squeeze(1).cpu().numpy(), samplerate=22050)
                    sd.wait()

                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, timeout=10)
                    response = recognizer.recognize_google(audio).strip().lower()

                    if 'yes' in response:
                        return voicelib, voiceset


def load_audio(item):
    waveform, sample_rate = torchaudio.load(item['file_path'])
    return {'waveform': waveform, 'sample_rate': sample_rate}


def generate_transcription(audio_file_path):
    # Load the pre-trained ASR model from SpeechBrain
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr")

    # Transcribe the audio file
    transcription = asr_model.transcribe_file(audio_file_path)

    return transcription

def copy_voice(bot_profile, response):
    bot_name = bot_profile['name']
    voice_folder_path = os.path.join(os.getcwd(), 'voice')
    bot_profile_path = os.path.join(os.getcwd(), f'bot_profiles/{bot_name}')
    dataset_path = os.path.join(bot_profile_path, 'dataset')
    audio_path = os.path.join(dataset_path, 'audio')
    text_path = os.path.join(dataset_path, 'text.txt')
    model_path = os.path.join(bot_profile_path, 'model')
    # Create the dataset and audio directories if they don't already exist
    os.makedirs(audio_path, exist_ok=True)

    # Set the file path for the recorded audio
    voice_file_path = os.path.join(audio_path, f'{bot_name}.wav')

    # Set up the recording parameters
    p = pyaudio.PyAudio()
    fs = 44100  # Sample rate
    duration = 360  # Duration in seconds
    chunk_size = 1024
    channels = 2
    format = pyaudio.paInt16

    # Open a stream for recording
    stream = p.open(rate=fs, channels=channels, format=format, input=True, frames_per_buffer=chunk_size)

    print(f"Recording audio for {duration} seconds...")
    frames = []
    for i in range(0, int(fs / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a file
    with wave.open(voice_file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))

# Call the create_dataset function to process the new audio
    create_dataset(bot_profile)

    # Check if a model already exists for this bot
    if os.path.exists(model_path):
        # Use bot_speak to ask the user
        bot_speak(bot_profile,
                  "A model already exists for this bot. Would you like to merge the new voice with the old one? Say 'yes' to merge or 'no' to train a new model.")

        # Set up the recognizer and microphone for listening to the user's answer
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        # Use speech recognition to get the user's response
        try:
            response = recognizer.recognize_google(audio).lower()
            if 'yes' in response:
                # Call merge_model if the user says 'yes'
                merge_model(model_path, dataset_path)
            else:
                # Call train_model if the user's response is not 'yes'
                train_model(dataset_path)
        except sr.UnknownValueError:
            # Handle unrecognized speech
            bot_speak(bot_profile, "I didn't understand that. Please try again.")
        except sr.RequestError:
            # Handle a request error
            bot_speak(bot_profile, "I am not able to reach the speech service at the moment. Please try again later.")

def create_dataset(bot_profile):
    bot_name = bot_profile['name']
    voice_folder_path = os.path.join(os.getcwd(), 'voice')
    dataset_path = os.path.join(os.getcwd(), f'bot_profiles/{bot_name}', 'dataset')
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    timestamped_dataset_path = os.path.join(dataset_path, timestamp)
    os.makedirs(timestamped_dataset_path, exist_ok=True)

    voice_file_path = os.path.join(voice_folder_path, f'{bot_name}.wav')
    transcription = generate_transcription(voice_file_path)

    dataset_info = {
        'wav': os.path.abspath(voice_file_path),
        'transcription': transcription,
        'speaker': bot_name,
        'duration': float(get_duration(voice_file_path))  # Assuming get_duration is a function that retrieves the duration of an audio file
    }

    json_path = os.path.join(timestamped_dataset_path, 'data.json')
    with open(json_path, 'w') as json_file:
        json.dump([dataset_info], json_file, indent=4)

    print(f'Dataset JSON created for {bot_name} at {timestamped_dataset_path}')


def load_existing_data(bot_profile):
    bot_name = bot_profile['name']
    dataset_root_path = os.path.join(os.getcwd(), f'bot_profiles/{bot_name}/dataset')

    # Initialize lists for collecting dataset information
    audio_files = []
    transcriptions = []

    # Iterate over each timestamped folder in the dataset directory
    for timestamped_folder in sorted(os.listdir(dataset_root_path)):
        timestamped_path = os.path.join(dataset_root_path, timestamped_folder)
        audio_folder_path = os.path.join(timestamped_path, 'audio')
        text_file_path = os.path.join(timestamped_path, 'text.txt')

        # Check if it's a directory
        if os.path.isdir(audio_folder_path):
            # List audio files in the folder
            for audio_file in os.listdir(audio_folder_path):
                audio_file_path = os.path.join(audio_folder_path, audio_file)
                audio_files.append(audio_file_path)

                # Read the transcription from text.txt
                with open(text_file_path, 'r') as text_file:
                    transcription = text_file.read().strip()
                    transcriptions.append(transcription)

    # Create a DataFrame
    df = pd.DataFrame({'audio': audio_files, 'transcription': transcriptions})

    # Save the DataFrame to a CSV file
    csv_path = os.path.join(dataset_root_path, 'dataset.csv')
    df.to_csv(csv_path, index=False)

    # Load the dataset using SpeechBrain's DynamicItemDataset
    dataset = DynamicItemDataset.from_csv(csv_path)

    # Add audio reading function
    dataset.add_dynamic_item(read_audio, takes='audio', provides='signal', read_kwargs={'sr': None})

    # Add encoding of the transcriptions if needed
    encoder = CategoricalEncoder()  # Define this as per your requirements
    dataset.add_dynamic_item(encoder.encode, takes='transcription', provides='transcription_encoded')

    # Set output keys for batching
    dataset.set_output_keys(['id', 'signal', 'transcription_encoded'])

    return dataset

def get_duration(audio_file_path):
    with wave.open(audio_file_path, 'rb') as wave_file:
        frames = wave_file.getnframes()
        rate = wave_file.getframerate()
        duration = frames / float(rate)
        return duration


def train_model(bot_profile):
    class CustomBrain(sb.Brain):
        def compute_forward(self, batch: PaddedBatch, stage: sb.Stage) -> torch.Tensor:
            # Forward computation. Returns the output after processing.
            inputs = batch.input.to(self.device)
            predictions = self.modules.model(inputs)

            # Apply softmax if the output is logits
            if self.hparams.output_neurons_type == "logits":
                predictions = torch.nn.functional.softmax(predictions, dim=-1)

            return predictions

        def compute_objectives(self, predictions: torch.Tensor, batch: PaddedBatch, stage: sb.Stage) -> torch.Tensor:
            # Computes the loss given predictions and targets.
            targets = batch.target.to(self.device)
            loss = self.hparams.loss_function(predictions, targets)

            if stage != sb.Stage.TEST:
                # Add predictions and targets to the metrics for non-test stages
                self.metrics.add_batch(batch.id, predictions, targets)

            return loss

    # Load hyperparameters file with paths and training options
    # 'overrides' can be used to provide a dictionary of hyperparameter overrides
    hparams = sb.core.load_hyperpyyaml('path_to_hparams.yaml', overrides={})

    # Create datasets, dataloaders and the brain class following SpeechBrain standards
    train_dataset = create_dataset(bot_profile)  # Function should return a SpeechBrain-compatible dataset
    train_dataloader = sb.data_io.dataloader.SaveableDataLoader(train_dataset, batch_size=hparams['batch_size'])

    # Checkpointer for saving/restoring model and training state
    checkpointer = sb.utils.checkpoints.Checkpointer(checkpoint_dir=hparams['output_folder'])

    # Define the Brain object for training using the custom class
    brain = CustomBrain(
        modules=hparams['modules'],  # Model and other computational modules
        opt_class=hparams['opt_class'],  # Optimizer class
        hparams=hparams,
        run_opts=hparams['run_opts'],  # Running options, e.g., device placement
        checkpointer=checkpointer
    )

    # Train/validation/test loop
    # Define valid_set or test_set if available
    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=train_dataloader,
        valid_set=None,  # Replace with validation dataloader if available
        test_set=None,  # Replace with test dataloader if available
    )

    # Save model and training hyperparameters
    checkpointer.save_checkpoint()  # Saving the state of the model

def merge_model(bot_profile, new_audio_files, new_transcriptions):
    # Define paths
    data_path = f"bot_profiles/{bot_profile['name']}/dataset"
    output_path = f"bot_profiles/{bot_profile['name']}/models"

    # Load hyperparameters and modules
    hparams = sb.core.load_hyperpyyaml('path_to_hparams.yaml', overrides={})
    modules = hparams['modules']
    existing_model = modules['tacotron2']  # Assuming the Tacotron2 model is under 'modules' in hyperparams

    # Load the existing dataset and combine with new data
    existing_data = load_existing_data(bot_profile)  # This function needs to be defined
    combined_audio_files = existing_data['audio_files'] + new_audio_files
    combined_transcriptions = existing_data['transcriptions'] + new_transcriptions
    combined_dataset = create_dataset(combined_audio_files, combined_transcriptions)  # Ensure this returns a SpeechBrain compatible dataset

    # Create dataloader for the combined dataset
    combined_dataloader = DataLoader(combined_dataset, batch_size=hparams['batch_size'], shuffle=True)

    # Load the existing model checkpoint
    checkpointer = sb.utils.checkpoints.Checkpointer(checkpoint_dir=output_path)
    existing_model = checkpointer.load_checkpoint('model.ckpt', modules)['modules']['tacotron2']

    # Create a custom Brain class for fine-tuning
    class FineTuningBrain(sb.Brain):
        def compute_forward(self, batch: PaddedBatch, stage: sb.Stage) -> torch.Tensor:
            # Forward computation. Returns the output after processing.
            inputs = batch.input.to(self.device)
            predictions = self.modules.model(inputs)

            # Apply softmax if the output is logits
            if self.hparams.output_neurons_type == "logits":
                predictions = torch.nn.functional.softmax(predictions, dim=-1)

            return predictions

        def compute_objectives(self, predictions: torch.Tensor, batch: PaddedBatch, stage: sb.Stage) -> torch.Tensor:
            # Computes the loss given predictions and targets.
            targets = batch.target.to(self.device)
            loss = self.hparams.loss_function(predictions, targets)

            if stage != sb.Stage.TEST:
                # Add predictions and targets to the metrics for non-test stages
                self.metrics.add_batch(batch.id, predictions, targets)

            return loss

    # Instantiate the Brain object for training
    brain = FineTuningBrain(
        modules=modules,
        opt_class=hparams['opt_class'],
        hparams=hparams,
        run_opts=hparams['run_opts'],
        checkpointer=checkpointer
    )

    # Fine-tuning loop
    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=combined_dataloader,
        valid_set=None,  # Include if validation data is available
        test_set=None,  # Include if test data is available
    )

    # Saving the fine-tuned model
    brain.checkpointer.save_checkpoint(name='fine_tuned_model.ckpt')


def save_model(model, hparams, model_path):
    # Save the model checkpoint
    torch.save(model.state_dict(), f"{model_path}/model.ckpt")

    # Save the hyperparameters to a yaml file
    with open(f"{model_path}/hparams.yaml", 'w') as file:
        yaml.dump(hparams, file)


def voice_generator(model_path, text):
    # Load your custom trained Tacotron2 model
    custom_tacotron2 = Tacotron2.from_hparams(source=model_path, savedir="tmpdir_tacotron2")

    # Load your custom trained HiFi-GAN vocoder model
    custom_vocoder = HIFIGAN.from_hparams(source=model_path, savedir="tmpdir_hifigan")

    # Encode the text to obtain mel spectrogram
    mel_output, _ = custom_tacotron2.encode_text(text)

    # Use the vocoder to convert the mel spectrogram to waveform
    waveform = custom_vocoder.decode_batch(mel_output)

    # Play the generated waveform
    sd.play(waveform.squeeze(1).cpu().numpy(), samplerate=custom_vocoder.hparams.sample_rate)
    sd.wait()

def bot_speak(bot_profile, response):
    bot_name = bot_profile['name']
    voicelib = bot_profile['voicelib']
    voiceset = bot_profile['voiceset']
    model_path = os.path.join('bot_profiles', bot_name, 'models')
    if os.path.exists(model_path):
        # Call voice_generator to handle text-to-speech synthesis
        voice_generator(model_path, response)
    else:
        if voicelib == 'pyttsx3':
            pyttsx3_engine = pyttsx3.init()
            # Iterate over all available voices and select the one that matches the voiceset
            voices = pyttsx3_engine.getProperty('voices')
            for voice in voices:
                if voice.id == voiceset:
                    pyttsx3_engine.setProperty('voice', voice.id)
                    break
            pyttsx3_engine.say(response)
            pyttsx3_engine.runAndWait()
        elif voicelib == 'gTTS':
            # Map voiceset to the correct language code for gTTS here
            tts = gtts.gTTS(response, lang='en')  # Replace 'en' with the correct language code
            tts.save('output.mp3')
            pygame.init()
            pygame.mixer.music.load('output.mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        elif voicelib == 'SpeechBrain':
            # Assuming 'voiceset' is the model identifier for SpeechBrain
            tts_model = sb.models.TTS.from_hparams(source=voiceset, savedir="tmpdir_tts")
            mel_output, mel_length, alignment = tts_model.encode_text(response)
            waveforms = tts_model.decode_batch(mel_output)
            # play the audio using sounddevice
            sd.play(waveforms.squeeze(1).cpu().numpy(), samplerate=22050)
            sd.wait()

def tts(bot_profile, response):
    bot_name = bot_profile['name']
    voicelib = bot_profile['voicelib']
    voiceset = bot_profile['voiceset']
    model_path = os.path.join('bot_profiles', bot_name, 'models')

    if os.path.exists(model_path):
        # Call voice_generator to handle text-to-speech synthesis
        audio_path = voice_generator(model_path, response)
    else:
        if voicelib == 'pyttsx3':
            pyttsx3_engine = pyttsx3.init()
            # Iterate over all available voices and select the one that matches the voiceset
            voices = pyttsx3_engine.getProperty('voices')
            for voice in voices:
                if voice.id == voiceset:
                    pyttsx3_engine.setProperty('voice', voice.id)
                    break
            pyttsx3_engine.save_to_file(response, 'output.mp3')
            pyttsx3_engine.runAndWait()
            audio_path = 'output.mp3'
        elif voicelib == 'gTTS':
            # Map voiceset to the correct language code for gTTS here
            tts = gtts.gTTS(response, lang='en')  # Replace 'en' with the correct language code
            audio_path = 'output.mp3'
            tts.save(audio_path)
        elif voicelib == 'SpeechBrain':
            # Assuming 'voiceset' is the model identifier for SpeechBrain
            tts_model = sb.models.TTS.from_hparams(source=voiceset, savedir="tmpdir_tts")
            mel_output, mel_length, alignment = tts_model.encode_text(response)
            waveforms = tts_model.decode_batch(mel_output)
            audio_path = f"output_{bot_name}.wav"
            torchaudio.save(audio_path, waveforms.cpu(), 22050)

    return audio_path

def create_avatar(text):
    instructions = f"Generate an image of a person that matches this description: {text}"
    # Load pre-trained BigGAN model
    biggan_model = torch.hub.load('huggingface/pytorch-pretrained-BigGAN', 'BigGAN-deep-128')

    # Generate image from text using BigGAN
    with torch.no_grad():
        noise_vector = torch.randn(1, 128)
        text_embedding = biggan_model.module.embeddings(torch.LongTensor([biggan_model.module.vocab[instructions]]))
        output = biggan_model.module.decoder([noise_vector, text_embedding], 0.7)

    generated_image = transforms.ToPILImage()(output[0] / 2.0 + 0.5)  # Convert to PIL Image

    # Convert PIL Image to NumPy array for DeepDream
    image_array = np.array(generated_image)

    # Load pre-trained InceptionV3 model for DeepDream
    inception_model = tfhub.load('https://tfhub.dev/google/deepdream/inception_v3/1')

    # Perform DeepDream on the generated image
    dream_img = tf.image.resize(np.expand_dims(image_array, axis=0), (224, 224))
    dream_img = inception_model(dream_img)['mixed3']

    # Save the DeepDream image
    dream_img = tf.squeeze(dream_img)
    dream_img = tf.image.resize(dream_img, (generated_image.size[1], generated_image.size[0]))
    image = Image.fromarray(np.array(dream_img))

    # Initialize and load Pix2Pix model
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                  torch_dtype=torch.float16, safety_checker=None)
    pipe.to("cuda")
    instruction = "Make it a more life-like, high-def, and realistic image like a real photo of a real person"
    # Generate a more life-like, high-def, and realistic image with Pix2Pix
    generated_image = pipe(instruction, image=image, num_inference_steps=10, image_guidance_scale=1).images[0]

    return generated_image


# Function to generate Stable Video Diffusion video
def pic_to_vid(input_image, randomize_seed, motion_bucket_id, fps_id):
    # Stable Video Diffusion pipeline initialization
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to("cuda")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    max_64_bit_int = 2 ** 63 - 1

    # Randomize seed if needed
    if randomize_seed:
        seed = random.randint(0, max_64_bit_int)
    generator = torch.manual_seed(seed)

    # Output folder
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Video generation
    base_count = len(glob(os.path.join(output_folder, "*.mp4")))
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")

    frames = pipe(input_image, decode_chunk_size=3, generator=generator, motion_bucket_id=motion_bucket_id,
                  noise_aug_strength=0.1, num_frames=25).frames[0]

    # Export frames to video
    export_to_video(frames, video_path, fps=fps_id)

    return video_path, seed

# Function to generate Pix2Pix video
def pix2pix_video(input_image, instruction, steps, seed, text_cfg_scale, image_cfg_scale):
    # Pix2Pix pipeline initialization
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                  torch_dtype=torch.float16, safety_checker=None)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    # Resize input image for consistency
    width, height = input_image.size
    factor = 512 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

    # Pix2Pix generation
    if instruction != "":
        generator = torch.manual_seed(seed)
        edited_image = pipe(
            instruction, image=input_image,
            guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale,
            num_inference_steps=steps, generator=generator,
        ).images[0]
        return edited_image

    return [input_image, seed]


# Function to get frames from a video
def get_frames(video_in):
    frames = []
    # Resize the video
    clip = VideoFileClip(video_in)

    # Check fps and resize accordingly
    if clip.fps > 30:
        print("Video rate is over 30, resetting to 30")
        clip_resized = clip.resize(height=512)
        clip_resized.write_videofile("video_resized.mp4", fps=30)
    else:
        print("Video rate is OK")
        clip_resized = clip.resize(height=512)
        clip_resized.write_videofile("video_resized.mp4", fps=clip.fps)

    print("Video resized to 512 height")

    # Opens the Video file with CV2
    cap = cv2.VideoCapture("video_resized.mp4")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video fps: " + str(fps))
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('kang' + str(i) + '.jpg', frame)
        frames.append('kang' + str(i) + '.jpg')
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Broke the video into frames")

    return frames, fps


# Function to create a video from frames
def create_video(frames, fps, audio_path):
    print("Building video result")
    clip = ImageSequenceClip(frames, fps=fps)

    # Load the audio and synchronize it with the video
    audio = AudioFileClip(audio_path)
    clip = clip.set_audio(audio)

    clip.write_videofile("response.mp4", fps=fps)

    return 'response.mp4'


# Function to add subtitles to the video
def add_subtitles(video_path, subtitles, output_path):
    video_clip = VideoFileClip(video_path)
    # Add subtitles to the video
    video_clip = video_clip.set_audio(None).set_duration(video_clip.duration)  # Remove existing audio
    video_clip = video_clip.set_pos("center").set_duration(video_clip.duration)
    video_clip = video_clip.set_subtitles(subtitles)

    # Write the video with subtitles
    video_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Function to infer and generate video
def edit_video(prompt,response, video_in, seed_in, bot_name, audio_path):
    print(prompt)


    break_vid = get_frames(video_in)

    frames_list = break_vid[0]
    fps = break_vid[1]

    # Measure length of the generated audio
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration

    # Measure length of the video
    video_duration = len(frames_list) / fps

    # Calculate trim or pad value
    trim_value = audio_duration - video_duration

    # Pad or trim the video
    if trim_value > 0:
        # Trim the video
        frames_list = frames_list[0:int((video_duration - trim_value) * fps)]
    elif trim_value < 0:
        # Pad the video
        frames_list.extend([frames_list[-1]] * int((-trim_value) * fps))

    print("Set stop frames to: " + str(len(frames_list)))

    # Create video from frames
    input_path = create_video(frames_list, fps, audio_path)



    instructions = (f"{bot_name}: This is an image of {bot_name}. The users said to {bot_name}: {prompt}, {bot_name} responded with {response}. Make the image match the dialogue's context and synchronize {bot_name}'s facial expression and mouth with the subtitles currently displayed in the image.")

    # Add subtitles to the adjusted video
    output_path = "video_response.mp4"

    add_subtitles(input_path, response, output_path)

    # Break down the adjusted video for Pix2Pix processing
    frames_list_adjusted = get_frames(output_path)

    # Process each frame with Pix2Pix
    result_frames = []
    print("Processing frames with Pix2Pix...")

    for i in frames_list_adjusted:
        pil_i = Image.open(i).convert("RGB")

        pix2pix_img = pix2pix_video(pil_i, instructions, 50, seed_in, 7.5, 1.5)

        # Exporting the image
        pix2pix_img.save(f"result_img-{i}.jpg")
        result_frames.append(f"result_img-{i}.jpg")
        print(f"Frame {i} processed.")

    # Reconstruct the final video after Pix2Pix processing
    final_video_path = create_video(result_frames, fps, audio_path)

    print("Finished!")

    return final_video_path


def display_image_box(image_path, screen):
    # Open the image using Pillow
    img = Image.open(image_path)

    # Resize the image to create a box above the video
    img = img.resize((800, 100))

    # Display the image box on the screen
    screen.blit(pygame.surfarray.make_surface(np.array(img)), (0, 0))
    pygame.display.flip()


def play_video(video_path, image_path):
    pygame.init()

    # Set the screen resolution
    screen = pygame.display.set_mode((800, 700), pygame.RESIZABLE)  # Increased height to accommodate the image box

    # Load the video
    pygame.display.set_caption("Video Player")
    video = pygame.movie.Movie(video_path)

    video_screen = pygame.Surface(video.get_size()).convert()

    video.set_display(video_screen)

    video.play()

    clock = pygame.time.Clock()

    # Display the video without the image box initially
    screen.blit(video_screen, (0, 100))
    pygame.display.flip()

    start_time = pygame.time.get_ticks()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        elapsed_time = pygame.time.get_ticks() - start_time

        # Display the image box after 5 seconds and if the video is longer than 10 seconds
        if elapsed_time >= 5000 and video.get_time() >= 10000:
            display_image_box(image_path, screen)

        # Draw video frame on the screen (bottom-left corner)
        screen.blit(video_screen, (0, 100))  # Adjusted y-coordinate to leave space for the image box
        pygame.display.flip()

        clock.tick_busy_loop(60)

    video.stop()
    pygame.quit()


# Function to generate video reponse
def respond(prompt, response, bot_profile, image_path):
    bot_name = bot_profile['name']

    audio_path = tts(bot_profile, response)

    # Find the bot's avatar image in the "Avatars" folder
    avatar_folder = "Avatars"
    bot_pic = os.path.join(avatar_folder, f"{bot_name}.jpg")  # Assuming the avatar image format is JPG

    # Check if the avatar image file exists
    if not os.path.exists(bot_pic):
        print(f"Avatar image for {bot_name} not found in {avatar_folder}. Please make sure the file exists.")
        return

    # Load image
    input_image = Image.open(bot_pic).convert("RGB")

    # Stable Video Diffusion parameters
    stable_randomize_seed = True
    stable_motion_bucket_id = 127
    stable_fps_id = 6

    # Generate Stable Video Diffusion video
    vid, seed_in = pic_to_vid(input_image, stable_randomize_seed, stable_motion_bucket_id, stable_fps_id)

    # Generate Pix2Pix video from Stable Video Diffusion result
    final_vid = edit_video(prompt, response, vid, seed_in, bot_name, audio_path)

    play_video(final_vid, image_path)