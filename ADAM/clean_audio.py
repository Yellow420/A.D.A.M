#Author: Chance Brownfield
#Email: ChanceBrownfield@protonmail.com
import webrtcvad
import soundfile as sf
import noisereduce as nr
import torch
import resampy
from pydub import AudioSegment
import audioop
import librosa
from scipy.signal import resample
import numpy as np
import speechbrain as sb
from speechbrain.pretrained import SpectralMaskEnhancement, SepformerSeparation
# Function to apply VAD
def apply_vad(audio, sr, aggressiveness=3):
    save_and_play(audio, sr, 'before_vad.wav')  # Before any processing
    print("vad1")
    print(f'Audio data type: {audio.dtype}')
    print(f'Audio data range: {np.min(audio)} to {np.max(audio)}')
    vad = webrtcvad.Vad(aggressiveness)
    # Ensure audio is 16-bit PCM
    print("vad2")
    raw_samples = (audio * 32768).astype(np.int16)
    samples_per_window = (sr * 10) // 1000
    segments = []
    print("vad3")
    for start in np.arange(0, len(raw_samples), samples_per_window):
        stop = min(start + samples_per_window, len(raw_samples))
        is_speech = vad.is_speech(raw_samples[start:stop].tobytes(),
                                  sample_rate=sr,
                                  length=stop - start)
        segments.append(dict(
            start=start,
            stop=stop,
            is_speech=is_speech))
    return segments


def prepare_for_vad(audio, audio_sample_rate, target_sr=16000):
    save_and_play(audio, audio_sample_rate, 'before_prep.wav')  # Before any processing

    print("prep1")
    # Normalize audio to float (-1 to 1) if necessary
    print(f'Audio values range before normalization: {np.min(audio)} to {np.max(audio)}')
    if np.max(np.abs(audio)) <= 1:
        print("Audio is already normalized")
    else:
        audio = audio.astype(np.float32) / 32768
    print(f'Audio values range after normalization: {np.min(audio)} to {np.max(audio)}')
    print("prep2")
    # Resample if necessary
    # Before resampling

    print("prep3")
    if audio_sample_rate != target_sr:
        audio = resampy.resample(audio.astype(float), audio_sample_rate, target_sr)
        # After resampling

    print("prep4")
    save_and_play(audio, target_sr, 'mid_prep.wav')  # After all processing
    # Convert to 16-bit PCM
    audio = (audio * 32768).astype(np.int16)
    print("prep5")
    # Pad audio to 10 minutes if shorter
    print("prep6")
    desired_length = target_sr * 60 * 10  # 10 minutes at target sample rate
    if len(audio) < desired_length:
        audio = np.pad(audio, (0, desired_length - len(audio)), mode='constant')
    print("prep7")
    save_and_play(audio, target_sr, 'after_prep.wav')  # After all processing
    return audio, target_sr

def reduce_noise(audio, sr, noise_sample_length=10000, prop_decrease=1.0, ):
    # Ensure noise_sample_length is within the length of audio
    noise_sample_length = min(noise_sample_length, len(audio))
    print("reduce1")
    # If audio is too short for noise_sample_length, increase audio length by padding with zeros
    if len(audio) < noise_sample_length:
        padding = np.zeros(noise_sample_length - len(audio))
        audio = np.concatenate((audio, padding))

    print("reduce2")
    # Perform noise reduction
    try:
        audio_denoised = nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)
    except Exception as e:
        print(f"Error during noise reduction: {e}")
        audio_denoised = audio  # if noise reduction fails, return the original audio
    print("reduce3")
    return audio_denoised


def enhance_and_separate_speech(audio_filename, sample_rate):
    print("enhance 1")
    # Load the pretrained models for speech enhancement and separation
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank", savedir="pretrained_models/metricgan-plus-voicebank"
    )
    separate_model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-whamr", savedir="pretrained_models/sepformer-whamr"
    )

    # Enhance a single audio file
    enhanced = enhance_model.enhance_file(audio_filename)

    print("enhance 2")
    print("Enhanced Audio Shape:", enhanced.shape)

    # Apply speech separation to separate speech sources
    separated = separate_model.separate_file(audio_filename)

    # Select the first speech source as the target speaker
    cleaned_audio = separated[0]
    sample_rate = 8000
    # Save the cleaned audio (you can adjust the filename and format as needed)
    enhanced_audio = 'enhanced_audio.wav'
    save_and_play(cleaned_audio.numpy(), sample_rate,  enhanced_audio)
    print("enhance 3")
    print("Cleaned Audio Shape:", cleaned_audio.shape)
    return enhanced_audio, 8000

def post_enhance(audio_filename, sample_rate, target_sr=16000):
    # Load the audio file
    audio, _ = sf.read(audio_filename)

    # Convert to pydub AudioSegment
    audio_segment = AudioSegment(
        audio.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio.dtype.itemsize,
        channels=1  # Assume mono for simplicity
    )

    # Resample using pydub
    try:
        audio_resampled = audio_segment.set_frame_rate(target_sr)
    except audioop.error:
        # Handle the case where sample width is not supported, by converting to 16-bit PCM first
        audio_16bit = (audio * 32768).astype(np.int16)
        audio_segment = AudioSegment(
            audio_16bit.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio_16bit.dtype.itemsize,
            channels=1
        )
        audio_resampled = audio_segment.set_frame_rate(target_sr)

    # Convert to numpy array
    audio_resampled_np = np.array(audio_resampled.get_array_of_samples())

    # Ensure the audio has two channels
    if audio_resampled_np.ndim == 1:
        audio_resampled_np = np.column_stack((audio_resampled_np, audio_resampled_np))
    elif audio_resampled_np.shape[1] != 2:
        audio_resampled_np = np.column_stack((audio_resampled_np[:, 0], audio_resampled_np[:, 0]))

    # Convert to 16-bit PCM
    audio_resampled_np = (audio_resampled_np * 32768).astype(np.int16)

    # Save the resampled audio
    resampled_filename = 'clean_audio.wav'
    save_and_play(audio_resampled_np, target_sr, resampled_filename)

    return resampled_filename
def save_and_play(audio, sr, filename='temp.wav'):
    # Save to file
    sf.write(filename, audio, sr)


def clean_audio(audio_filename):
    # Load the audio file
    audio, sample_rate = sf.read(audio_filename)
    print("Input Audio Shape:", audio.shape)

    # Apply noise reduction
    audio_denoised = reduce_noise(audio, sample_rate, noise_sample_length=10000, prop_decrease=1.0)

    # Save the processed audio back to audio_filename
    sf.write(audio_filename, audio_denoised, sample_rate)
    print("Input Audio Shape:", audio.shape)
    # Prepare audio for VAD (this will handle the conversion to 16-bit PCM and resampling)
    audio_denoised_resampled, new_sample_rate = prepare_for_vad(audio_denoised, sample_rate)

    segments = apply_vad(audio_denoised_resampled, new_sample_rate)

    # Concatenate segments flagged as speech
    speech_audio = np.concatenate(
        [audio_denoised_resampled[seg['start']:seg['stop']] for seg in segments if seg['is_speech']]
    )
    save_and_play(speech_audio, new_sample_rate, 'post_proc.wav')  # save the concatenated speech segments

    # Check if speech_audio is empty (i.e., no segments containing speech were found)
    if len(speech_audio) == 0:
        print("No segments containing speech were found.")
        return None  # Return None if no speech segments were found
    print("Input Audio Shape:", audio.shape)

    # Additional step: Perform final speech enhancement and separation using SpeechBrain
    #enhanced_audio, enhanced_sample_rate = enhance_and_separate_speech('post_proc.wav', new_sample_rate)
    #cleaned_audio = post_enhance(enhanced_audio, enhanced_sample_rate)
    return 'post_proc.wav'  # Return the path to the processed audio file
