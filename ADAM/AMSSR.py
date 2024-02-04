# Author: Chance Brownfield
# Email: ChanceBrownfield@protonmail.com
import pyaudio
import os
import torch
import speech_recognition as sr
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as stt
from ADAM.Temporal import get_user
from ADAM.clean_audio import clean_audio

class Audio_Recognition:
    def __init__(self):
        self.segmented_audio_dir = "segmented_audio"

    def listen_and_record(self, source=None, timeout=10):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=source) if source is not None else p.open(format=FORMAT,
                                                                                     channels=CHANNELS,
                                                                                     rate=RATE,
                                                                                     input=True,
                                                                                     frames_per_buffer=CHUNK)

        recognizer = sr.Recognizer()

        print("Listening...")

        audio_data = None

        try:
            if source is not None:
                print("Listening...1")
                with sr.Microphone(device_index=source) as src:
                    audio_data = recognizer.listen(src, timeout=timeout)
            else:
                print("Listening...2")
                with sr.Microphone() as src:
                    audio_data = recognizer.listen(src, timeout=timeout)
        except sr.WaitTimeoutError:
            print("Timeout reached. Recording stopped.")

        print("Finished recording.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        return audio_data


    def speaker_diarization(self, audio_data):
        # load the pipeline from Huggingface Hub
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2022.07")

        # apply the pipeline to the audio file
        diarization = pipeline(audio_data)

        # dump the diarization output to disk using RTTM format
        with open("audio.rttm", "w") as rttm:
            diarization.write_rttm(rttm)

        # return the diarization result
        return diarization

    def segment_audio(self, audio_data, diarization_result):
        # Create the output directory if it doesn't exist
        os.makedirs(self.segmented_audio_dir, exist_ok=True)

        # Clear the contents of the output directory
        for file in os.listdir(self.segmented_audio_dir):
            file_path = os.path.join(self.segmented_audio_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)

        # Convert audio_data to AudioSegment
        audio = AudioSegment.from_raw(audio_data, sample_width=2, frame_rate=44100, channels=1)

        # Read the RTTM file
        with open(diarization_result, 'r') as rttm_file:
            lines = rttm_file.readlines()

        # Initialize a list to store segmented audio files
        segmented_audios = []

        # Process each line in the RTTM file
        for line in lines:
            fields = line.strip().split()
            start_time = float(fields[3])
            duration = float(fields[4])
            speaker_id = fields[7]

            # Calculate start and end positions in milliseconds
            start_pos = int(start_time * 1000)
            end_pos = start_pos + int(duration * 1000)

            # Extract the segment from the original audio
            segment = audio[start_pos:end_pos]

            # Save the segmented audio to a new file
            output_path = os.path.join(self.segmented_audio_dir, f"{speaker_id}_{start_time}.wav")
            segment.export(output_path, format="wav")

            # Add the path to the list
            segmented_audios.append(output_path)

        return segmented_audios

    def identify_speakers(self, segmented_audios):
        user_profiles = []

        for audio_path in segmented_audios:
            # Extract speaker_id and start_time from the file name
            file_name = os.path.basename(audio_path)
            speaker_id, start_time = os.path.splitext(file_name)[0].split('_')

            user_id, user_profile = get_user(audio_path)

            # If the user is not recognized, delete the audio segment and skip to the next iteration
            if user_id == "unknown":
                os.remove(audio_path)
                continue

            # Update the file name with the speaker's name
            new_file_name = f"{user_profile['name']}_{start_time}.wav"
            new_audio_path = os.path.join(os.path.dirname(audio_path), new_file_name)

            # Rename the file
            os.rename(audio_path, new_audio_path)

            # Append the identified user profile to the list
            user_profiles.append(user_profile)

        return user_profiles

    def transcribe_audio(self, audio_path):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = stt(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={"language": "english"}  # Specify the target language as English
        )

        # Transcribe the audio using Whisper
        result = pipe(audio_path)

        # Get the transcribed text
        dialogue = result["text"]

        # Save the transcribed text to a file
        transcript_file_path = os.path.join(
            self.segmented_audio_dir,
            f"{os.path.basename(audio_path)}_transcript.txt")
        with open(transcript_file_path, 'w') as transcript_file:
            transcript_file.write(dialogue)

        # Return the transcribed text
        return dialogue

    def transcribe_multi_audio(self, segmented_audios):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        pipe = stt(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
            generate_kwargs={"language": "english"}  # Specify the target language as English
        )

        # Sort segmented audios based on timestamp in the file name
        segmented_audios.sort(key=lambda x: float(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

        dialogue = ""

        for audio_path in segmented_audios:
            # Extract speaker name and timestamp from the file name
            file_name = os.path.basename(audio_path)
            speaker, timestamp = os.path.splitext(file_name)[0].split('_')

            # Transcribe the audio using Whisper
            result = pipe(audio_path)

            # Get the transcribed text
            transcribed_text = result["text"]

            # Save the transcribed text to a file
            transcript_file_path = os.path.join(self.segmented_audio_dir,
                                                f"{file_name}transcript.txt")
            with open(transcript_file_path, 'w') as transcript_file:
                transcript_file.write(transcribed_text)

            # Add the transcribed text to the dialogue
            dialogue += f"{speaker} said: \"{transcribed_text}\"\n"

            # Return the dialogue string
        return dialogue

    def ASR(self, source=None, timeout=None):
        # Step 1: Record audio
        audio_data = self.listen_and_record(source, timeout)
        cleaned_audio = clean_audio(audio_data)
        # Step 2: Perform speaker diarization
        diarization_result = self.speaker_diarization(cleaned_audio)

        # Check if diarization found no speakers/speech
        if not diarization_result:
            return "", []

        # Step 5: Transcribe each audio segment
        dialogue = self.transcribe_audio(cleaned_audio)

        return dialogue

    def AMSSR(self, source=None, timeout=None):
        # Step 1: Record audio
        audio_data = clean_audio(self.listen_and_record(source, timeout))
        # Step 2: Perform speaker diarization
        diarization_result = self.speaker_diarization(audio_data)

        # Check if diarization found no speakers/speech
        if not diarization_result:
            return "", []

        # Step 3: Segment the audio based on diarization
        segmented_audios = self.segment_audio(audio_data, diarization_result)

        # Step 4: Identify speakers in each segment
        user_profiles = self.identify_speakers(segmented_audios)

        # Step 5: Transcribe each audio segment
        dialogue = self.transcribe_multi_audio(segmented_audios)

        return dialogue, user_profiles

