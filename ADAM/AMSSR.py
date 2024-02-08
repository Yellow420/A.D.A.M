# Author: Chance Brownfield
# Email: ChanceBrownfield@protonmail.com
import pyaudio
import os
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as stt
from ADAM.Temporal import get_user
from ADAM.clean_audio import clean_audio
from faster_whisper import WhisperModel

class Audio_Recognition:
    def __init__(self):
        self.segmented_audio_dir = "segmented_audio"
        self.whisper_model = WhisperModel("large-v3")

    def record_audio(self, source=None, timeout=None, text=None, earlystop=None, CHUNK=1024, FORMAT=pyaudio.paInt16, CHANNELS=1, RATE=44100):
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

        print("Listening...")

        audio_data = b""  # Initialize as byte string
        speech_detected = False
        silence_count = 0

        try:
            while True:
                audio_chunk = stream.read(CHUNK)
                audio_data += audio_chunk  # Concatenate chunks into one audio stream

                # Check if audio chunk contains speech
                if self.whisper_model.is_speech(audio_chunk):
                    speech_detected = True
                    silence_count = 0  # Reset silence count

                # If speech was detected but now it's silent, increment silence count
                if speech_detected and not self.whisper_model.is_speech(audio_chunk):
                    silence_count += 1

                # Check if timeout is reached and stop recording if timeout is specified
                if timeout and silence_count * CHUNK / RATE >= timeout:
                    print("Timeout reached. Recording stopped.")
                    return self.handle_record_result(text, audio_data)

                # Check if early stop is requested and stop recording if time limit is reached
                if earlystop and (earlystop * RATE / CHUNK <= silence_count):
                    print("Early stop. Recording stopped.")
                    return self.handle_record_result(text, audio_data)

        except KeyboardInterrupt:
            print("Recording stopped.")

        print("Finished recording.")

        stream.stop_stream()
        stream.close()
        p.terminate()

        return self.handle_record_result(text, audio_data)

    def handle_record_result(self, text, audio_data):
        if text is None:
            return self.whisper_model.transcribe(audio_data), audio_data
        elif text == "true":
            return self.whisper_model.transcribe(audio_data)
        elif text == "false":
            return audio_data
        else:
            return None

    def listen_and_record(self, source=None, timeout=None, text=None, earlystop=None):
        return self.record_audio(source, timeout, text=text, earlystop=earlystop)

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

    def transcribe_audio_common(self, audio_path, is_multi=False):
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
        transcribed_text = result["text"]

        # Save the transcribed text to a file
        transcript_file_path = os.path.join(
            self.segmented_audio_dir,
            f"{os.path.basename(audio_path)}_transcript.txt")
        with open(transcript_file_path, 'w') as transcript_file:
            transcript_file.write(transcribed_text)

        # If it's a single audio file, return the transcribed text directly
        if not is_multi:
            return transcribed_text

        # If it's for multiple audio files, return a tuple with the speaker name and transcribed text
        speaker = os.path.splitext(os.path.basename(audio_path))[0].split('_')[0]
        return speaker, transcribed_text

    def transcribe_audio(self, audio_path):
        return self.transcribe_audio_common(audio_path)

    def transcribe_multi_audio(self, segmented_audios):
        segmented_audios.sort(key=lambda x: float(os.path.splitext(os.path.basename(x))[0].split('_')[1]))

        dialogue = ""

        for audio_path in segmented_audios:
            # Call the common transcribe method for each audio file
            speaker, transcribed_text = self.transcribe_audio_common(audio_path, is_multi=True)

            # Add the speaker and transcribed text to the dialogue
            dialogue += f"{speaker} said: \"{transcribed_text}\"\n"

        return dialogue

    def ASR(self, source=None, timeout=None, earlystop=None):
        # Step 1: Record audio
        text = "true"
        dialogue = self.listen_and_record(source, timeout, text, earlystop)

        return dialogue

    def AMSSR(self, source=None, timeout=None, earlystop=None):
        # Step 1: Record audio
        text = "false"
        audio_data = clean_audio(self.listen_and_record(source, timeout, text, earlystop))
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

