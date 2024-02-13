# Author:Chance Brownfield
# Email:ChanceBrownfield@protonmail.com
import time
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
import os
from ADAM.Temporal import get_user
recognizer = sr.Recognizer()

class Audio_Recognition:
    """
        Basic Speech Recognition:
          recognizer = Audio_Recognition()
          audio_data = recognizer.capture_audio()
          transcribed_text = recognizer.transcribe_audio(audio_data)
          print(transcribed_text)
        Speaker Diarization:
          recognizer = Audio_Recognition()
          audio_data = recognizer.capture_audio(source="audio_file.wav")
          diarization_result = recognizer.speaker_diarization(audio_data)
        Segment Audio based on Diarization:
          segmented_audios = recognizer.segment_audio(audio_data, diarization_result)
        Identify Speakers in Segmented Audio:
          user_profiles = recognizer.identify_speakers(segmented_audios)
        Transcribe Multiple Audio Segments:
          dialogue = recognizer.transcribe_multi_audio(segmented_audios)
        Perform ASR (Automatic Speech Recognition):
          dialogue = recognizer.ASR(source="audio_file.wav")
        Perform ASR with Speaker Recognition:
          dialogue, user_profiles = recognizer.AMSSR(source="audio_file.wav")
        Perform ASR with Hotword Detection:
          dialogue = recognizer.ASR(source=None, hotword="snowboy_config.json")
        Perform ASR with Timeout:
          dialogue = recognizer.ASR(source=None, timeout=5)
        Perform ASR with a Limit on the Number of Iterations:
          dialogue = recognizer.ASR(source=None, stop=5)
        Perform ASR with a Stop Time:
          dialogue = recognizer.ASR(source=None, stoptime=60)
        Perform ASR with Both Text and Audio Output:
          result_text, result_audio = recognizer.ASR(source="audio_file.wav", output="b.
    """
    def __init__(self):
        self.segmented_audio_dir = "segmented_audio"
        pass

    def capture_audio(self, source=None, timeout=None, hotword=None):
        """
        Captures audio data using SpeechRecognition.
        """
        recognizer = sr.Recognizer()
        if source:
            with sr.AudioFile(source) as audio_file:
                audio_data = recognizer.record(audio_file)
        else:
            if hotword:
                with sr.Microphone() as microphone:
                    if timeout:
                        audio_data = recognizer.listen(microphone, timeout=timeout)
                    else:
                        audio_data = recognizer.listen(microphone)
            else:
                with sr.Microphone() as microphone:
                    if timeout:
                        audio_data = recognizer.listen(microphone, timeout=timeout)
                    else:
                        audio_data = recognizer.listen(microphone)
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
        """
           Segment Audio Based on Diarization.
        """
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
        """
           Segmented Audio Speaker Identification.
        """
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

    def transcribe_audio(self, audio_data):
        """
        Common method for transcribing audio.
        """
        model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        audio_np = np.frombuffer(audio_data.frame_data, dtype=np.int16)
        segments, _ = model.transcribe(audio_np, vad_filter=True)
        dialogue = ""
        for segment in segments:
            dialogue += segment.text + " "
        return dialogue.strip()

    def transcribe_audio_file(self, audio_path):
        """
        Transcribes audio file.
        """
        audio_data = self.capture_audio(audio_path)
        dialogue = self.transcribe_audio(audio_data)

        return dialogue

    def transcribe_multi_audio(self, segmented_audios):
        """
        Transcribes multiple audio files.
        """
        dialogue = ""
        for audio_path in segmented_audios:
            audio_data = self.capture_audio(audio_path)
            dialogue += self.transcribe_audio_file(audio_data)
        return dialogue.strip()

    def ASR(self, source=None, output="both", timeout=None, stop=0, stoptime=None, hotword=None):
        """
        Performs ASR (Automatic Speech Recognition) on audio data.
        """
        dialogue = ""
        audio_data = None
        start_time = time.time()
        last_segment_texts = []
        iteration = 0
        if hotword:
            recognizer.energy_threshold = 4000
            recognizer.pause_threshold = 0.8
            recognizer.dynamic_energy_threshold = True
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
        while not dialogue or iteration < stop or (stoptime and time.time() - start_time < stoptime):
            audio_data = self.capture_audio(source, timeout, hotword)
            dialogue += self.transcribe_audio(audio_data)
            last_segment_texts.append(dialogue.strip())
            if len(last_segment_texts) > stop:
                last_segment_texts.pop(0)
            if dialogue.strip() and last_segment_texts[-1].strip() == "":
                iteration += 1
        if "text" in output:
            result = dialogue.strip()
        elif "audio" in output:
            result = audio_data
        else:
            result = (dialogue.strip(), audio_data)
        return result

    def AMSSR(self, source=None, timeout=None, stop=0, stoptime=None, hotword=None):
        """
        Performs ASR with speaker recognition.
        """
        dialogue = ""
        start_time = time.time()
        iteration = 0
        recognizer = sr.Recognizer()
        if hotword:
            recognizer.energy_threshold = 4000
            recognizer.pause_threshold = 0.8
            recognizer.dynamic_energy_threshold = True
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
        user_profiles = []
        while not dialogue or iteration < stop or (stoptime and time.time() - start_time < stoptime):
            audio_data = self.capture_audio(source, timeout, hotword)

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
