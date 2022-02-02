import settings
import audio_preprocessing.extract_audio as extractor
import settings as s

# extract audio from video
extractor.run_extractor()
hlvu_location = s.HLVU_LOCATION
audio_path = f"{hlvu_location}/audio"