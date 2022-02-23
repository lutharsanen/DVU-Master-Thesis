from speechbrain.pretrained.interfaces import foreign_class
from speechbrain.pretrained import EncoderDecoderASR

def get_speech_emotion(audio_file):
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
        pymodule_file="custom_interface.py", 
        classname="CustomEncoderWav2vec2Classifier")
    _, score, _, text_lab = classifier.classify_file(audio_file)
    return text_lab, score


def get_speech_to_text(audio_file):
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir="./pretrained_ASR")
    text = asr_model.transcribe_file(audio_file)
    return text