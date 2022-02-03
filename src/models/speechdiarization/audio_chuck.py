

def generate_chunk(audio, start, end, chunk_loc, idx):
    start_chunk = start *1000
    end_chunk = end *1000
    audio_chunk=audio[start_chunk:end_chunk]
    audio_chunk.export( f"{chunk_loc}/chunk_{idx}.wav", format="wav")