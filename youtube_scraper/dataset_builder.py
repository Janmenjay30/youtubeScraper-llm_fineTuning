import pandas as pd
from scrapper import get_video_ids
from transcript_fetcher import get_transcript
from whisper_transcriber import download_audio, transcribe_audio
from cleaner import clean_text, split_sentences
import os

CHANNEL_URL = "https://www.youtube.com/@mkbhd/videos"
MAX_VIDEOS = 5

os.makedirs("dataset", exist_ok=True)
os.makedirs("audio", exist_ok=True)


def build_dataset(CHANNEL_URL,MAX_VIDEOS):

    video_ids=get_video_ids(CHANNEL_URL,MAX_VIDEOS)

    raw_data=[]
    sentences_all=[]

    for vid in video_ids:

        print(f"Processing video: {vid}")

        transcript=get_transcript(vid)

        if transcript is None:


            print(f"Failed to get transcript for video: {vid}")

            print("Attempting audio transcription...")

            audio_path=download_audio(vid)

            transcript=transcribe_audio(audio_path)

        raw_data.append({
            "video_id": vid,
            "transcript": transcript
        })

        cleaned=clean_text(transcript)

        sentences=split_sentences(cleaned)

        sentences_all.extend(sentences)

    df=pd.DataFrame(raw_data)

    df.to_csv("dataset/raw_dataset.csv", index=False)

    with open("dataset/cleaned_dataset.txt", "w", encoding="utf-8") as f:
        for sentence in sentences_all:
            print(sentence)
            f.write(sentence + "\n")
        
        print("\nDataset Building Complete")

if __name__ == "__main__":
    build_dataset(CHANNEL_URL,MAX_VIDEOS)

            

