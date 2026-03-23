import yt_dlp
import pandas as pd
import os
from youtube_transcript_api import YouTubeTranscriptApi
from cleaner import clean_text, split_sentences

CHANNEL_URL = "https://www.youtube.com/@mkbhd/videos"
MAX_VIDEOS = 5

ytt_api = YouTubeTranscriptApi()


def get_video_ids(channel_url, max_videos=20):

    ydl_opts = {
        "quiet": True,
        "extract_flat": "in_playlist"
    }

    video_ids = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)

        entries = info.get("entries", [])

        for entry in entries:
            if entry and entry.get("id"):
                video_ids.append(entry["id"])

            if len(video_ids) >= max_videos:
                break

    return video_ids


def get_transcript(video_id):

    try:
        transcript = ytt_api.fetch(video_id)

        text = " ".join([snippet.text for snippet in transcript])

        return text

    except Exception as e:
        print(f"Transcript not available for {video_id}: {e}")
        return None


def scrape_channel(channel_url, max_videos=20):

    print("Fetching video list...")

    video_ids = get_video_ids(channel_url, max_videos)

    print(f"Found {len(video_ids)} videos")

    raw_data = []
    cleaned_sentences = []

    os.makedirs("dataset", exist_ok=True)

    for vid in video_ids:

        print("Processing video:", vid)

        transcript = get_transcript(vid)

        if transcript:

            raw_data.append({
                "video_id": vid,
                "transcript": transcript
            })

            cleaned = clean_text(transcript)

            sentences = split_sentences(cleaned)

            cleaned_sentences.extend(sentences)

    raw_df = pd.DataFrame(raw_data)
    raw_df.to_csv("dataset/raw_transcripts.csv", index=False)

    with open("dataset/cleaned_dataset.txt", "w", encoding="utf-8") as f:
        for sentence in cleaned_sentences:
            print(sentence)
            f.write(sentence + "\n")

    print("\nScraping Complete")
    print("Raw dataset saved -> dataset/raw_transcripts.csv")
    print("Cleaned dataset saved -> dataset/cleaned_dataset.txt")


if __name__ == "__main__":
    scrape_channel(CHANNEL_URL, MAX_VIDEOS)