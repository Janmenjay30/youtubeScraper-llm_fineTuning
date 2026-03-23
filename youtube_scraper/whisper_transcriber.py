import yt_dlp
import whisper

model = whisper.load_model("base")

def download_audio(video_id):

    url=f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts={
        "format": "bestaudio/best",
        "outtmpl": f"{video_id}.%(ext)s",
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return f"audio/{video_id}.webm"

def transcribe_audio(audio_path):

    
    result=model.transcribe(audio_path)

    return result["text"]

