from dotenv import load_dotenv
from elevenlabs import set_api_key, generate, play, voices
import os

#os.environ["ELEVENLABS_PLAY_CMD"] = "Users/athar/Documents/ffmpeg-2025-02-17-git-b92577405b-essentials_build/bin/ffplay.exe"


load_dotenv()
api_key=os.getenv("ELEVENLABS_API_KEY")
set_api_key(os.getenv("ELEVENLABS_API_KEY"))

my_voices = voices()
for v in my_voices:
    print(f"Name: {v.name}, ID: {v.voice_id}")

audio = generate(
    text="Hello world",
    voice="Bill",  
)

play(audio)