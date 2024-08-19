import streamlit as st
import openai
import replicate
from pydub import AudioSegment
from pydub.playback import play
import io

# Initialize OpenAI and Replicate API keys
openai.api_key = "your-openai-api-key"
replicate_api_key = "your-replicate-api-key"
replicate_client = replicate.Client(api_token=replicate_api_key)

# GPT-4o-mini powered generation function
def generate_text(prompt):
    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to generate music using Replicate's MusicGen model
def generate_music(prompt):
    music_model = replicate_client.models.get("meta/musicgen")
    output = music_model.predict(prompt=prompt)
    return output['audio']

# Function to convert text to speech
def text_to_speech(text):
    speech_model = replicate_client.models.get("coqui-ai/coqui_tts")
    output = speech_model.predict(text=text)
    return output['audio']

# Function to play audio
def play_audio(audio_data):
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
    play(audio_segment)

# Streamlit app UI
def main():
    st.title("Audio App powered by GPT-4o-mini and Other Technologies")
    
    st.sidebar.header("Choose an option")
    option = st.sidebar.selectbox("What would you like to do?", 
                                  ["Generate Text", "Generate Music", "Text to Speech", "Upload & Process Audio"])

    if option == "Generate Text":
        prompt = st.text_input("Enter your prompt for GPT-4o-mini")
        if st.button("Generate"):
            generated_text = generate_text(prompt)
            st.write("Generated Text:")
            st.write(generated_text)

    elif option == "Generate Music":
        prompt = st.text_input("Enter a music prompt")
        if st.button("Generate"):
            generated_music = generate_music(prompt)
            st.audio(generated_music, format='audio/mp3')

    elif option == "Text to Speech":
        text = st.text_area("Enter text to convert to speech")
        if st.button("Convert"):
            tts_audio = text_to_speech(text)
            st.audio(tts_audio, format='audio/mp3')

    elif option == "Upload & Process Audio":
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
        if uploaded_file is not None:
            audio_data = uploaded_file.read()
            st.audio(audio_data)
            # Process the audio (e.g., transcribe, analyze)
            st.write("Processing functionality to be added.")

if __name__ == "__main__":
    main()
