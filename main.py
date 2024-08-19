import streamlit as st
import openai
import replicate
from pydub import AudioSegment
from pydub.playback import play
import io

# GPT-4o-mini powered generation function
def generate_text(prompt):
    response = openai.Completion.create(
        engine="gpt-4o-mini",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Function to generate music using Replicate's MusicGen model
def generate_music(prompt, replicate_client):
    music_model = replicate_client.models.get("meta/musicgen")
    output = music_model.predict(prompt=prompt)
    return output['audio']

# Function to convert text to speech
def text_to_speech(text, replicate_client):
    speech_model = replicate_client.models.get("coqui-ai/coqui_tts")
    output = speech_model.predict(text=text)
    return output['audio']

# Streamlit app UI
def main():
    st.title("Audio App powered by GPT-4o-mini and Other Technologies")

    # Sidebar for API key inputs
    st.sidebar.header("API Key Setup")
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    replicate_api_key = st.sidebar.text_input("Replicate API Key", type="password")

    if openai_api_key and replicate_api_key:
        openai.api_key = openai_api_key
        replicate_client = replicate.Client(api_token=replicate_api_key)

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
                generated_music = generate_music(prompt, replicate_client)
                st.audio(generated_music, format='audio/mp3')

        elif option == "Text to Speech":
            text = st.text_area("Enter text to convert to speech")
            if st.button("Convert"):
                tts_audio = text_to_speech(text, replicate_client)
                st.audio(tts_audio, format='audio/mp3')

        elif option == "Upload & Process Audio":
            uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
            if uploaded_file is not None:
                audio_data = uploaded_file.read()
                st.audio(audio_data)
                # Process the audio (e.g., transcribe, analyze)
                st.write("Processing functionality to be added.")
    else:
        st.sidebar.warning("Please enter your OpenAI and Replicate API keys to proceed.")

if __name__ == "__main__":
    main()
