import streamlit as st
import replicate
import requests
import tempfile
import os

# Function to generate music using Replicate's meta/musicgen model
def generate_music(api_key, prompt, model_version, output_format, normalization_strategy):
    input = {
        "prompt": prompt,
        "model_version": model_version,
        "output_format": output_format,
        "normalization_strategy": normalization_strategy,
    }
    
    output = replicate.run(
        "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
        input=input,
        api_token=api_key
    )
    
    return output

# Function to apply effects (dummy implementation)
def apply_effects(music_path, effects):
    # For demonstration purposes, this function just returns the original music path.
    # You would need to implement actual audio processing here.
    return music_path

# Function to save and download the file
def save_and_download_file(file_path):
    with open(file_path, 'rb') as f:
        st.download_button(
            label="Download Music",
            data=f,
            file_name=os.path.basename(file_path),
            mime="audio/mpeg"
        )

# Streamlit UI
st.title("AI Music Generator with Effects")

# Input for Replicate API key
replicate_api_key = st.text_input("Enter your Replicate API key:", type="password")

# User input
text_prompt = st.text_area("Enter your music prompt:")
model_version = st.selectbox("Select model version:", ["small", "medium", "large", "stereo-large"])
output_format = st.selectbox("Select output format:", ["mp3", "wav"])
normalization_strategy = st.selectbox("Select normalization strategy:", ["peak", "rms"])

if st.button("Generate Music"):
    if not replicate_api_key:
        st.error("Please enter your Replicate API key.")
    elif text_prompt:
        st.write("Generating music...")
        with st.spinner("Please wait..."):
            # Generate music
            music_url = generate_music(
                replicate_api_key,
                text_prompt,
                model_version,
                output_format,
                normalization_strategy
            )
            if music_url:
                st.write("Music generated successfully!")
                
                # Download the generated music
                music_path = tempfile.mktemp(suffix=f".{output_format}")
                response = requests.get(music_url)
                with open(music_path, 'wb') as f:
                    f.write(response.content)
                
                st.audio(music_path, format=f'audio/{output_format}')

                # Apply effects
                st.write("Apply effects:")
                effects = []
                if st.checkbox("Reverb"):
                    effects.append("reverb")
                if st.checkbox("Echo"):
                    effects.append("echo")
                if st.checkbox("Distortion"):
                    effects.append("distortion")

                if effects:
                    st.write("Applying effects...")
                    with st.spinner("Please wait..."):
                        # Apply effects (dummy implementation)
                        processed_music_path = apply_effects(music_path, effects)
                        st.write("Effects applied successfully!")
                        st.audio(processed_music_path, format=f'audio/{output_format}')

                        # Export post-production result
                        save_and_download_file(processed_music_path)
            else:
                st.error("Failed to generate music. Please try again.")
    else:
        st.error("Please enter a music prompt.")

if __name__ == "__main__":
    st.run()
