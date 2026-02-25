import streamlit as st
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
from gtts import gTTS

st.title("Image to Speech App 🎙️")

uploaded_file = st.file_uploader("Upload an image with English text", type=["jpg", "png", "jpeg"])
language = st.selectbox("Choose Language to Listen", ["English", "Hindi", "Marathi"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button("Extract Text & Read Aloud"):
        with st.spinner("Scanning image..."):
            english_text = pytesseract.image_to_string(image)
            st.write("**Extracted Text:**", english_text)
            
            text_to_speak = english_text
            lang_code = 'en'
            
        with st.spinner("Translating..."):
            if language == "Hindi":
                text_to_speak = GoogleTranslator(source='en', target='hi').translate(english_text)
                lang_code = 'hi'
                st.write("**Translated to Hindi:**", text_to_speak)
                
            elif language == "Marathi":
                text_to_speak = GoogleTranslator(source='en', target='mr').translate(english_text)
                lang_code = 'mr'
                st.write("**Translated to Marathi:**", text_to_speak)
                
        with st.spinner("Generating Audio..."):
            tts = gTTS(text=text_to_speak, lang=lang_code, slow=False)
            tts.save("output.mp3")
            st.audio("output.mp3", format="audio/mp3")
