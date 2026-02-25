import streamlit as st
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
from gtts import gTTS
import io

# Summarization libraries
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# Download necessary language data for summarizer (cached so it only happens once)
@st.cache_resource
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

setup_nltk()

st.title("Image to Speech App 🎙️")

# --- STATE MANAGEMENT ---
# We use st.session_state to temporarily save the extracted text 
# so the user can edit it without it disappearing.
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

uploaded_file = st.file_uploader("Upload an image with English text", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # ==========================================
    # STEP 1: EXTRACT TEXT
    # ==========================================
    if st.button("1. Extract Text from Image"):
        with st.spinner("Scanning image..."):
            # Extract text and save it to session state
            st.session_state.extracted_text = pytesseract.image_to_string(image)

    # Only show the rest of the app if text has been extracted
    if st.session_state.extracted_text:
        
        st.markdown("### Review & Edit")
        
        # FEATURE 1: Editable Text Area
        edited_text = st.text_area(
            "Fix any typos here before generating audio:", 
            value=st.session_state.extracted_text, 
            height=200
        )
        
        # FEATURE 2: Word Count
        word_count = len(edited_text.split())
        st.caption(f"**Word count:** {word_count} words")
        
        # FEATURE 3: Summarizer (Only shows if words > 120)
        if word_count > 120:
            st.info("💡 The text is quite long. You can generate a summary to save time.")
            if st.button("Summarize (Keep Important Info)"):
                with st.spinner("Summarizing..."):
                    parser = PlaintextParser.from_string(edited_text, Tokenizer("english"))
                    summarizer = LsaSummarizer()
                    
                    # Extract top 3-4 most important sentences (usually equates to 50-60 words)
                    summary_sentences = summarizer(parser.document, sentences_count=3)
                    short_text = " ".join([str(sentence) for sentence in summary_sentences])
                    
                    # Update the box with the new shortened text and refresh screen
                    st.session_state.extracted_text = short_text
                    st.rerun()

        st.divider()
        
        # ==========================================
        # STEP 2: TRANSLATE AND READ
        # ==========================================
        st.markdown("### Generate Audio")
        language = st.selectbox("Choose Language to Listen", ["English", "Hindi", "Marathi"])
        
        if st.button("2. Translate & Read Aloud"):
            # Use the EDITED text, not the original raw text
            text_to_speak = edited_text 
            lang_code = 'en'
            
            with st.spinner("Translating..."):
                if language == "Hindi":
                    text_to_speak = GoogleTranslator(source='en', target='hi').translate(edited_text)
                    lang_code = 'hi'
                    st.write("**Translated to Hindi:**", text_to_speak)
                    
                elif language == "Marathi":
                    text_to_speak = GoogleTranslator(source='en', target='mr').translate(edited_text)
                    lang_code = 'mr'
                    st.write("**Translated to Marathi:**", text_to_speak)
                    
            with st.spinner("Generating Audio..."):
                tts = gTTS(text=text_to_speak, lang=lang_code, slow=False)
                tts.save("output.mp3")
                st.audio("output.mp3", format="audio/mp3")
