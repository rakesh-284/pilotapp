import streamlit as st
from PIL import Image
import pytesseract
from deep_translator import GoogleTranslator
from gtts import gTTS
import io
import cv2
import numpy as np
import re

# Summarization libraries
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# Download necessary language data for summarizer
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

# ==========================================
# 1. IMAGE PREPROCESSING
# ==========================================
def preprocess_image(pil_image):
    img = np.array(pil_image)
    
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Scale up
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Slight blur to remove background noise before thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu's thresholding
    _, processed_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return processed_img

# ==========================================
# 2. BULLET POINT CLEANUP
# ==========================================
def clean_extracted_text(text):
    cleaned_lines =[]
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
            
        # Regex: Force bullet point cleanup
        if re.match(r'^([^a-zA-Z0-9]|o|O)\s', line):
            line = re.sub(r'^([^a-zA-Z0-9]|o|O)\s', '- ', line, count=1)
            
        cleaned_lines.append(line)
        
    return "\n".join(cleaned_lines)


st.title("Image to Speech App 🎙️")

# --- STATE MANAGEMENT ---
if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

# ==========================================
# NEW: TABS FOR UPLOAD VS CAMERA
# ==========================================
tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Take a Photo"])

with tab1:
    uploaded_file = st.file_uploader("Upload an image with English text", type=["jpg", "png", "jpeg"])

with tab2:
    camera_photo = st.camera_input("Take a picture of the text")

# Figure out which image source the user provided
# If they took a photo, use that. Otherwise, use the uploaded file.
image_data = camera_photo if camera_photo is not None else uploaded_file

if image_data is not None:
    image = Image.open(image_data)
    
    # Hide the default large image display if it's from the camera 
    # (since the camera widget already shows the preview)
    if camera_photo is None:
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # ==========================================
    # STEP 1: EXTRACT TEXT
    # ==========================================
    if st.button("1. Extract Text from Image"):
        with st.spinner("Enhancing image and scanning..."):
            
            enhanced_image = preprocess_image(image)
            
            # --psm 4 (Assume a single column of text of variable sizes - great for lists)
            custom_config = r'--oem 3 --psm 4'
            extracted = pytesseract.image_to_string(enhanced_image, config=custom_config)
            
            # Run the text through our bullet-point fixer
            cleaned_text = clean_extracted_text(extracted)
            
            st.session_state.extracted_text = cleaned_text

    # Only show the rest of the app if text has been extracted
    if st.session_state.extracted_text:
        
        st.markdown("### Review & Edit")
        
        # Editable Text Area
        edited_text = st.text_area(
            "Fix any typos here before generating audio:", 
            value=st.session_state.extracted_text, 
            height=200
        )
        
        # Word Count
        word_count = len(edited_text.split())
        st.caption(f"**Word count:** {word_count} words")
        
        # Summarizer
        if word_count > 120:
            st.info("💡 The text is quite long. You can generate a summary to save time.")
            if st.button("Summarize (Keep Important Info)"):
                with st.spinner("Summarizing..."):
                    parser = PlaintextParser.from_string(edited_text, Tokenizer("english"))
                    summarizer = LsaSummarizer()
                    
                    summary_sentences = summarizer(parser.document, sentences_count=3)
                    short_text = " ".join([str(sentence) for sentence in summary_sentences])
                    
                    st.session_state.extracted_text = short_text
                    st.rerun()

        st.divider()
        
        # ==========================================
        # STEP 2: TRANSLATE AND READ
        # ==========================================
        st.markdown("### Generate Audio")
        language = st.selectbox("Choose Language to Listen",["English", "Hindi", "Marathi"])
        
        if st.button("2. Translate & Read Aloud"):
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
