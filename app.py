import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from faster_whisper import WhisperModel
from moviepy import VideoFileClip
import cv2
import tempfile
import os
import re

# ---------------- UI ----------------
st.set_page_config(page_title="AttentionX", page_icon="🎥", layout="wide")
st.title("🎥 AttentionX - Smart Vertical Renderer")

# ---------------- API SETUP ----------------
client = genai.Client(
    api_key="AIzaSyD8a42QPr1W_0sVQ1SVfT78aIkORCbqkNw",
    http_options=types.HttpOptions(api_version="v1")
)

@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")

whisper_model = load_whisper()

# ---------------- SMART CROP FUNCTION ----------------
def apply_smart_crop(clip):
    try:
        w, h = clip.size
        target_w = h * (9/16)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        frame = clip.get_frame(0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            (x, y, fw, fh) = faces[0]
            face_center_x = x + (fw / 2)
        else:
            face_center_x = w / 2

        x1 = max(0, min(w - target_w, face_center_x - (target_w / 2)))
        x2 = x1 + target_w
        return clip.cropped(x1=x1, y1=0, x2=x2, y2=h).resized(height=1280)
    except:
        return clip

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload 16:9 Video", type=["mp4"])

if uploaded_file:
    if "video_path" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_file.read())
            st.session_state.video_path = tfile.name
    
    # --- STEP 1: TRANSCRIPTION ---
    if "transcript" not in st.session_state:
        with st.status("🧠 Transcribing...") as s:
            result_segments, _ = whisper_model.transcribe(st.session_state.video_path)
            st.session_state.transcript = "".join([f"[{s.start:.2f}-{s.end:.2f}] {s.text}\n" for s in list(result_segments)])
            s.update(label="Transcription Done!", state="complete")

    # --- STEP 2: AI MOMENTS (STRICT PROMPT) ---
    if st.button("🔥 Step 2: Find Viral Moments"):
        with st.spinner("AI analyzing timestamps..."):
            # We tell the AI to ONLY look at the [seconds] in the transcript
            prompt = f"""
            You are a video editor. Look at the transcript below.
            Identify the TOP 3 most viral segments.
            IMPORTANT: Use ONLY the numbers in brackets [] for START and END.
            
            Format: START_SECONDS | END_SECONDS | REASON
            Example: 15.20 | 45.00 | Funny joke about AI
            
            Transcript:
            {st.session_state.transcript[:8000]}
            """
            response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            st.session_state.ai_moments = response.text.strip()
            st.rerun()

    # --- STEP 3: RENDERER (SMART PARSING) ---
    if "ai_moments" in st.session_state:
        st.markdown("### 📋 AI Suggestions")
        st.code(st.session_state.ai_moments)
        
        if st.button("🎬 Step 3: Render Vertical Reels"):
            full_video = VideoFileClip(st.session_state.video_path)
            lines = st.session_state.ai_moments.split("\n")
            
            valid_clips = 0
            for line in lines:
                if "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        try:
                            # Use regex to find the first number in the string (ignores stars and words)
                            start_str = re.findall(r"[-+]?\d*\.\d+|\d+", parts[0])[0]
                            end_str = re.findall(r"[-+]?\d*\.\d+|\d+", parts[1])[0]
                            
                            start_t = float(start_str)
                            end_t = float(end_str)
                            reason = parts[2].strip() if len(parts) > 2 else "Viral Clip"

                            st.info(f"Rendering Clip {valid_clips+1}: {reason} ({start_t}s - {end_t}s)")
                            
                            clip = full_video.subclipped(start_t, end_t)
                            vertical = apply_smart_crop(clip)
                            
                            out_name = f"final_reel_{valid_clips}.mp4"
                            vertical.write_videofile(out_name, codec="libx264", audio_codec="aac")
                            
                            st.video(out_name)
                            st.download_button(f"Download Reel {valid_clips+1}", open(out_name, "rb"), file_name=out_name, key=f"v_{valid_clips}")
                            
                            valid_clips += 1
                        except Exception as e:
                            st.warning(f"Skipping line: {line} (Reason: Could not find valid seconds)")
            
            full_video.close()
            st.balloons()