import numpy as np
import streamlit as st
import pandas as pd
import cv2
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from chatbot import get_bard_response  # Import chatbot function

# Set Streamlit page configuration
st.set_page_config(page_title="EmoTunes", page_icon="🎵")

# Set background
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("https://images.unsplash.com/uploads/1412282232015a06e258a/4bdd2a58?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTJ8fG11c2ljJTIwYmFja2dyb3VuZHxlbnwwfHwwfHx8MA%3D%3D");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stSidebar"] > div:first-child {{
    background-color: rgba(255, 255, 255, 0.3);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load dataset
df = pd.read_csv('muse_v3.csv')
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
df = df.sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

# Split data into emotion categories
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

# Recommend songs based on emotions
def fun(emotions_list):
    data = pd.DataFrame()
    sample_sizes = [30, 20, 15, 10, 7]

    for i, emotion in enumerate(emotions_list[:len(sample_sizes)]):
        sample_size = sample_sizes[i]
        if emotion == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=sample_size)])
        elif emotion == 'Angry':
            data = pd.concat([data, df_angry.sample(n=sample_size)])
        elif emotion == 'Fearful':
            data = pd.concat([data, df_fear.sample(n=sample_size)])
        elif emotion == 'Happy':
            data = pd.concat([data, df_happy.sample(n=sample_size)])
        else:
            data = pd.concat([data, df_sad.sample(n=sample_size)])
    return data

# Preprocess detected emotions
def pre(emotion_list):
    return [emotion for emotion, _ in Counter(emotion_list).most_common()]

# Build the model structure
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Load weights
model.load_weights('model.h5')

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# App title
st.title("🎵 EmoTunes")

# Tabs
tab1, tab2 = st.tabs(["🎭 Emotion-Based Music", "🤖 Chatbot"])

# --------------------- TAB 1: Emotion-Based Music ---------------------
with tab1:
    st.subheader("🎭 Upload Your Image & Get Songs")

    uploaded_file = st.file_uploader("📤 Upload an image of your face", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            st.warning("😕 No face detected in the image. Please upload a clear image of your face.")
        else:
            detected_emotions = []

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                detected_emotions.append(emotion_dict[max_index])

            unique_emotions = pre(detected_emotions)

            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown("### 🎭 Detected Emotions:")
            st.write(", ".join(unique_emotions))

            recommended_songs = fun(unique_emotions)

            if not recommended_songs.empty:
                st.markdown("### 🎶 Recommended Songs:")
                for link, artist, name in zip(recommended_songs['link'], recommended_songs['artist'], recommended_songs['name']):
                    st.markdown(f"[{name} by {artist}]({link})", unsafe_allow_html=True)
            else:
                st.write("No songs to recommend based on detected emotions.")

# --------------------- TAB 2: Bard Chatbot ---------------------
with tab2:
    st.subheader("🤖 Ask the AI Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask me anything...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        response = get_bard_response(user_query)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
