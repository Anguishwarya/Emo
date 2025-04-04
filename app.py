import numpy as np
import streamlit as st
import pandas as pd
import cv2
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from chatbot import get_bard_response  # Custom chatbot module

# Set Streamlit config
st.set_page_config(page_title="EmoTunes", page_icon="🎵")

# Background style
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/uploads/1412282232015a06e258a/4bdd2a58?w=500&auto=format&fit=crop&q=60");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] > div:first-child {
    background-color: rgba(255, 255, 255, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Load dataset
df = pd.read_csv('muse_v3.csv')
df = df.rename(columns={'lastfm_url': 'link', 'track': 'name', 'number_of_emotion_tags': 'emotional', 'valence_tags': 'pleasant'})
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']].sort_values(by=["emotional", "pleasant"]).reset_index(drop=True)

# Emotion split
df_sad, df_fear, df_angry, df_neutral, df_happy = np.array_split(df, 5)

# Recommendation logic
def recommend_songs(emotions):
    emotion_map = {
        'Sad': df_sad,
        'Fearful': df_fear,
        'Angry': df_angry,
        'Neutral': df_neutral,
        'Happy': df_happy
    }
    sample_sizes = [30, 20, 15, 10, 7]
    recommendations = pd.DataFrame()

    for i, emotion in enumerate(emotions[:len(sample_sizes)]):
        sample_df = emotion_map.get(emotion, df_sad)
        recommendations = pd.concat([recommendations, sample_df.sample(n=sample_sizes[i])])

    return recommendations

# Preprocess top emotions
def top_emotions(emotion_list):
    return [emotion for emotion, _ in Counter(emotion_list).most_common()]

# Model initialization
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Load model weights
model.load_weights('model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Title
st.title("🎵 EmoTunes")

# Tabs
tab1, tab2 = st.tabs(["🎭 Emotion-Based Music", "🤖 Chatbot"])

# Tab 1: Emotion Detection and Music Recommendation
with tab1:
    st.subheader("🎭 Scan Your Emotion & Get Songs")

    if st.button('📷 Scan Emotion'):
        cap = cv2.VideoCapture(0)
        detected_emotions = []
        stframe = st.empty()

        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                st.error("Unable to access camera. Please check permissions.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                detected_emotions.append(emotion_dict[max_index])

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Camera Feed")

        cap.release()
        cv2.destroyAllWindows()

        filtered_emotions = top_emotions(detected_emotions)
        st.markdown("### 🎭 Detected Emotions:")
        st.write(", ".join(filtered_emotions))

        songs = recommend_songs(filtered_emotions)

        if not songs.empty:
            st.markdown("### 🎶 Recommended Songs:")
            for _, row in songs.iterrows():
                st.markdown(f"[{row['name']} by {row['artist']}]({row['link']})", unsafe_allow_html=True)
        else:
            st.write("No songs to recommend based on detected emotions.")

# Tab 2: Chatbot
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
