import numpy as np
import streamlit as st
import pandas as pd
import cv2
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Set Streamlit page configuration
st.set_page_config(page_title="EmoTunes", page_icon="🎵")

# Load the dataset
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

# Function to recommend songs based on emotions
def fun(emotions_list):
    data = pd.DataFrame()
    sample_sizes = [30, 20, 15, 10, 7]  # Sample sizes for different combinations of emotions

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

# Preprocessing to remove duplicates and sort by frequency
def pre(emotion_list):
    emotion_counts = Counter(emotion_list)
    sorted_emotions = [emotion for emotion, count in emotion_counts.most_common()]
    return sorted_emotions

# Build the emotion detection model
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

# Load pre-trained weights
model.load_weights('model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Streamlit UI
st.title("EmoTunes")
st.markdown("<h5 style='text-align: center;'>Click on a song to listen</h5>", unsafe_allow_html=True)

# Scan emotions on button click
if st.button('SCAN EMOTION'):
    cap = cv2.VideoCapture(0)  # Open the webcam
    detected_emotions = []

    stframe = st.empty()  # Placeholder for video frames

    for _ in range(20):  # Process 20 frames
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

        # Display the frame in Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Camera Feed")

    cap.release()
    cv2.destroyAllWindows()

    # Process detected emotions
    unique_emotions = pre(detected_emotions)

    # Display detected emotions
    st.markdown("<h4>Detected Emotions:</h4>", unsafe_allow_html=True)
    st.write(", ".join(unique_emotions))

    # Recommend songs based on detected emotions
    recommended_songs = fun(unique_emotions)

    # Display recommended songs
    if not recommended_songs.empty:
        st.markdown("<h4>Recommended Songs:</h4>", unsafe_allow_html=True)
        for link, artist, name in zip(recommended_songs['link'], recommended_songs['artist'], recommended_songs['name']):
            st.markdown(f"<a href='{link}' target='_blank'>{name} by {artist}</a>", unsafe_allow_html=True)
    else:
        st.write("No songs to recommend based on detected emotions.")
