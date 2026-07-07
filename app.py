import os
import re
import pickle
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- CONFIGURATION ---------------- #

MODEL_FILE = "spam_model.keras"
TOKENIZER_FILE = "tokenizer.pkl"

MAX_WORDS = 5000
MAX_LEN = 50

# ---------------- TEXT CLEANING ---------------- #

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------- TRAIN MODEL ---------------- #

def train_model():

    print("Loading dataset...")

    df = pd.read_csv("spam.csv", encoding="latin-1")

    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    print(df.head())
    print(df["label"].value_counts())

    # Convert labels

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Clean messages

    df["message"] = df["text"].apply(clean_text)

    # Tokenizer

    tokenizer = Tokenizer(
        num_words=MAX_WORDS,
        oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(df["message"])

    sequences = tokenizer.texts_to_sequences(df["message"])

    X = pad_sequences(
        sequences,
        maxlen=MAX_LEN,
        padding="post"
    )

    y = df["label"]

    print("X Shape:", X.shape)
    print("Y Shape:", y.shape)

    # Save tokenizer

    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump(tokenizer, f)

    # Split dataset

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build Model

    model = Sequential()

    model.add(
        Embedding(
            input_dim=MAX_WORDS,
            output_dim=32,
            input_length=MAX_LEN
        )
    )

    model.add(
        SimpleRNN(128)
    )

    model.add(
        Dense(
            64,
            activation="relu"
        )
    )

    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Train

    model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # Save Model

    model.save(MODEL_FILE)

    # Evaluate

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("\nAccuracy:", accuracy)

    predictions = (
        model.predict(X_test, verbose=0) > 0.5
    ).astype(int)

    print(classification_report(
        y_test,
        predictions
    ))

    print(confusion_matrix(
        y_test,
        predictions
    ))

# ---------------- PREDICTION ---------------- #

def predict_sms(message):

    model = load_model(MODEL_FILE)

    with open(TOKENIZER_FILE, "rb") as f:
        tokenizer = pickle.load(f)

    message = clean_text(message)

    sequence = tokenizer.texts_to_sequences([message])

    sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post"
    )

    probability = model.predict(
        sequence,
        verbose=0
    )[0][0]

    if probability >= 0.5:
        return "Spam", probability
    else:
        return "Ham", probability

# ---------------- TRAIN IF MODEL DOESN'T EXIST ---------------- #

if not os.path.exists(MODEL_FILE):
    train_model()

# ---------------- STREAMLIT UI ---------------- #

st.title("📩 SMS Spam Detector using RNN")

st.write("### Many-to-One Recurrent Neural Network Example")

message = st.text_area(
    "Enter SMS Message"
)

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter an SMS message.")
    else:

        prediction, probability = predict_sms(message)

        st.subheader("Prediction")

        st.success(prediction)

        st.write(
            "Confidence:",
            f"{probability*100:.2f}%"
        )