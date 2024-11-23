import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('stopwords')
nltk.download('wordnet')


disaster_keywords = [
    
    "flood", "flash floods", "heavy rains", "water level rise", "inundation", "deluge", "flooding", "rainstorm",
    "waterlogging", "river overflow", "dam break", "urban flooding", "coastal flooding", "flash flooding",
    
    # Earthquake-related keywords
    "earthquake", "tremor", "quake", "seismic activity", "aftershock", "shockwave", "fault line", "magnitude",
    "tectonic shift", "epicenter", "seismic wave", "ground shaking", "seismic tremor", "earthquake warning",

    # Fire-related keywords
    "forest fire", "wildfire", "bushfire", "inferno", "burning forest", "smoke", "firestorm", "blaze",
    "fire outbreak", "grassfire", "fire hazard", "fire spread", "burnt area", "fire damage",

    # Landslide-related keywords
    "landslide", "mudslide", "debris flow", "soil erosion", "rockslide", "earth slip", "slope failure",
    "terrain collapse", "land instability", "mudflow", "land movement", "landslide warning",

    # Tsunami-related keywords
    "tsunami", "sea wave", "tidal wave", "wave surge", "ocean wave", "tidal surge", "storm surge",
    "underwater earthquake", "ocean flooding", "tsunami alert", "tsunami warning",

    # Cyclone, Hurricane, Typhoon-related keywords
    "hurricane", "typhoon", "cyclone", "tropical storm", "super cyclone", "windstorm", "storm surge",
    "gale", "tornado", "tropical depression", "eye of the storm", "storm warning", "super typhoon",

    # Tornado-related keywords
    "tornado", "twister", "funnel cloud", "whirlwind", "vortex", "storm cell", "supercell", "storm warning",
    "rotating storm", "tornado outbreak", "tornado path", "destructive winds",

    # Heatwave-related keywords
    "heatwave", "extreme heat", "heat advisory", "record temperatures", "heatstroke", "high temperature",
    "scorcher", "drought", "dry spell", "hot weather", "heat exhaustion", "summer heat", "heat crisis",

    # Volcanic eruption-related keywords
    "volcano", "eruption", "lava flow", "magma", "volcanic activity", "crater", "ash cloud", "geothermal eruption",
    "lava burst", "pyroclastic flow", "volcanic ash", "eruption warning", "volcanic hazard",

    # Snowstorm and Blizzard-related keywords
    "heavy snowfall", "snowstorm", "blizzard", "snow accumulation", "snow drift", "icy roads", "winter storm",
    "avalanche", "whiteout", "frostbite", "snow hazard", "cold snap", "subzero temperatures", "freezing rain",

    # General Disaster-related keywords
    "disaster", "natural disaster", "emergency", "catastrophe", "calamity", "crisis", "hazard", "tragedy",
    "rescue operation", "disaster relief", "aid response", "disaster management", "humanitarian crisis",
    "evacuation", "damage assessment", "disaster recovery",

    # Pandemic-related keywords
    "pandemic", "epidemic", "outbreak", "virus spread", "disease control", "quarantine", "lockdown",
    "public health emergency", "infection", "vaccination", "health crisis", "global pandemic",

    # Other extreme weather-related keywords
    "storm", "thunderstorm", "hailstorm", "windstorm", "lightning strike", "extreme weather",
    "dust storm", "sandstorm", "hail damage", "weather warning", "storm damage", "cold wave",

    # Famine and drought-related keywords
    "famine", "drought", "food crisis", "water scarcity", "crop failure", "hunger crisis",
    "malnutrition", "starvation", "food shortage", "dry season", "water shortage", "water crisis",

    # Industrial disasters
    "industrial accident", "chemical spill", "gas leak", "oil spill", "nuclear disaster", "radiation leak",
    "toxic fumes", "explosion", "factory fire", "industrial hazard", "environmental contamination",

    # Transportation-related disasters
    "plane crash", "train derailment", "shipwreck", "ferry sinking", "road accident", "vehicle collision",
    "aviation disaster", "traffic accident", "maritime disaster", "highway pileup", "transportation hazard",

    # Additional disaster-related terms
    "power outage", "blackout", "electricity failure", "infrastructure damage", "building collapse",
    "dam failure", "levee breach", "urban disaster", "environmental hazard", "public safety warning"
]




def preprocess_text(text):
    lemma = nltk.WordNetLemmatizer()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
    text = ' '.join([lemma.lemmatize(word) for word in text.split() if word not in nltk.corpus.stopwords.words('english')])
    return text


model = load_model("disaster_rnn_model_optimized.h5")


tokenizer = Tokenizer()
word_index = tokenizer.word_index


st.title("Disaster Classification App")
st.write("This app classifies text as disaster-related or not disaster-related using an RNN-based model.")

user_input = st.text_area("Enter the news or tweet text:")

max_len = 250


def preprocess_input(news):
    words = preprocess_text(news).split()
    encoded_review = [word_index.get(word, 2) for word in words]
    padded_review = pad_sequences([encoded_review], maxlen=max_len)
    return padded_review


def predict_news(news):
  
    for keyword in disaster_keywords:
        if keyword in news.lower():
            return "DisasterRelated", 1.0
    

    preprocessed_text = preprocess_input(news)
    prediction = model.predict(preprocessed_text)[0][0]
    sentiment = 'DisasterRelated' if prediction > 0.5 else 'Not Related'
    return sentiment, prediction


if st.button("Predict"):
    if user_input:
        result, score = predict_news(user_input)
        st.write(f"**Result:** {result}")
        st.write(f"**Confidence Score:** {score:.2f}")
    else:
        st.write("Please enter some text to predict.")
