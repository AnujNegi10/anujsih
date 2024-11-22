import numpy as np
import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from gensim.models import Word2Vec


nltk.download('stopwords')
nltk.download('wordnet')


DISASTER_KEYWORDS = {
    "FLOOD": ["flood", "flash floods", "heavy rains", "water level rise", "inundation", "deluge", "flooding", "rainstorm"],
    "EARTHQUAKE": ["earthquake", "tremor", "quake", "seismic activity", "aftershock", "shockwave", "fault line", "magnitude", "aftershock"],
    "FOREST FIRE": ["forest fire", "wildfire", "blaze", "bushfire", "inferno", "burning forest", "smoke", "firestorm"],
    "LANDSLIDE": ["landslide", "mudslide", "debris flow", "soil erosion", "rockslide", "slip", "landslide warning", "earth slip"],
    "TSUNAMI": ["tsunami", "sea wave", "tidal wave", "wave surge", "ocean wave", "tidal surge", "coastal flooding", "storm surge"],
    "HURRICANE": ["hurricane", "typhoon", "cyclone", "tropical storm", "windstorm", "storm surge", "gale", "tornado", "tropical depression"],
    "TORNADO": ["tornado", "twister", "cyclone", "whirlwind", "funnel cloud", "vortex", "storm", "supercell", "thunderstorm"],
    "HEATWAVE": ["heatwave", "extreme heat", "heat advisory", "record temperatures", "heatstroke", "high temperature", "scorcher", "summer heat"],
    "VOLCANO": ["volcano", "eruption", "lava flow", "magma", "volcanic activity", "crater", "ash cloud", "geothermal eruption", "lava burst"],
    "CYCLONE": ["cyclone", "tropical cyclone", "storm", "tornado", "superstorm", "windstorm", "typhoon", "storm surge", "rotating storm"],
    "WILDFIRE": ["wildfire", "forest fire", "bushfire", "firestorm", "flames", "blaze", "smoke", "burning forest"],
    "HEAVY SNOWFALL": ["heavy snowfall", "snowstorm", "snow accumulation", "snow drift", "icy roads", "winter storm"],
}


def preprocess_text(text):
    lemma = nltk.WordNetLemmatizer()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()  # Remove non-alphabetic characters
    text = ' '.join([lemma.lemmatize(word) for word in text.split() if word not in nltk.corpus.stopwords.words('english')])
    return text


def contains_disaster_keyword(text):
    for category, keywords in DISASTER_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return 1
    return 0


df = pd.read_csv(r'extended_trainDisaster.csv')


df['text'] = df['text'].fillna('').apply(preprocess_text)
df['keyword'] = df['keyword'].fillna('unknown')
df['location'] = df['location'].fillna('unknown')


df['contains_keyword'] = df['text'].apply(contains_disaster_keyword)


df['target'] = df['target'].astype(int)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
word_index = tokenizer.word_index

X_text = tokenizer.texts_to_sequences(df['text'])
X_text = pad_sequences(X_text, maxlen=250)
X_keywords = df['contains_keyword'].values.reshape(-1, 1)
y = df['target'].values.reshape(-1, 1)  # Reshape target

X_train_text, X_test_text, X_train_keywords, X_test_keywords, y_train, y_test = train_test_split(
    X_text, X_keywords, y, test_size=0.2, random_state=42
)

w2v_model = Word2Vec(sentences=df['text'].apply(lambda x: x.split()), vector_size=100, window=5, min_count=1, workers=4)


embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]


text_input = Input(shape=(250,), name="text_input")
embedding_layer = Embedding(
    input_dim=len(word_index) + 1,
    output_dim=embedding_dim,
    input_length=250,
    weights=[embedding_matrix],
    trainable=True
)(text_input)
lstm_layer = Bidirectional(LSTM(128, return_sequences=False, dropout=0.4, recurrent_dropout=0.3))(embedding_layer)


keyword_input = Input(shape=(1,), name="keyword_input")


combined = Concatenate()([lstm_layer, keyword_input])
dense_1 = Dense(128, activation='relu')(combined)
dropout_1 = Dropout(0.4)(dense_1)
dense_2 = Dense(64, activation='relu')(dropout_1)
dropout_2 = Dropout(0.3)(dense_2)
output = Dense(1, activation='sigmoid')(dropout_2)

model = Model(inputs=[text_input, keyword_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


earlystopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

history = model.fit(
    [X_train_text, X_train_keywords], y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[earlystopping, reduce_lr]
)


model.save('disaster_keyword_modelnewone.h5')

def preprocess_input(news):
    news_preprocessed = preprocess_text(news)
    contains_keyword = contains_disaster_keyword(news_preprocessed)
    encoded_text = tokenizer.texts_to_sequences([news_preprocessed])
    padded_text = pad_sequences(encoded_text, maxlen=250)
    return padded_text, np.array([[contains_keyword]])

def predict_news(news):
    text_input, keyword_input = preprocess_input(news)
    
   
    if contains_disaster_keyword(news):
        return 'DisasterRelated', 1.0

    prediction = model.predict([text_input, keyword_input])[0][0]
    sentiment = 'DisasterRelated' if prediction > 0.5 else 'Not Related'
    return sentiment, prediction


news = "The flood caused devastation across the city."
result, score = predict_news(news)
print(f'Result: {result}')
print(f'Score: {score:.2f}')
