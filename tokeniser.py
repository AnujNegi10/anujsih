import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd

# Example: Save the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.read_csv(r'extended_trainDisaster.csv')['text'])

with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
