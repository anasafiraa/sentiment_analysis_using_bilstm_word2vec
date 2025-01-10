import re
import nltk
import pickle
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from nltk.corpus import stopwords

# Pastikan stopwords diunduh terlebih dahulu
nltk.download('stopwords')
nltk.download('punkt')

class TextProcessor:
    def __init__(self, max_sequence_length=80, stemming_cache_file='stemming_cache.pkl'):
        self.max_sequence_length = max_sequence_length
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()

        # Load  stemming cache
        self.stemming_cache_file = stemming_cache_file
        self.stemming_cache = self.load_stemming_cache()

    # Fungsi untuk membersihkan teks
    def clean_text(self, text):
        text = re.sub(r'(?:http?\://|https?\://|www\.)\S+', '', text)  # Menghapus link
        text = re.sub(r'<.*?>', '', text)  # Menghapus karakter HTML
        text = re.sub(r'[^\w\s]', ' ', text)  # Menghapus tanda baca dan karakter khusus
        text = re.sub(r'\d+', ' ', text)  # Menghapus angka
        text = re.sub(r'_', ' ', text)  # Menghapus karakter underscore
        text = re.sub(r'\n', ' ', text)  # Mengganti baris baru dengan spasi
        text = re.sub(r'\s+', ' ', text).strip()  # Menghapus spasi berlebihan
        return text

    def casefolding_text(self, text):
        return text.lower()

    def tokenize_text(self, text):
        return nltk.word_tokenize(text)

    def normalize_with_slang(self, tokens, slang_dict):
        return [slang_dict.get(token, token) for token in tokens]

    # Fungsi untuk menghapus stopwords
    stopwords_indonesia = set(stopwords.words('indonesian'))

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token.lower() not in self.stopwords_indonesia]

    # Fungsi stemming yang menyimpan hasilnya ke cache
    def stemming_text(self, tokens):
        stemmed_tokens = []
        for token in tokens:
            if token in self.stemming_cache:
                stemmed_token = self.stemming_cache[token]
            else:
                stemmed_token = self.stemmer.stem(token)
                self.stemming_cache[token] = stemmed_token  # Save result in cache
            stemmed_tokens.append(stemmed_token)

        return stemmed_tokens

    def preprocess_text(self, text, slang_dict, tokenizer):
        text = self.clean_text(text)
        text = self.casefolding_text(text)
        tokens = self.tokenize_text(text)
        tokens = self.normalize_with_slang(tokens, slang_dict)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stemming_text(tokens)
        return pad_sequences(tokenizer.texts_to_sequences([' '.join(tokens)]), maxlen=self.max_sequence_length, padding='post')

    # Load stemming cache from file
    def load_stemming_cache(self):
        if os.path.exists(self.stemming_cache_file):
            with open(self.stemming_cache_file, 'rb') as f:
                return pickle.load(f)
        return {}

    # Save stemming cache to file (simpan di akhir batch)
    def save_stemming_cache(self):
        with open(self.stemming_cache_file, 'wb') as f:
            pickle.dump(self.stemming_cache, f)
