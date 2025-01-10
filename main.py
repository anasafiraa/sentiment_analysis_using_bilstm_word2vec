import streamlit as st
from text_processor import TextProcessor
from model_handler import ModelHandler
from data_handler import DataHandler
from ui import UI

# Main Class for Execution
class Main:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.data_handler = DataHandler()
        self.model_handler = ModelHandler(self.data_handler, self.text_processor)
        self.ui = UI(self.model_handler)
        self.tokenizer = None
        self.slang_dict = None

    def run(self):
        self.tokenizer = self.data_handler.load_tokenizer('training/tokenizer.pkl')
        self.slang_dict = self.data_handler.load_slang_dict('kamus slang/slangwords.txt')
        page = self.ui.display_sidebar()
        if page == 'Pilih Model':
            self.ui.select_model_ui()
        elif page == 'Klasifikasi Sentimen':
            self.ui.text_classification_ui(self.tokenizer, self.slang_dict)
        elif page == 'Prediksi Teks Baru':
            self.ui.text_prediction_ui(self.tokenizer, self.slang_dict)

if __name__ == '__main__':
    st.markdown("# Analisis Sentimen Ulasan Aplikasi Shopee Menggunakan *Bidirectional Long Short-Term Memory*")
    app = Main()
    app.run()
