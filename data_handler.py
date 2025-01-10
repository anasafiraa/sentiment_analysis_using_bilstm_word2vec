import pickle
import ast
import streamlit as st
import pandas as pd


class DataHandler:
    def load_slang_dict(self, file_path_slang):
        try:
            with open(file_path_slang, 'r') as file:
                file_content = file.read()
            slang_dict = ast.literal_eval(file_content)
            return slang_dict
        except Exception as e:
            st.error(f'Kesalahan saat memuat slang dictionary: {e}')
            return {}

    def load_tokenizer(self, tokenizer_path):
        try:
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
                return tokenizer
        except FileNotFoundError:
            st.error(f'File tokenizer tidak ditemukan. Pastikan path "{tokenizer_path}" benar.')
            return None
        except Exception as e:
            st.error(f'Kesalahan saat memuat tokenizer: {e}')
            return None

    def load_history(self, history_file_path):
        try:
            with open(history_file_path, 'rb') as file:
                history = pickle.load(file)
            return True, history  # Return True jika berhasil
        except FileNotFoundError:
            return False, f'History "{history_file_path}" tidak ditemukan.'  
        except Exception as e:
            return False, f'Terjadi kesalahan saat memuat history: {e}' 
        
    def load_csv(self, file):
        try:
            df = pd.read_csv(file)
            if 'ulasan' not in df.columns or 'label' not in df.columns:
                return False, "File CSV harus memiliki kolom 'ulasan' dan 'label'."
            return True, df
        except Exception as e:
            return False, f"Gagal membaca file CSV: {e}"