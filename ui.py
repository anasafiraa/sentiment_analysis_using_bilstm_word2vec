import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class UI:
    def __init__(self, model_handler):
        self.model_handler = model_handler

    def display_sidebar(self):
        return st.sidebar.selectbox("Pilih Halaman", ["Pilih Model", "Klasifikasi Sentimen", "Prediksi Teks Baru"])

    def select_model_ui(self):
        st.header('Silahkan Pilih Model!')
        model_dir = 'model_new/'
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]

        def extract_model_number(filename):
            try:
                return int(filename.split('_')[1].split(' ')[0].split('.')[0])
            except (IndexError, ValueError):
                return float('inf')

        model_files = sorted(model_files, key=extract_model_number)
        # Deskripsi untuk setiap model
        model_descriptions = {
             "model_1": "LSTM Units = 64, Dropout LSTM = 0.3, Recurrent Dropout = 0.2, Dense Layer = 128, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10",
            "model_2": "LSTM Units = 96, Dropout LSTM = 0.3, Recurrent Dropout = 0.2, Dense Layer = 128, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10",
            "model_3": "LSTM Units = 128, Dropout LSTM = 0.3, Recurrent Dropout = 0.2, Dense Layer = 128, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10",
            "model_4": "LSTM Units = 256, Dropout LSTM = 0.3, Recurrent Dropout = 0.2, Dense Layer = 128, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10",
            "model_5": "LSTM Units = 64, Dropout LSTM = 0.2, Recurrent Dropout = 0.2, Dense Layer = 128, Dropout Layer = 0.2, Learning Rate = 0.001, Batch Size = 32, Epoch = 10", 
            "model_6": "LSTM Units = 64, Dropout LSTM = 0.4, Recurrent Dropout = 0.2, Dense Layer = 128, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10",
            "model_7": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 128, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10",
            "model_8": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.3, Dense Layer = 128, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10", 
            "model_9": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.4, Dense Layer = 128, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10", 
            "model_10": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.5, Dense Layer = 128, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10", 
            "model_11 (model terbaik)": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10", 
            "model_12": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 96, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 10", 
            "model_13": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 256, Dropout Layer = 0.3, Learning Rate = 0.001, Batch Size = 32, Epoch = 10", 
            "model_14": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.2, Learning Rate = 0.001, Batch Size = 32, Epoch = 10", 
            "model_15": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.3, Learning Rate = 0.001, Batch Size = 32, Epoch = 10", 
            "model_16": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.4, Learning Rate = 0.001, Batch Size = 32, Epoch = 10", 
            "model_17": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.5, Learning Rate = 0.0001, Batch Size = 32, Epoch = 10", 
            "model_18": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.5, Learning Rate = 0.01, Batch Size = 32, Epoch = 10", 
            "model_19": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.5, Learning Rate = 0.005, Batch Size = 32, Epoch = 10", 
            "model_20": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 64, Epoch = 10", 
            "model_21": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 128, Epoch = 10", 
            "model_22": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 256, Epoch = 10", 
            "model_23": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 15", 
            "model_24": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 20", 
            "model_25": "LSTM Units = 64, Dropout LSTM = 0.5, Recurrent Dropout = 0.2, Dense Layer = 64, Dropout Layer = 0.5, Learning Rate = 0.001, Batch Size = 32, Epoch = 25"}

        display_to_original_map = {}
        formatted_model_files = []

        for model in model_files:
            model_name = model.split('.')[0]
            display_name = model_name
            if model_name == "model_11":
                display_name = "model_11 (model terbaik)"
        
            formatted_model_files.append(display_name)
            display_to_original_map[display_name] = model

        if model_files:
            selected_model = st.selectbox('Pilih model', formatted_model_files, index=0)
            
            # Tampilkan deskripsi model jika tersedia
            if selected_model in display_to_original_map:
                original_model_name = display_to_original_map[selected_model]
                st.write("*Konfigurasi Model Bi-LSTM:*")
                description = model_descriptions.get(original_model_name.split('.')[0], "Deskripsi model tidak tersedia")
                st.write(description)
                
            if st.button('Pilih Model'):
                    if original_model_name:
                        model_path = os.path.join(model_dir, original_model_name)
                        success, message = self.model_handler.load_uploaded_model(model_path)

                    if success:
                        st.success(message)
                        st.session_state.model_loaded = True
                    else:
                        st.error(message)
                    
                    # Load history terkait model
                    model_number = extract_model_number(original_model_name)
                    history_path = f'history_new/history_{model_number}.pkl'

                    success, history_or_message = self.model_handler.load_history(history_path)

                    if success:
                        st.success('History berhasil dimuat.')
                        history = history_or_message
                        if hasattr(history, 'history'):
                            self.display_training_history(history.history)
                        else:
                            st.warning('History tidak sesuai format yang diharapkan.')
                    else:
                        st.error(history_or_message)  # Menampilkan pesan kesalahan jika gagal memuat history

                        
    def display_training_history(self, history):
        st.subheader('Grafik Akurasi Pelatihan dan Validasi')
        fig, ax = plt.subplots()
        ax.plot(history['accuracy'], label='Training Accuracy')
        ax.plot(history['val_accuracy'], label='Validation Accuracy')
        ax.set_title('Training and Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        st.pyplot(fig)

        st.subheader('Grafik Loss Pelatihan dan Validasi')
        fig, ax = plt.subplots()
        ax.plot(history['loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

    def text_classification_ui(self, tokenizer, slang_dict):
        st.header('Klasifikasi Sentimen')
        st.subheader('Unggah Dataset dan Lakukan Klasifikasi')

        if 'model' not in st.session_state or st.session_state['model'] is None:
            st.warning('Silakan pilih model terlebih dahulu sebelum melakukan klasifikasi sentimen.')
            return 

        data_test = st.file_uploader("Unggah File (.csv)", type=["csv"])
        if data_test:
            success, result = self.model_handler.load_csv_via_modelhandler(data_test)
            if success:
                df_test = result
                st.write(f'Dataset berhasil diunggah dengan {df_test.shape[0]} baris dan {df_test.shape[1]} kolom.')
                st.dataframe(df_test[['ulasan', 'label']], height=400)

                if st.button('Klasifikasi dan Evaluasi'):
                    results = self.model_handler.evaluate_and_classify(df_test, tokenizer, slang_dict)
                    if results:
                        df_pred, accuracy, precision, recall, f1, conf_matrix, class_report = results
                        st.subheader('Tabel Hasil Klasifikasi')
                        st.dataframe(df_pred, height=400)

                        st.write(f'Accuracy: {accuracy:.4f}')
                        st.write(f'Precision: {precision:.4f}')
                        st.write(f'Recall: {recall:.4f}')
                        st.write(f'F1 Score: {f1:.4f}')

                        # Menampilkan confusion matrix dan classification report
                        self.display_evaluation_results(conf_matrix, class_report)
            else:
                st.error(result)  # Menampilkan pesan kesalahan jika gagal memuat CSV
       
    def display_evaluation_results(self, conf_matrix, class_report):
        st.subheader('Confusion Matrix')
        class_labels = ['positive', 'neutral', 'negative']
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=class_labels, yticklabels=class_labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)

        st.subheader('Classification Report')
        st.text(class_report)

    def text_prediction_ui(self, tokenizer, slang_dict):
        st.header('Prediksi Teks Baru')

        if 'model_loaded' not in st.session_state or not st.session_state.model_loaded:
            st.warning('Silakan pilih model terlebih dahulu.')
            return

        input_text = st.text_area("Masukkan teks untuk prediksi")
        if st.button('Prediksi'):
            if input_text:
                predicted_label, confidence_score = self.model_handler.predict(input_text, tokenizer, slang_dict)
                if predicted_label and confidence_score:
                    st.write(f"Hasil Prediksi: {predicted_label}")
                    st.write(f"Skor Confidence: {confidence_score:.2f}%")
            else:
                st.warning('Masukkan teks untuk diprediksi.')
