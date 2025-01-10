import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelHandler:
    def __init__(self, data_handler, text_processor):
        self.model = None
        self.data_handler = data_handler
        self.text_processor = text_processor
    
    def load_uploaded_model(self, model_path):
        try:
            self.model = load_model(model_path)
            st.session_state['model'] = self.model  # Simpan model di session state
            message = f'Model berhasil dimuat'
            return True, message  
        except Exception as e:
            message = f'Terjadi kesalahan saat memuat model: {str(e)}'
            return False, message 
        
    def load_history(self, history_path):
   
        success, history_or_message = self.data_handler.load_history(history_path)

        if success:
            # Jika sukses, kembalikan success dan data history
            return True, history_or_message
        else:
            # Jika gagal, kembalikan success dan pesan error
            return False, f"Gagal memuat history: {history_or_message}"

    def load_csv_via_modelhandler(self, file):
        return self.data_handler.load_csv(file)

    def evaluate_and_classify(self, df_test, tokenizer, slang_dict):
        if 'model' in st.session_state and st.session_state['model'] is not None:
            model = st.session_state['model']
        else:
            st.error('Model belum dimuat, silahkan muat model terlebih dahulu.')
            return None

        X_test = df_test['ulasan']
        y_test = df_test['label']
        processed_texts, X_test_padded = [], []

        for text in X_test:
            preprocessed_sequence = self.text_processor.preprocess_text(text, slang_dict, tokenizer)
            processed_texts.append(' '.join(self.text_processor.stemming_text(
                self.text_processor.remove_stopwords(
                    self.text_processor.tokenize_text(self.text_processor.casefolding_text(self.text_processor.clean_text(text)))))))
            X_test_padded.append(preprocessed_sequence[0])

        self.text_processor.save_stemming_cache()
        X_test_padded = np.array(X_test_padded)

        y_pred = model.predict(X_test_padded)
        y_pred_classes = np.argmax(y_pred, axis=1)

        class_labels = ['positive', 'neutral', 'negative']
        y_pred_labels = [class_labels[i] for i in y_pred_classes]

        label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
        y_test_encoded = [label_mapping.get(label, -1) for label in y_test if label in label_mapping]
        y_test_classes = np.array(y_test_encoded)

        df_pred = pd.DataFrame({
            'Hasil Pra-Processing': processed_texts,
            'Actual Label': y_test,
            'Predicted Label': y_pred_labels
        })

        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')

        # Menghitung confusion matrix dan classification report
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
        class_report = classification_report(y_test_classes, y_pred_classes, target_names=class_labels, digits=4)

        return df_pred, accuracy, precision, recall, f1, conf_matrix, class_report

    def predict(self, review, tokenizer, slang_dict):
        if 'model' in st.session_state and st.session_state['model'] is not None:
            # Pra-pemrosesan teks sebelum prediksi
            review_padded = self.text_processor.preprocess_text(review, slang_dict, tokenizer)
            prediction = st.session_state['model'].predict(review_padded)
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence_score = np.max(prediction) * 100
            class_labels = ['Positif', 'Netral', 'Negatif']
            return class_labels[predicted_class], confidence_score
        else:
            st.error('Model belum dimuat.')
            return None, None