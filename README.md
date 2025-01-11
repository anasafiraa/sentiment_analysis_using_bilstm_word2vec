# SENTIMENT ANALYSIS OF SHOPEE APP REVIEWS ON GOOGLE PLAY STORE USING BI-LSTM AND WORD2VEC METHODS

Shopee is one of the largest e-commerce platforms in Indonesia that provides various products and services. User reviews of the Shopee application are an important source of information for companies to understand user needs and experiences, in order to improve service quality. This research aims to conduct sentiment analysis on user reviews of the Shopee application with the Bidirectional Long Short-Term Memory (Bi-LSTM) and Word2Vec methods. Word2Vec is used to convert the words in the review into numerical vectors that reflect the meaning and context of each word, so that the model can understand the meaning and relationship between words more precisely and accurately. Meanwhile, Bidirectional Long Short-Term Memory (Bi-LSTM), as a recurrent neural network, analyzes the word order from two directions, so that a thorough understanding of the sentence context can be achieved. The dataset used consists of 50,000 reviews from the Google Play Store. Tests were conducted with 8 configuration scenarios to obtain optimal results. The best configuration involved 64 LSTM units, LSTM dropout 0.5, recurrent dropout 0.2, dense layer 64 neurons, layer dropout 0.5, learning rate 0.001, batch size 32, and 10 training epochs. The best model showed an accuracy of 86.62% on training data and 85.75% on validation data, while evaluation on test data resulted in an accuracy of 86.15%. The model also showed average values of macro precision, recall, and F1-Score of 77.42%, 67.93%, and 70.44%, as well as weighted precision, recall, and F1-Score of 85.05%, 86.15%, and 85.07% respectively.

## 📚 Project Overview
This research focuses on sentiment analysis of Shopee app reviews using deep learning-based approach with BiLSTM model and word representation using Word2Vec.

## 💾 Dataset Shopee Reviews
The dataset used consists of 50,000 Shopee app reviews obtained through a scraping process from the Google Play Store. Once collected, the reviews are automatically labeled using the indobert-sentiment-classifier pretrain model to classify the sentiment into positive, negative, or neutral. This dataset has challenging characteristics as it contains a wide range of emotional expressions, informal language usage, as well as a variety of slangs and abbreviations commonly encountered in app reviews.

## 🔧 Data Pre-processing
The pre-processing stage aims to clean the raw data and transform it into a format that can be optimally processed by the model. The process includes:

Data Cleaning: Removing symbols, numbers, and irrelevant characters.
Case Folding: Converts all letters to lowercase.
Tokenization: Separates text into individual word units.
Normalization: Converts slang and abbreviations to standardized form using a slang dictionary.
Stopword Removal: Eliminates common words that are not significant.
Stemming: Converts words to their base form.

## 🔍 Word2Vec Embedding
After pre-processing is complete, the next step is to convert the review text into numeric vectors using Word2Vec. The Word2Vec model used is a pre-trained model that has been trained on Indonesian Wikipedia data. This process allows words that have similar meanings to be represented with adjacent vectors in vector space, so that the semantic relationship between words can be better understood by the model.

## 🤖 BiLSTM (Bidirectional Long Short-Term Memory)
BiLSTM (Bidirectional Long Short-Term Memory) is a type of deep neural network designed to handle sequential data such as text. Unlike a regular LSTM, a BiLSTM processes information in two directions, namely forward and backward, so it is able to capture context from both sides of a sentence. This makes BiLSTM very effective in understanding the relationship between words in reviews, especially in the context of sentiment analysis.

### ⚙️ BiLSTM Model Experiment
In this study, 25 BiLSTM model configurations were tested with various parameter combinations to find the best model that produces optimal performance on sentiment analysis. Each model was tested with different parameters, such as the number of units in the LSTM layer, dropout, recurrent dropout, learning rate, and number of epochs.
Each tested model is evaluated based on accuracy on training and validation data, as well as evaluation metrics on test data, such as precision, recall, and F1-score.

## 🏆 Best Model Results
Based on the results of the analysis, Model 11 can be concluded as the best model for this study. This model has an LSTM unit value of 64, LSTM Dropout of 0.5, Recurrent Dropout of 0.2, Dense Layer with 64 neurons, Layer Dropout of 0.5, Learning Rate of 0.001, Batch Size of 32, and Epoch of 10.

This model produces 86.62% accuracy on training data and 85.75% on validation data. The model performance evaluation results show an accuracy of 86.15%, with average values of macro precision, macro recall, and macro F1-Score of 77.42%, 67.93%, and 70.44%, respectively, and weighted precision, weighted recall, and weighted F1-Score reaching 85.05%, 86.15%, and 85.07%, respectively.

These results show that Model 11 provides the best performance compared to other models for sentiment analysis.

## 🚀 How to Run the Program
1. Clone the repository.
2. Create virtual environment
3. Install dependencies from [requirements.txt] (https://github.com/anasafiraa/sentiment_analysis_using_bilstm_word2vec/blob/main/requirements.txt
4. Download a model file and put inside assets folder. [Download Model] (https://drive.google.com/file/d/1V2vOR4YrtujSGmzovQiZgQsxqI9OsfLP/view?usp=drive_link)
5. Just run the app python -u streamlit run ["main.py"] (https://github.com/anasafiraa/sentiment_analysis_using_bilstm_word2vec/blob/main/main.py)

# CONTACT
anasafira579@gmail.com
