# Thư viện sử dụng
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyvi import ViTokenizer
import re
from tensorflow.keras import backend as K
import os
import joblib
import pickle

import warnings
warnings.filterwarnings("ignore")

# Class Attention
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# Class LSTMModel
class LSTMModel:
    def __init__(self, data_path, max_sequence_length, num_classes, lstm_units=64, embedding_dim=100, stopword_file='vietnamese.txt'):
        self.data_path = data_path
        self.max_sequence_length = max_sequence_length
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.embedding_dim = embedding_dim
        self.stopword_file = stopword_file
        self.tokenizer = None

    def load_data(self):
        df = pd.read_excel(self.data_path)
        self.data = df[['tokenized_contents', 'Category']]

    def preprocess_data(self):
        # Split data into tokenized_contents and Category
        self.tokenized_contents = self.data['tokenized_contents']
        self.categories = self.data['Category']

        # Label encoding
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.categories)
        label_encoder.classes_ = np.unique(self.categories)  # Set classes_ sau khi fit_transform
        np.save('classes.npy', label_encoder.classes_)  # Lưu classes_ vào tệp classes.npy

        # Text preprocessing
        self.tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ', oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.tokenized_contents)
        sequences = self.tokenizer.texts_to_sequences(self.tokenized_contents)
        self.vocab_size = len(self.tokenizer.word_index) + 1

        # Padding sequences
        self.padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)

    def prepare_model(self):
        # Convert labels to one-hot encoding
        num_classes = len(np.unique(self.labels))
        self.one_hot_labels = to_categorical(self.labels, num_classes=num_classes)

        # Build LSTM model with attention
        input_layer = tf.keras.layers.Input(shape=(self.max_sequence_length,))
        embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_sequence_length)(input_layer)
        lstm_layer = tf.keras.layers.LSTM(self.lstm_units, return_sequences=True)(embedding_layer)
        attention_layer = Attention()(lstm_layer)
        output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(attention_layer)
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, test_size=0.2, epochs=5):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.padded_sequences, self.one_hot_labels, test_size=test_size, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs)

    def evaluate_model(self):
        # Evaluate the model on the testing set
        y_pred = np.argmax(self.model.predict(self.padded_sequences), axis=1)
        y_true = np.argmax(self.one_hot_labels, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")


    def predict(self, sentences):
        if self.tokenizer is None:
            raise ValueError("The tokenizer has not been initialized. Please preprocess the data first.")

        # Text preprocessing for prediction
        cleaned_sentences = [re.sub(r'[^\w\s]', '', sentence) for sentence in sentences]
        tokenized_sentences = [ViTokenizer.tokenize(sentence) for sentence in cleaned_sentences]
        sequences = self.tokenizer.texts_to_sequences(tokenized_sentences)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)

        # Make predictions
        predictions = np.argmax(self.model.predict(padded_sequences), axis=1)

        # Decode predictions
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load('classes.npy', allow_pickle=True)
        predicted_categories = label_encoder.inverse_transform(predictions)

        return predicted_categories
    
    def save_model(self, file_path):
        self.model.save(file_path, save_format='tf')

    def load_model(self, file_path):
        self.model = load_model(file_path)
    def save_tokenizer_and_encoder(self, tokenizer_path, encoder_path):
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            np.save(encoder_path, self.labels.classes_)
            
    def load_tokenizer_and_encoder(self, tokenizer_path, encoder_path):
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        label_classes = np.load(encoder_path, allow_pickle=True)
        label_encoder = LabelEncoder()
        label_encoder.classes_ = label_classes
        self.labels = label_encoder

if __name__ == '__main__':
    # Initialize the LSTM model
    lstm_model = LSTMModel(data_path='data.xlsx', max_sequence_length=100, num_classes=5, lstm_units=64, embedding_dim=100)

    # Load and preprocess the data
    lstm_model.load_data()
    lstm_model.preprocess_data()

    # Prepare and train the model
    lstm_model.prepare_model()
    lstm_model.train_model()
    
    # Evaluate the model
    lstm_model.evaluate_model()

    loaded_model = LSTMModel(data_path='data.xlsx', max_sequence_length=100, num_classes=5, lstm_units=64, embedding_dim=100)

    # Make predictions
    sentences = ["Sự phát triển của ngành công nghiệp điện ảnh đã tạo ra nhiều phim ảnh sáng tạo và hấp dẫn, mang đến cho khán giả trải nghiệm giải trí độc đáo.",
                "Công nghệ AI đang thay đổi cách chúng ta tương tác với thế giới, từ việc giúp chúng ta tìm kiếm thông tin nhanh chóng đến việc tạo ra trợ lý ảo thông minh.",
                "Thay đổi khí hậu đang là một vấn đề toàn cầu cấp bách, yêu cầu sự hợp tác và hành động quyết liệt từ tất cả các quốc gia trên thế giới."]
    predicted_categories = lstm_model.predict(sentences)
    print(predicted_categories)