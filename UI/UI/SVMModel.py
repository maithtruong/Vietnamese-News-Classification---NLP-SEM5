import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextDataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.X_train, self.X_val, self.X_test = None, None, None
        self.y_train, self.y_val, self.y_test = None, None, None
        self.vectorizer = None
        self.le = None

    def load_data(self):
        self.machine_df = pd.read_excel(self.file_path)
        self.contents = self.machine_df['tokenized_contents']
        self.categories = self.machine_df['Category']
    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.contents, self.categories, test_size=test_size, random_state=42)
        
    def create_feature_vectors(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.X_train = self.vectorizer.fit_transform(self.X_train)
        self.X_test = self.vectorizer.transform(self.X_test)
        
    def _extract_features(self, sentences):
        features = self.vectorizer.transform(sentences)
        return features
    
    def encode_labels(self):
        self.le = LabelEncoder()
        self.y = self.le.fit_transform(self.categories)
        joblib.dump(self.le, 'label_encoder.pkl')

 
    def save_vector(self, file_path):
        joblib.dump(self.vectorizer, file_path)

    def load_model(self, file_path):
        self.vectorizer = joblib.load(file_path)
class SVMModel:
    def __init__(self):
        self.model = SVC(C=10, kernel='rbf')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        return accuracy, precision, recall, f1

    def save_model(self, file_path):
        joblib.dump(self.model, file_path)

    def load_model(self, file_path):
        self.model = joblib.load(file_path)


# Ví dụ sử dụng
if __name__ == "__main__":
    processor = TextDataProcessor("data.xlsx")
    processor.load_data()
    processor.split_data()
    processor.create_feature_vectors()
    processor.encode_labels()
    
    processor.save_vector('tfidf.joblib')
    svm_model = SVMModel()
    svm_model.train(processor.X_train, processor.y_train)
    svm_model.save_model('SVM.joblib')
    accuracy, precision, recall, f1 = svm_model.evaluate(processor.X_test, processor.y_test)

    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1-score: {f1}")
    new = TextDataProcessor("data.xlsx")
    new.load_model('tfidf.joblib')
    sentence = 'hãng xe châu âu lexus toyota mazda lọt top thương_hiệu ôtô tin_cậy consumer reports'
    input = new._extract_features([sentence])
    svm = SVMModel()
    svm.load_model('SVM.joblib')
    t = svm.predict(input.toarray())
    print(t)

