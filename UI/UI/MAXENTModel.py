import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pyvi import ViTokenizer
import re
import joblib

class MaxEntClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = LogisticRegression(C=10.0, penalty='l2', solver='liblinear')

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = ViTokenizer.tokenize(text)
        return text

    def _extract_features(self, sentences):
        features = self.vectorizer.transform(sentences)
        return features

    def train(self, sentences, labels):
        preprocessed_sentences = [self.preprocess_text(sentence) for sentence in sentences]
        self.vectorizer.fit(preprocessed_sentences)
        features = self._extract_features(preprocessed_sentences)
        self.model.fit(features, labels)

    def predict(self, sentences):
        preprocessed_sentences = [self.preprocess_text(sentence) for sentence in sentences]
        features = self._extract_features(preprocessed_sentences)
        predictions = self.model.predict(features)
        return predictions

    def evaluate(self, sentences, labels):
        predictions = self.predict(sentences)
        report = classification_report(labels, predictions)
        confusion = confusion_matrix(labels, predictions)
        return report, confusion

if __name__ == '__main__':
    # Load data from file
    df = pd.read_excel('data.xlsx')

    # Split data into train and validation sets
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(df['tokenized_contents'], df['Category'],test_size=0.2,random_state=42)

    # Create and train MaxEntClassifier
    classifier = MaxEntClassifier()
    classifier.train(train_sentences, train_labels)

    # Evaluate the classifier on validation set
    report, confusion = classifier.evaluate(val_sentences, val_labels)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(confusion)

    # Save the trained model
    joblib.dump(classifier, 'maxent_model.joblib')

    # Load the trained model
    loaded_classifier = joblib.load('maxent_model.joblib')

    # Predict labels for test data
    test_sentences = ["Sự phát triển của ngành công nghiệp điện ảnh đã tạo ra nhiều phim ảnh sáng tạo và hấp dẫn, mang đến cho khán giả trải nghiệm giải trí độc đáo.",
                "Công nghệ AI đang thay đổi cách chúng ta tương tác với thế giới, từ việc giúp chúng ta tìm kiếm thông tin nhanh chóng đến việc tạo ra trợ lý ảo thông minh.",
                "Thay đổi khí hậu đang là một vấn đề toàn cầu cấp bách, yêu cầu sự hợp tác và hành động quyết liệt từ tất cả các quốc gia trên thế giới."]
    predictions = loaded_classifier.predict(test_sentences)
    print("Predictions:")
    for sentence, prediction in zip(test_sentences, predictions):
        print(f"Sentence: {sentence} -> Prediction: {prediction}")