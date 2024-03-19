from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal
from UI import Ui_LabelPrediction
from SVMModel import SVMModel, TextDataProcessor
from MAXENTModel import MaxEntClassifier
from LSTMModel import LSTMModel
import re
import numpy as np
from pyvi import ViTokenizer
import string
import joblib
import time
from PyQt5.QtWidgets import QTableWidgetItem

class Worker(QThread):
    finished = pyqtSignal(str, float)

    def __init__(self, model, input_data):
        super(Worker, self).__init__()
        self.model = model
        self.input_data = input_data

    def run(self):
        start_time = time.time() 
        prediction = self.model.predict(self.input_data)
        end_time = time.time()
        prediction_time = end_time - start_time
        self.finished.emit(prediction[0], prediction_time)

class UI():
    def __init__(self):
        self.mainUI = QMainWindow()
        self.main = Ui_LabelPrediction()
        self.main.setupUi(self.mainUI)
        self.mainUI.show()
        
        self.main.findButton.clicked.connect(self.get_sentence)
        self.svm_worker = None
        self.maxent_worker = None
        self.lstm_worker = None

    def tokenize(self, sent):
        with open('vietnamese.txt', 'r', encoding='utf-8') as f:
            stop_words = f.read().split('\n')
        sent = re.sub(f'[{string.punctuation}\d\n]', '', sent)
        sent = re.sub(r'[^\w\s]', '', sent)
        sent = ViTokenizer.tokenize(sent.lower())
        sent = [w for w in sent.split() if w not in stop_words]
        return ' '.join(sent)
    
    def get_sentence(self):
        sentence = self.main.inputText.text()
        token = self.tokenize(sentence)

        #SVM
        new = TextDataProcessor("data.xlsx")
        new.load_data()
        new.split_data()
        new.create_feature_vectors()
        new.encode_labels()
        new.load_model('tfidf.joblib')
        input_features = new._extract_features([token])
        svm = SVMModel()
        svm.load_model('SVM.joblib')
        self.svm_worker = Worker(svm, input_features)
        self.svm_worker.finished.connect(self.update_svm_label)
        self.svm_worker.start()

        #MaxEnt
        maxent = MaxEntClassifier()
        maxent = joblib.load('maxent_model.joblib')
        self.maxent_worker = Worker(maxent, [sentence])
        self.maxent_worker.finished.connect(self.update_maxent_label)
        self.maxent_worker.start()

        #LSTM
        lstm_model = LSTMModel(data_path='data.xlsx', max_sequence_length=100, num_classes=5, lstm_units=64, embedding_dim=100)
        lstm_model.load_data()
        lstm_model.preprocess_data()
        lstm_model.prepare_model()
        lstm_model.train_model()
        self.lstm_worker = Worker(lstm_model, [sentence])
        self.lstm_worker.finished.connect(self.update_lstm_label)
        self.lstm_worker.start()

    def update_svm_label(self, prediction, time):
        self.main.svm.setText(prediction)
        time = f"{time*1000:.2f}ms"
        self.main.outsvm.setText(time)
        self.svm_worker = None

    def update_maxent_label(self, prediction, time):
        self.main.maxent.setText(prediction)
        time = f"{time*1000:.2f}ms"
        self.main.outmaxent.setText(time)
        self.maxent_worker = None

    def update_lstm_label(self, prediction, time):
        self.main.label_2.setText(prediction)
        time = f"{time:.2f}ms"
        self.main.outlstm.setText(time)
        self.lstm_worker = None

if __name__ == "__main__":
    app = QApplication([])
    ui = UI()
    app.exec_()
