import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import dump, load


class BreastCancerClassifier:
    def __init__(self, data_path, model_path=None):
        self.model_path = model_path
        self.data_path = data_path
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42
        )

    def load_data(self):
        self.dataset = pd.read_csv(self.data_path)

    def prepare_data(self):
        X = self.dataset.drop(["id", "diagnosis", "Unnamed: 32"], axis=1)  # Features
        print(X)
        Y = self.dataset["diagnosis"]  # Target
        Y = self.label_encoder.fit_transform(Y)
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.25, random_state=42
        )

    def normalize_data(self):
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_model(self):
        self.model.fit(self.X_train, self.Y_train)

    def evaluate_model(self):
        Y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.Y_test, Y_pred)
        print("Accuracy:", accuracy)

    def predict(self, data):
        data = self.scaler.transform(data)
        print(data)
        return self.model.predict(data)

    def train_and_evaluate(self):
        self.load_data()
        self.prepare_data()
        self.normalize_data()
        self.train_model()
        self.evaluate_model()

    def save_model(self):
        dump(self.model, self.model_path)

    def load_model(self):
        self.model = load(self.model_path)


# Utilizzo della classe
model_path = "models\\breast_cancer_model.joblib"
classifier = BreastCancerClassifier("data\Cancer_Data.csv", model_path=model_path)
classifier.train_and_evaluate()
classifier.save_model()

random_data = [18.61, 20.25, 122.1, 1094, 0.0944, 0.1066, 0.149, 0.07731]
