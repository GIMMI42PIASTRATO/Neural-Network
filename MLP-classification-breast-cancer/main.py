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

    def train_and_evaluate(self):
        self.load_data()
        self.prepare_data()
        self.normalize_data()
        self.train_model()
        self.evaluate_model()

    def predict(self, data):
        # Normalizza i nuovi dati
        data_normalized = self.scaler.transform(data)
        # Fai previsioni
        predictions = self.model.predict(data_normalized)
        # Decodifica le previsioni in label (M o B)
        decoded_predictions = self.label_encoder.inverse_transform(predictions)
        return decoded_predictions

    def save_model(self):
        dump(self.model, self.model_path)

    def load_model(self):
        self.model = load(self.model_path)


# Utilizzo della classe
model_path = "MLP-classification-breast-cancer\models\\breast_cancer_model.joblib"
classifier = BreastCancerClassifier(
    "MLP-classification-breast-cancer\data\Cancer_Data.csv", model_path=model_path
)
classifier.train_and_evaluate()
classifier.save_model()

data_to_predict = [
    [
        17.99,
        10.38,
        122.8,
        1001,
        0.1184,
        0.2776,
        0.3001,
        0.1471,
        0.2419,
        0.07871,
        1.095,
        0.9053,
        8.589,
        153.4,
        0.006399,
        0.04904,
        0.05373,
        0.01587,
        0.03003,
        0.006193,
        25.38,
        17.33,
        184.6,
        2019,
        0.1622,
        0.6656,
        0.7119,
        0.2654,
        0.4601,
        0.1189,
    ],
    [
        13.54,
        14.36,
        87.46,
        566.3,
        0.09779,
        0.08129,
        0.06664,
        0.04781,
        0.1885,
        0.05766,
        0.2699,
        0.7886,
        2.058,
        23.56,
        0.008462,
        0.0146,
        0.02387,
        0.01315,
        0.0198,
        0.0023,
        15.11,
        19.26,
        99.7,
        711.2,
        0.144,
        0.1773,
        0.239,
        0.1288,
        0.2977,
        0.07259,
    ],
]

prediction = classifier.predict(data_to_predict)
print("Prediction:", prediction)
