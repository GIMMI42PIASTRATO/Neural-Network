from main import BreastCancerClassifier

# Create an instance of the BreastCancerClassifier class
classifier = BreastCancerClassifier(
    data_path="data\\Cancer_Data.csv", model_path="models\\breast_cancer_model.joblib"
)
classifier.load_data()
classifier.prepare_data()
classifier.normalize_data()
classifier.evaluate_model()
