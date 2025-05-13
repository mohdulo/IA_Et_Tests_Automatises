import unittest
import joblib
import pandas as pd

# Charger le modèle et l'ordre des colonnes
model = joblib.load('random_forest_churn_model.pkl')

# Ordre des colonnes d'entrée du modèle (sans la colonne cible 'Churn')
feature_order = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

class TestPredictModel(unittest.TestCase):

    def build_valid_df(self, custom_values):
        """Construit un DataFrame avec toutes les colonnes attendues"""
        row = {col: 0 for col in feature_order}  # initialise à 0
        row.update(custom_values)  # applique les valeurs personnalisées
        return pd.DataFrame([row])[feature_order]

    def test_prediction_known_input(self):
        custom = {
            'gender': 0, 'SeniorCitizen': 0, 'Partner': 1, 'Dependents': 0,
            'tenure': 12, 'PhoneService': 1, 'PaperlessBilling': 1,
            'MonthlyCharges': 70.3, 'TotalCharges': 840.5,
            'Contract_Month-to-month': 1, 'InternetService_DSL': 1,
            'PaymentMethod_Electronic check': 1
        }
        df = self.build_valid_df(custom)
        prediction = model.predict(df)[0]
        self.assertIn(prediction, [0, 1])

    def test_empty_input(self):
        data = pd.DataFrame()
        with self.assertRaises(ValueError):
            model.predict(data)

    def test_missing_column(self):
        data = pd.DataFrame([{'gender': 0, 'SeniorCitizen': 0}])
        with self.assertRaises(ValueError):
            model.predict(data)

    def test_wrong_type_input(self):
        bad_data = pd.DataFrame([{
            'gender': 'Homme', 'SeniorCitizen': 'Non'
        }])
        with self.assertRaises(ValueError):
            model.predict(bad_data)

    def test_batch_prediction(self):
        custom = {
            'gender': 0, 'SeniorCitizen': 0, 'Partner': 0, 'Dependents': 1,
            'tenure': 21, 'PhoneService': 1, 'PaperlessBilling': 0,
            'MonthlyCharges': 64.85, 'TotalCharges': 1336.8,
            'Contract_One year': 1, 'InternetService_DSL': 1,
            'PaymentMethod_Mailed check': 1
        }
        df = pd.DataFrame([custom] * 3)
        df_full = df.apply(lambda row: self.build_valid_df(row.to_dict()).iloc[0], axis=1)
        predictions = model.predict(df_full)
        self.assertEqual(len(predictions), 3)

if __name__ == '__main__':
    unittest.main()
