import joblib
from sklearn.preprocessing import StandardScaler



def predict_pulsar(model, scaler, input_data):
    """
    Predicts whether a given input data point is a pulsar or not.

    Args:
        model: The trained model.
        scaler: The StandardScaler used for data preprocessing.
        input_data: A list or array of input features.

    Returns:
        A prediction (0 or 1) indicating whether the input data represents a pulsar (1) or not (0).
    """    
    input_data_scaled = scaler.transform(input_data)
    return model.predict(input_data_scaled)[0]

  
  


  
  