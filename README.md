# Model_Reliability_Task-


This repository contains a Flask API for predicting model probabilities based on input features. The API takes a JSON request with meta-features and returns a JSON response containing probabilities for different regression models.

## Files Included
- `app.py`: Flask application file containing the API logic.
- `model.pkl`: Pre-trained model used for inference.
- `scaler.pkl`: Standard scaler for preprocessing.
- `pca.pkl`: PCA transformer for dimensionality reduction.
- `Calibration Task.ipynb`: Notebook for the model development.

## How to Run the API
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. The API will be available at `http://127.0.0.1:5000`.

## Example Request and Response

### Request
Use the following JSON structure to send a `POST` request to the `/predict` endpoint:

#### Endpoint: `POST /predict`
```json
{
    "0": {
        "num_clients": 10,
        "Sum of Instances in Clients": 13821,
        "Max. Of Instances in Clients": 1383,
        "Min. Of Instances in Clients": 1382,
        "Stddev of Instances in Clients": 0.3,
        "Average Dataset Missing Values %": 4.992465884583631,
        "Min Dataset Missing Values %": 4.121475054229935,
        "Max Dataset Missing Values %": 5.571635311143271,
        "Stddev Dataset Missing Values %": 0.4489697353421885,
        "Entropy of Target Stationarity": 0.3250829733914482
    },
    "1": {
        "num_clients": 5,
        "Sum of Instances in Clients": 4031,
        "Entropy of Target Stationarity": 0.6730116670092565
    },
    ...
}
```

### Response
The API will return the following JSON response:
```json
{
    "0": {
        "ELASTICNETCV": 0.01865,
        "HUBERREGRESSOR": 0.04917,
        "LASSO": 0.03675,
        "LinearSVR": 0.05987,
        "QUANTILEREGRESSOR": 0.02231,
        "XGBRegressor": 0.81324
    },
    "1": {
        "ELASTICNETCV": 0.02179,
        "HUBERREGRESSOR": 0.07082,
        "LASSO": 0.32416,
        "LinearSVR": 0.10403,
        "QUANTILEREGRESSOR": 0.04251,
        "XGBRegressor": 0.43669
    },
    ...
}
```

## Testing the API with Postman
1. Open Postman.
2. Set the method to `POST` and the URL to `http://127.0.0.1:5000/predict`.
3. In the `Headers` tab, add:
   ```
   Key: Content-Type
   Value: application/json
   ```
4. In the `Body` tab, select `raw` and paste the request JSON structure as shown above.
5. Click `Send` to get the response.

