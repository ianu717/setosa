import pandas as pd

from flask import Flask, jsonify, Blueprint, request
from preprocessing import preprocess
from service import SeverityInferenceService
from utils import generate_random_collision

api_bp = Blueprint('api', __name__)
severity_inference_service = SeverityInferenceService()

@api_bp.route('/data', methods=['GET'])
def get_data():
    casualties, vehicles, collisions = generate_random_collision()
    processed_data = preprocess(casualties, vehicles, collisions)
    response = jsonify(processed_data.head(1).to_dict(orient='records'))

    return response.json[0]

@api_bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    model_input = pd.DataFrame(data, index=[0])

    response = severity_inference_service.predict(model_input)

    return response

def create_app() -> Flask:
    app_1 = Flask(__name__)
    app_1.register_blueprint(api_bp, url_prefix='/api/v1')
    #Swagger(app)
    return app_1

app = create_app()

if __name__ == '__main__':
    app.run(debug=False)
