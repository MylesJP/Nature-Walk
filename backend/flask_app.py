from flask import Flask, request, Response
from gradio_client import Client
import json

app = Flask(__name__)

def gradioPrediction(image_path):
    client = Client("https://mylesjp-nature-walk.hf.space/")
    result_tuple = client.predict(image_path, api_name="/predict")

    # Read the JSON file
    json_file_path = result_tuple[0]
    with open(json_file_path, "r") as f:
        result_json_str = f.read()

    # Parse the JSON string
    result_dict = json.loads(result_json_str)

    # Extract the top result
    top_result = result_dict["confidences"][0]
    top_label = top_result["label"]
    top_confidence = top_result["confidence"]

    if top_confidence < 0.2:
        # If less than 0.2, probably not a flower or a flower it doesn't recognize
        return "Hmmmm, I don't recognize this flower."
    elif top_confidence < 0.4:
        # If less than 0.4, could be either of the top 2 results
        return f"Could be {result_dict['confidences'][0]['label']} or {result_dict['confidences'][1]['label']}."
    else:
        # If greater than 0.4, just return the top result
        return top_label.title()
    
@app.route('/predict', methods=["POST"])
def predict():
    data = request.json
    image_url = data.get('image_url')

    prediction = gradioPrediction(image_url)

    return Response(prediction, mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)