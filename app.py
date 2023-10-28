from flask import Flask, render_template,request
from flask import jsonify
from utilities import predict_pipeline
import pickle


# Create an instance of the Flask class
app = Flask(__name__)


# Load the model
with open('models/pipeline.pickle', 'rb') as f:
    loaded_pipe = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

# Define the route for model prediction
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        user_input = request.form['content']
        text = [user_input]
        predictions = predict_pipeline(text)
        result = predictions[0]['label']

        return render_template('index.html', text=user_input, result=result)

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)