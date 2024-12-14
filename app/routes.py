from flask import Blueprint, request, render_template
from .controllers.predict import predict_topic, predict_sentiment

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return render_template('home.html', title="Home Page")


@main.route('/predict_topic')
def topic():
    text = request.args.get('text')
    res = predict_topic(text)
    return res

@main.route('/predict_sentiment')
def sentiment():
    text = request.args.get('text')
    res = predict_sentiment(text)
    return res