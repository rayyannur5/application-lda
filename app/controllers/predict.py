from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# model = AutoModelForSequenceClassification.from_pretrained("app\models\\topic")
# tokenizer = AutoTokenizer.from_pretrained("app\models\\topic")
topic_model = pipeline("sentiment-analysis", model="app\models\\topic")


sentiment_model = pipeline("sentiment-analysis", model="app\models\sentiment")

res = sentiment_model('BARU satu bulan menjabat, Wakil Presiden Gibran Rakabuming Raka aktif bagi-bagi bantuan sosial atau bansos. Hal yang jadi sorotan adalah penyematan label di kantong tas bansos: Bantuan Wapres Gibran. ')


def predict_topic(text):
    res = topic_model(text)
    return res

def predict_sentiment(text):
    res = sentiment_model(text)
    return res