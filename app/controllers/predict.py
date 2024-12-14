from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# model = AutoModelForSequenceClassification.from_pretrained("app\models\\topic")
# tokenizer = AutoTokenizer.from_pretrained("app\models\\topic")
topic_model = pipeline("sentiment-analysis", model="app\models\\topic")


sentiment_model = pipeline("sentiment-analysis", model="app\models\sentiment")

res = sentiment_model('BARU satu bulan menjabat, Wakil Presiden Gibran Rakabuming Raka aktif bagi-bagi bantuan sosial atau bansos. Hal yang jadi sorotan adalah penyematan label di kantong tas bansos: Bantuan Wapres Gibran. ')


def predict_topic(text):
    res = topic_model(text)
    if res[0]['label'] == 'LABEL_0':
        response = {
            'message' : 'Persepsi Kesejahteraan dan Desa',
            'content' : 'Percakapan di Twitter terkait kemiskinan di Jawa Timur cenderung mengaitkan keberhasilan program kesejahteraan dengan perbaikan kondisi di pedesaan, menyoroti tantangan pembangunan wilayah tersebut.',
            'score' : res[0]['score']
        }
    elif res[0]['label'] == 'LABEL_1':
        response = {
            'message' : 'Aspirasi dan Kritik',
            'content' : 'Terdapat sentimen beragam terhadap kebijakan Jatim Sejahtera, mencakup harapan sekaligus kritik dari masyarakat terhadap upaya pemerintah dalam mengurangi kemiskinan.',
            'score' : res[0]['score']
        }
    else:
        response = {
            'message' : 'Ketimpangan Sosial',
            'content' : ' Diskusi sering kali menggarisbawahi isu ketimpangan antara yang miskin dan kaya, dengan sentimen yang mengarah pada pandangan kritis terhadap kesenjangan tersebut.',
            'score' : res[0]['score']
        }
    return response

def predict_sentiment(text):
    res = sentiment_model(text)
    if res[0]['label'] == 'positive':
        res[0]['message'] = 'Positif'
    elif res[0]['label'] == 'negative':
        res[0]['message'] = 'Negatif'
    elif res[0]['label'] == 'neutral':
        res[0]['message'] = 'Netral'
    return res