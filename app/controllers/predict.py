# /app/controllers/predict.py

import pandas as pd
from transformers import pipeline, AutoTokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import gensim
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim

# Model dan komponen di-load sekali saja
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
sentiment_model = pipeline("sentiment-analysis", model="app/models/sentiment")
kebijakan_model = pipeline("sentiment-analysis", model="app/models/kebijakan_model", tokenizer=tokenizer)
factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = StopWordRemoverFactory().get_stop_words()

# Fungsi-fungsi utilitas (tidak ada perubahan)
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text); text = re.sub(r'#[A-Za-z0-9]+', '', text)
    text = re.sub(r"http\S+", '', text); text = re.sub(r'[0-9]+', '', text)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF" u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF" "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text); text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation)); text = text.strip(' ')
    return text
def tokenizingText(text): return word_tokenize(text)
def filteringText(text): return [word for word in text if word not in list_stopwords]
def stemmingText(text): return [stemmer.stem(word) for word in text]
def classify_sentiment(text):
    res = sentiment_model(text)
    return 'positive' if res[0]['label'] == 'neutral' else res[0]['label']
def classify_kebijakan(text):
    res = kebijakan_model(text)
    label_map = {'LABEL_0': 'Jatim Agro', 'LABEL_1': 'Jatim Akses', 'LABEL_2': 'Jatim Amanah', 'LABEL_3': 'Jatim Berdaya', 'LABEL_4': 'Jatim Berkah', 'LABEL_5': 'Jatim Cerdas dan Sehat', 'LABEL_6': 'Jatim Harmoni', 'LABEL_7': 'Jatim Kerja', 'LABEL_8': 'Jatim Sejahtera'}
    return label_map.get(res[0]['label'], 'Lainnya')

# --- Fungsi Proses sekarang menerima `status_callback` ---

def preprocess(filepath, status_callback):
    status_callback('update_status', {'message': '➡️ Memulai tahap preprocessing...'})
    df = pd.read_csv(filepath)
    status_callback('update_status', {'message': '1/5 - Membersihkan teks (cleaning)...'})
    df['clean_text'] = df['full_text'].apply(cleaningText)
    status_callback('update_status', {'message': '2/5 - Melakukan tokenisasi...'})
    df['tokenize_text'] = df['clean_text'].apply(tokenizingText)
    status_callback('update_status', {'message': '3/5 - Menghapus stopwords (filtering)...'})
    df['filter_text'] = df['tokenize_text'].apply(filteringText)
    status_callback('update_status', {'message': '4/5 - Melakukan stemming...'})
    df['stem_text'] = df['filter_text'].apply(stemmingText)
    status_callback('update_status', {'message': '5/5 - ✅ Preprocessing selesai.'})
    return df

def process_sentiment(df, status_callback):
    status_callback('update_status', {'message': '➡️ Memulai analisis sentimen...'})
    df['sentiment'] = df['full_text'].apply(classify_sentiment)
    plt.figure(figsize=(6, 4)); sns.countplot(x=df["sentiment"], palette=['#ff4f4f', '#0cad00', '#C0C0C0']); plt.savefig('./app/static/plot_sentiment.png'); plt.close()
    plt.figure(figsize=(4, 4)); counts = df["sentiment"].value_counts(); colors = {'negative': '#ff4f4f', 'positive': '#0cad00', 'neutral': '#C0C0C0'}; plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=[colors.get(label, '#C0C0C0') for label in counts.index], startangle=140); plt.savefig('./app/static/pie_sentiment.png'); plt.close()
    status_callback('update_status', {'message': '✅ Analisis sentimen selesai.'})
    return df

def process_kebijakan(df, status_callback):
    status_callback('update_status', {'message': '➡️ Memulai klasifikasi kebijakan...'})
    df['kebijakan'] = df['full_text'].apply(classify_kebijakan)
    status_callback('update_status', {'message': '✅ Klasifikasi kebijakan selesai.'})
    return df

def lda(df, status_callback):
    status_callback('update_status', {'message': '➡️ Memulai pemodelan topik (LDA)...'})
    id2word = gensim.corpora.Dictionary(df["stem_text"]); corpus = [id2word.doc2bow(text) for text in df["stem_text"]]
    lda_model = LdaModel(corpus, num_topics=3, id2word=id2word, passes=10, random_state=42)
    status_callback('update_status', {'message': '➡️ Menyiapkan visualisasi LDA...'})
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False); html = pyLDAvis.prepared_data_to_html(vis)
    def get_max_topics(topics): return max(topics, key=lambda item: item[1])[0] if topics else -1
    df['topic'] = [get_max_topics(lda_model.get_document_topics(item)) for item in corpus]
    status_callback('update_status', {'message': '✅ Pemodelan topik (LDA) selesai.'})
    return html, df

# --- PERBAIKAN UTAMA: Fungsi ini harus menerima `status_callback` ---
def main_process(filepath, status_callback):
    """Orkestrasi utama, sekarang sepenuhnya independen dari Socket.IO."""
    data = preprocess(filepath, status_callback)
    data = process_sentiment(data, status_callback)
    data = process_kebijakan(data, status_callback)
    
    html, data = lda(data, status_callback)
    
    sentiment = data["sentiment"].value_counts().to_dict()
    kebijakan = data["kebijakan"].value_counts().to_dict()
    
    status_callback('update_status', {'message': '➡️ Menyimpan hasil akhir...'})
    data.to_csv('./app/static/result.csv', index=False)

    status_callback('update_status', {'message': '🚀 Semua proses selesai!'})
    
    # Kembalikan hasil akhir untuk dikirim oleh pemanggilnya
    return {
        'sentiment': sentiment,
        'kebijakan': kebijakan,
        'html': html
    }