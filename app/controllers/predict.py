import pandas as pd
from transformers import pipeline, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from wordcloud import WordCloud
import gensim
from gensim.models import CoherenceModel, LdaModel
import pyLDAvis
import pyLDAvis.gensim


tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

sentiment_model = pipeline("sentiment-analysis", model="app/models/sentiment")
kebijakan_model = pipeline("sentiment-analysis", model="app/models/kebijakan_model", tokenizer=tokenizer)

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'&[A-Za-z0-9]+', '', text) # remove space
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers
    emoji_pattern = re.compile("["
                         u"\U0001F600-\U0001F64F"  # emoticons
                         u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                         u"\U0001F680-\U0001F6FF"  # transport & map symbols
                         u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text

    return text

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text)
    return text


def filteringText(text): # Remove stopwords in a text
    listStopwords = StopWordRemoverFactory().get_stop_words()
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered
    return text


factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words

    text = [stemmer.stem(word) for word in text]
    return text

def preprocess(file):
    df = pd.read_csv(file)
    df['clean_text'] = df['full_text'].apply(cleaningText)
    df['tokenize_text'] = df['clean_text'].apply(tokenizingText)
    df['filter_text'] = df['tokenize_text'].apply(filteringText)
    df['stem_text'] = df['filter_text'].apply(stemmingText)

    return df



def classify_sentiment(text):
    res = sentiment_model(text)
    result = res[0]['label']
    if result == 'neutral':
        result = 'positive'
    return result

def classify_kebijakan(text):
    res = kebijakan_model(text)

    result = res[0]['label']
    if result == 'LABEL_0':
        result = 'Jatim Agro'
    elif result == 'LABEL_1':
        result = 'Jatim Akses'
    elif result == 'LABEL_2':
        result = 'Jatim Amanah'
    elif result == 'LABEL_3':
        result = 'Jatim Berdaya'
    elif result == 'LABEL_4':
        result = 'Jatim Berkah'
    elif result == 'LABEL_5':
        result = 'Jatim Cerdas dan Sehat'
    elif result == 'LABEL_6':
        result = 'Jatim Harmoni'
    elif result == 'LABEL_7':
        result = 'Jatim Kerja'
    elif result == 'LABEL_8':
        result = 'Jatim Sejahtera'

    return result

def process_sentiment(df):
    print("========= SENTIMEN ============")
    df['sentiment'] = df['full_text'].apply(classify_sentiment)

    plt.figure(figsize=(6,4))  # Atur ukuran gambar
    sns.countplot(x=df["sentiment"], palette=['#ff4f4f', '#0cad00', '#C0C0C0' ])
    plt.savefig('./app/static/plot_sentiment.png')

    plt.figure(figsize=(4,4))
    counts = df["sentiment"].value_counts()
    colors = {'negative':'#ff4f4f', 'positive': '#0cad00', 'neutral': '#C0C0C0'}
    plt.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=[colors[label] for label in counts.index], startangle=140)
    plt.savefig('./app/static/pie_sentiment.png')

    return df

def process_kebijakan(df):
    print("========= KEBIJAKAN ============")
    df['kebijakan'] = df['full_text'].apply(classify_kebijakan)

    return df

def lda(df):
    print("========= LDA ============")

    # Create Dictionary
    id2word = gensim.corpora.Dictionary(df["stem_text"])

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in df["stem_text"]]

    # Compute coherence score
    no_of_topics = []
    coherence_score = []

    # for i in range(2,5):
    #     lda_model = LdaModel(corpus=corpus,
    #                             id2word=id2word,
    #                             num_topics=i)
        
    #     print('masuk sini')

    #     # Instantiate topic coherence model
    #     coherence_model_lda = CoherenceModel(model=lda_model, texts=df["stem_text"], dictionary=id2word, coherence='c_v')

    #     # Get topic coherence score
    #     coherence_lda = coherence_model_lda.get_coherence()
    #     no_of_topics.append(i)
    #     coherence_score.append(coherence_lda)

    # max_value = max(coherence_score)
    # print(f"max score = {max_value}")
    # index = coherence_score.index(max_value)
    # print(f"index = {index}")
    # no_of_topic = no_of_topics[index]
    # print(f"banyak topik = {no_of_topic}")

    lda_model = LdaModel(corpus, num_topics=3, id2word=id2word, passes=10)

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word, sort_topics=False)
    #   pyLDAvis.save_html(vis, './app/templates/lda.html')
    html = pyLDAvis.prepared_data_to_html(vis)

    return html

def main_process(file):
    data = preprocess(file)
    data = process_sentiment(data)
    data = process_kebijakan(data)
    data.to_csv('./app/static/result.csv')
    html = lda(data)

    sentiment = data["sentiment"].value_counts().to_dict()
    kebijakan = data["kebijakan"].value_counts().to_dict()

    return {
        'df': data,
        'sentiment': sentiment,
        'kebijakan': kebijakan,
        'html': html
    }