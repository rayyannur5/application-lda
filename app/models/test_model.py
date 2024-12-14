import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


sentiment_model = pipeline("sentiment-analysis", model="app\models\sentiment")

res = sentiment_model('BARU satu bulan menjabat, Wakil Presiden Gibran Rakabuming Raka aktif bagi-bagi bantuan sosial atau bansos. Hal yang jadi sorotan adalah penyematan label di kantong tas bansos: Bantuan Wapres Gibran. ')

print(res)

model = AutoModelForSequenceClassification.from_pretrained("app\models\\topic")
tokenizer = AutoTokenizer.from_pretrained("app\models\\topic")

topic_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = topic_model("Sdh jamak kalau pemerintah menolak data yg tdk sesuai dng kepentingan penguasa meskipun data itu dibuat sendiri oleh pemerintah Makanya sering sekali data BPS disesuaikan kepentingan penguasa termasuk data kemiskinan Bkn kebijakannya yg diubah tp datanya")

print(res)