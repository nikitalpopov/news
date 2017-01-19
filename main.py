import pymorphy2
import sklearn
import pandas
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB

from news.text_edit import *

txt_input_file = r"news_train.txt"
csv_input_file = r"news_train.csv"
morph = pymorphy2.MorphAnalyzer()

# get_csv(txt_input_file, csv_input_file)
train = clear_text(csv_input_file)
coder = HashingVectorizer(tokenizer = f_tokenizer)
TrainNotDuble = train.drop_duplicates()
trn = coder.fit_transform(TrainNotDuble.text.tolist())
print(trn)
