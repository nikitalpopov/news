import pandas
from sklearn.feature_extraction.text import HashingVectorizer

from news.text_edit import *

txt_train_input_file = r"news_train.txt"
csv_train_input_file = r"news_train.csv"
csv_train_file = r"train_file.csv"
txt_test_input_file = r"news_test.txt"
csv_test_input_file = r"news_test.csv"
csv_test_file = r"test_file.csv"
text_test_output_file = r"news_output.txt"

print('Getting training file...')
get_training_csv(txt_input_file, csv_input_file)

print('Cleaning training text...')
train = clean_text(csv_input_file)

print('Tokenization training file...')
tokenization(train, csv_train_file)
train = read_csv(csv_train_file, encoding = 'utf-8')
print(train)

print('Vectorizing training text...')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
coder = CountVectorizer()
tfidf_transformer = TfidfTransformer()
trn = coder.fit_transform(train.tokens)
train_tfidf = tfidf_transformer.fit_transform(trn)
# print(train_tfidf)
# print(train_tfidf.data)

print('Creating model...')
# from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
clf = LinearSVC().fit(trn, train.rubric)
# print(clf)

print('Getting test file...')
get_test_csv(txt_test_input_file, csv_test_input_file)

print('Cleaning testing text...')
test = clean_text(csv_test_input_file)

print('Tokenization testing file...')
tokenization_test(test, csv_test_file)
test = read_csv(csv_test_file, encoding = 'utf-8')
print(test)

print('Vectorizing testing text...')
tst = coder.fit_transform(test.tokens)
test_tfidf = tfidf_transformer.fit_transform(tst)

print('Predicting rubric for each test news...')
# result = clf.predict(tst.tokens)
# result = clf.decision_function(tst.tokens)
# get_output(test, result, text_test_output_file)

