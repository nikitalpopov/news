import time
from news.text_edit import *

txt_train_input_file = r"news_train.txt"
csv_train_input_file = r"news_train.csv"
csv_train_file = r"train_file.csv"
txt_test_input_file = r"news_test.txt"
csv_test_input_file = r"news_test.csv"
csv_test_file = r"test_file.csv"
text_test_output_file = r"news_output.txt"

print('Getting training file...')
get_training_csv(txt_train_input_file, csv_train_input_file)
print('')

print('Cleaning training text...')
train = clean_text(csv_train_input_file)
print('')

print('Tokenization training file...')
start = time.clock()
tokenization(train, csv_train_file)
stop = time.clock()
print('Time of tokenization: ', stop - start, ' s')
train = read_csv(csv_train_file, encoding = 'utf-8')
# print(train)
print('')

print('Vectorizing training text...')
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
coder = CountVectorizer()
start = time.clock()
trn = coder.fit_transform(train.tokens)
stop = time.clock()
print('Time of vectorization: ', stop - start, ' s')
# tfidf_transformer = TfidfTransformer()
# train_tfidf = tfidf_transformer.fit_transform(trn)
print('')

print('Creating model...')
# from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
start = time.clock()
clf = LinearSVC().fit(trn, train.rubric)
stop = time.clock()
print('Time of model creation: ', stop - start, ' s')
print('')

print('Getting test file...')
get_test_csv(txt_test_input_file, csv_test_input_file)
print('')

print('Cleaning testing text...')
test = clean_text(csv_test_input_file)
print('')

print('Tokenization testing file...')
start = time.clock()
tokenization_test(test, csv_test_file)
stop = time.clock()
print('Time of tokenization: ', stop - start, ' s')
test = read_csv(csv_test_file, encoding = 'utf-8')
# print(test)
print('')

print('Vectorizing testing text...')
start = time.clock()
tst = coder.transform(test.tokens)
stop = time.clock()
print('Time of vectorization: ', stop - start, ' s')
# test_tfidf = tfidf_transformer.ransform(tst)
print('')

print('Predicting rubric for each test news...')
result = clf.predict(tst)
print('')

print('Outputting result...')
get_output(test, result, text_test_output_file)
print("That's all!")
