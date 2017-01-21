import csv
import pymorphy2
from pandas import read_csv


def get_training_csv(txt_input_file, csv_input_file):
    in_txt = csv.reader(open(txt_input_file, "r", encoding = 'utf-8'), delimiter = '\t')
    out_csv = csv.writer(open(csv_input_file, 'w', encoding = 'utf-8'))
    out_csv.writerow(["rubric", "title", "text"])
    out_csv.writerows(in_txt)


def get_test_csv(txt_input_file, csv_input_file):
    in_txt = csv.reader(open(txt_input_file, "r", encoding = 'utf-8'), delimiter = '\t')
    out_csv = csv.writer(open(csv_input_file, 'w', encoding = 'utf-8'))
    out_csv.writerow(["title", "text"])
    out_csv.writerows(in_txt)


def clean_text(csv_input_file):
    train = read_csv(csv_input_file, encoding = 'utf-8')

    # All text to lower case
    train.title = train.title.str.lower()
    train.text = train.text.str.lower()

    # Cleaning text from useless characters
    train.title = train.title.str.replace(' - ', ' ')
    train.title = train.title.str.replace('[0-9]', '')
    train.title = train.title.str.replace('- ', ' ')
    train.title = train.title.str.replace(' -', ' ')
    train.title = train.title.str.replace(u' . ', ' ')
    train.title = train.title.str.replace(u'.', ' ')
    train.title = train.title.str.replace(',', ' ')
    train.title = train.title.str.replace('!', ' ')
    train.title = train.title.str.replace('/', ' ')
    train.title = train.title.str.replace('(', ' ')
    train.title = train.title.str.replace(')', ' ')
    train.title = train.title.str.replace(':', ' ')
    train.title = train.title.str.replace('"', ' ')
    train.title = train.title.str.replace('«', ' ')
    train.title = train.title.str.replace('»', ' ')
    train.title = train.title.str.replace(u' +', ' ')
    train.title = train.title.str.strip()
    train.text = train.text.str.replace(' - ', ' ')
    train.text = train.text.str.replace('[0-9]', '')
    train.text = train.text.str.replace('- ', ' ')
    train.text = train.text.str.replace(' -', ' ')
    train.text = train.text.str.replace(u' . ', ' ')
    train.text = train.text.str.replace(u'.', ' ')
    train.text = train.text.str.replace(',', ' ')
    train.text = train.text.str.replace('!', ' ')
    train.text = train.text.str.replace('/', ' ')
    train.text = train.text.str.replace('(', ' ')
    train.text = train.text.str.replace(')', ' ')
    train.text = train.text.str.replace(':', ' ')
    train.text = train.text.str.replace('"', ' ')
    train.text = train.text.str.replace('«', ' ')
    train.text = train.text.str.replace('»', ' ')
    train.text = train.text.str.replace(u' +', ' ')
    train.text = train.text.str.strip()

    return train


def my_tokenizer(s, morph):
    t = s.split(' ')
    # print('t: ', t)
    f = ''
    for j in t:
        # print('j: ', j)
        m = morph.parse(j.replace('.', ''))
        if len(m) != 0:
            wrd = m[0]
            # print('wrd: ', wrd)
            if wrd.tag.POS not in ('NUMR', 'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO', 'COMP', 'PRED'):
                f = f + ' ' + str(wrd.normal_form)
    # print('f: ', f)
    return f


def tokenization(train, csv_train_file):
    out_csv = csv.writer(open(csv_train_file, "w", encoding = 'utf-8'))
    out_csv.writerow(["rubric", "tokens"])
    morph = pymorphy2.MorphAnalyzer()
    for i in range(60000):
        # print(i)
        y = train.iloc[i]['title']
        z = train.iloc[i]['text']
        if type(z) == str:
            out_csv.writerow([train.iloc[i]['rubric'], my_tokenizer(y, morph) + my_tokenizer(z, morph)])
        else:
            out_csv.writerow([train.iloc[i]['rubric'], my_tokenizer(y, morph)])


def tokenization_test(test, csv_test_file):
    out_csv = csv.writer(open(csv_test_file, "w", encoding = 'utf-8'))
    out_csv.writerow(["tag", "tokens"])
    morph = pymorphy2.MorphAnalyzer()
    for i in range(15000):
        # print(i)
        y = test.iloc[i]['title']
        z = test.iloc[i]['text']
        if type(z) == str:
            out_csv.writerow([[], my_tokenizer(y, morph) + my_tokenizer(z, morph)])
        else:
            out_csv.writerow([[], my_tokenizer(y, morph)])


def get_output(test, result, output_file):
    with open(output_file, mode = 'w', encoding = 'utf-8') as output:
        output.write('\n'.join(result))


