import csv
from pandas import read_csv
import pymorphy2


def get_csv(txt_input_file, csv_input_file):
    in_txt = csv.reader(open(txt_input_file, "r", encoding = 'utf-8'), delimiter = '\t')
    out_csv = csv.writer(open(csv_input_file, 'w', encoding = 'utf-8'))
    out_csv.writerow(["rubric", "title", "text"])
    out_csv.writerows(in_txt)


def clear_text(csv_input_file):
    train = read_csv(csv_input_file, encoding = 'utf-8')

    # All text to lower case
    train.title = train.title.str.lower()
    train.text = train.text.str.lower()

    # Cleaning text from useless characters
    train.text = train.text.str.replace(u' - ?', u'-')
    train.text = train.text.str.replace(u'[0-9]', '')
    train.text = train.text.str.replace(u'- ', ' ')
    train.text = train.text.str.replace(u' -', ' ')
    train.text = train.text.str.replace(u'  *', ' ')
    train.text = train.text.str.replace(u'. ', ' ')
    train.text = train.text.str.replace(u'.', '')
    train.text = train.text.str.replace(u', ', ' ')
    train.text = train.text.str.replace(u',', '')
    train.text = train.text.str.replace(u'! ', ' ')
    train.text = train.text.str.replace(u'!', '')
    train.text = train.text.str.replace(u'? ', ' ')
    train.text = train.text.str.replace(u'?', '')
    train.text = train.text.str.replace(u'/', '')
    train.text = train.text.str.replace(u'(', '')
    train.text = train.text.str.replace(u')', '')
    train.text = train.text.str.replace(u'"', '')
    train.text = train.text.str.replace(u'«', '')
    train.text = train.text.str.replace(u'»', '')

    # Dropping 'title' column
    train = train.drop(['title'], axis = 1)
    print(train)
    return train


def f_tokenizer(s):
    morph = pymorphy2.MorphAnalyzer()
    t = s.split(' ')
    # print('t: ', t)
    f = []
    for j in t:
        # print('j: ', j)
        m = morph.parse(j.replace('.', ''))
        if len(m) != 0:
            wrd = m[0]
            # print('wrd: ', wrd)
            if wrd.tag.POS not in ('NUMR', 'PREP', 'CONJ', 'PRCL', 'INTJ', 'NPRO', 'COMP', 'PRED'):
                f.append(wrd.normal_form)
    # print('f: ', f)
    return f
#

# def training():
