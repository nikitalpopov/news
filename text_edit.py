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
    train.title = train.title.str.replace('[0-9]', ' ')
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
    train.text = train.text.str.replace('А', ' А')
    train.text = train.text.str.replace('Б', ' Б')
    train.text = train.text.str.replace('В', ' В')
    train.text = train.text.str.replace('Г', ' Г')
    train.text = train.text.str.replace('Д', ' Д')
    train.text = train.text.str.replace('Е', ' Е')
    train.text = train.text.str.replace('Ё', ' Ё')
    train.text = train.text.str.replace('Ж', ' Ж')
    train.text = train.text.str.replace('З', ' З')
    train.text = train.text.str.replace('И', ' И')
    train.text = train.text.str.replace('Й', ' Й')
    train.text = train.text.str.replace('К', ' К')
    train.text = train.text.str.replace('Л', ' Л')
    train.text = train.text.str.replace('М', ' М')
    train.text = train.text.str.replace('Н', ' Н')
    train.text = train.text.str.replace('О', ' О')
    train.text = train.text.str.replace('П', ' П')
    train.text = train.text.str.replace('Р', ' Р')
    train.text = train.text.str.replace('С', ' С')
    train.text = train.text.str.replace('Т', ' Т')
    train.text = train.text.str.replace('У', ' У')
    train.text = train.text.str.replace('Ф', ' Ф')
    train.text = train.text.str.replace('Х', ' Х')
    train.text = train.text.str.replace('Ц', ' Ц')
    train.text = train.text.str.replace('Ч', ' Ч')
    train.text = train.text.str.replace('Ш', ' Ш')
    train.text = train.text.str.replace('Щ', ' Щ')
    train.text = train.text.str.replace('Э', ' Э')
    train.text = train.text.str.replace('Ю', ' Ю')
    train.text = train.text.str.replace('Я', ' Я')
    train.text = train.text.str.replace(' - ', ' ')
    train.text = train.text.str.replace('[0-9]', ' ')
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


def get_output(result, output_file):
    with open(output_file, mode = 'w', encoding = 'utf-8') as output:
        output.write('\n'.join(result))


