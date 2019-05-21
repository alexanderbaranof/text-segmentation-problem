import os
import shutil
import re
import random
from model import ModelClass
import tensorflow as tf
from keras import backend as K
import gensim
from keras.models import load_model as load_model_keras
import numpy as np
import nltk
import pdfminer.high_level
import docx


INPUT_PATH = './files/'
OUTPUT_PATH = './output_files/'
TEMP_FILE_FOLDER_PATH = './tmp/'
NUMBER_OF_WORDS_IN_TMP_FILE = 1000


def load_model():
    model = ModelClass()
    return model


def load_input_files_list():
    return os.listdir(INPUT_PATH)


def clear_tmp():
    files_path = os.listdir(TEMP_FILE_FOLDER_PATH)
    for file_path in files_path:
        os.remove(TEMP_FILE_FOLDER_PATH+file_path)


def text_cleaner(text):
    text = text.lower()  # приведение в lowercase,
    text = re.sub(r'https?://[\S]+', ' ', text)  # замена интернет ссылок
    text = re.sub(r'[\w\./]+\.[a-z]+', ' ', text)
    text = re.sub(r'\d+:\d+(:\d+)?', ' ', text)
    text = re.sub(r'#\w+', ' ', text)  # замена хештегов
    text = re.sub(r'<[^>]*>', ' ', text)  # удаление html тагов
    text = re.sub(r'[\W]+', ' ', text)  # удаление лишних символов
    text = re.sub(r'\b\w\b', ' ', text)  # удаление отдельно стоящих букв
    text = re.sub(r'\b\d+\b', ' ', text)  # замена цифр

    return text


def get_files_of_a_thousand_words(file_path):
    clear_tmp()
    file = open(INPUT_PATH+file_path, 'r')
    text = file.read()
    text = text_cleaner(text)
    words = text.split(' ')
    tmp_text = ''
    for word in words:
        tmp_text += word + ' '
        if len(tmp_text.split(' ')) % NUMBER_OF_WORDS_IN_TMP_FILE == 0:
            tmp_file = open(TEMP_FILE_FOLDER_PATH+str(random.randint(0,100000))+'.txt', 'w')
            tmp_file.write(tmp_text)
            tmp_file.close()
            tmp_text = ''
    tmp_file = open(TEMP_FILE_FOLDER_PATH + str(random.randint(0, 100000)) + '.txt', 'w')
    tmp_file.write(tmp_text)
    tmp_file.close()
    return os.listdir(TEMP_FILE_FOLDER_PATH)


def copy_to_the_appropriate_directory(file_path, label):
    shutil.copy(INPUT_PATH+file_path, OUTPUT_PATH+str(label)+'/')


def clear_output():
    files_path = os.listdir(OUTPUT_PATH+'0/')
    for file_path in files_path:
        os.remove(OUTPUT_PATH+'0/' + file_path)

    files_path = os.listdir(OUTPUT_PATH+'1/')
    for file_path in files_path:
        os.remove(OUTPUT_PATH+'1/' + file_path)

    files_path = os.listdir(OUTPUT_PATH+'2/')
    for file_path in files_path:
        os.remove(OUTPUT_PATH+'2/' + file_path)


def clear_input():
    files_path = os.listdir(INPUT_PATH)
    for file_path in files_path:
        os.remove(INPUT_PATH + file_path)


def tmp_check_exist():
    if not os.path.exists('./tmp/'):
        os.makedirs('./tmp/')

def text2list(text):
    text = text.lower()
    text = re.sub(r'\\d+:\\d+(:\\d+)?', ' время ', text)
    text = re.sub(r'\\b\\w\\b', ' ', text)
    text = re.sub(r'\\b\\d+\\b', ' цифра ', text)
    text = re.sub("^\s+|\n|\r|\s+$", ' ', text) # регулярка удаляет пробелы в начале и конце строки, а также символы переноса
    text = re.sub("([\[]).*?([\]])", "", text) # регулярка удаляет квадратные скобки
    text = re.sub(r'[\\W]+', ' ', text)
    text = re.sub('[!@#$(),^+=*><№«//•»;]', '', text)
    sentences = nltk.sent_tokenize(text)
    sentences_with_filters = list()
    for sent in sentences:
        if len(sent) > 20:
            sentences_with_filters.append(sent)
    return sentences_with_filters


def text2vect(text, model_vect):
    v = np.zeros(300)
    t = 0
    for s in text:
        for word in s.split():
            try:
                v += model_vect[word]
                t+=1
            except KeyError:
                pass
    return v/t


def get_vectors_model():
    return gensim.models.KeyedVectors.load("./model.model")

def get_classification_model():
    return load_model_keras('classification_model.h5')

def check_file_type(file_name):
    file_type = str(file_name).split('.')[1]
    return file_type


def get_part_of_text():
    file_name = load_input_files_list() # забираем имя файла
    file_name = file_name[0]
    file_type = check_file_type(file_name) # возрващаем тип файла
    mode = 0 # забираем режим работы
    classification_model = get_classification_model()
    vectors_model = get_vectors_model()
    if file_type == 'txt' and mode == 0:
        f = open('./files/' + file_name)
        text = f.read()
        sentences = text2list(text)

        number_of_hyp_intro = len(sentences)//6
        p_intro = list()
        p_sections = list()

        for i in range(number_of_hyp_intro):
            srez = sentences[i:i+5]
            tmp_text = ''
            for j in srez:
                tmp_text += j + ' '
            tmp_vect = text2vect(tmp_text, vectors_model)
            tmp_p = classification_model.predict(tmp_vect.reshape(1, 300))[0][0]
            p_intro.append(tmp_p)
            p_sections.append((i, i+5))

        intro_border = p_sections[np.argmax(np.array(p_intro))]
        intro_srez = sentences[intro_border[0]:intro_border[1]]

        #print(intro_srez)

        sentences = sentences[intro_border[1]:]

        number_of_hyp_main = int(len(sentences) * 0.7)
        p_main = list()
        p_sections_main = list()

        for i in range(number_of_hyp_main):
            srez = sentences[i:i + number_of_hyp_main-1]
            tmp_text = ''
            for j in srez:
                tmp_text += j + ' '
            tmp_vect = text2vect(tmp_text, vectors_model)
            tmp_p = classification_model.predict(tmp_vect.reshape(1, 300))[0][1]
            p_main.append(tmp_p)
            p_sections_main.append((i, i + number_of_hyp_main-1))

        main_border = p_sections_main[np.argmax(np.array(p_main))]
        main_srez = sentences[main_border[0]:main_border[1]]

        #print(main_srez)

        sentences = sentences[main_border[1]:]

        number_of_hyp_end = len(sentences) // 2
        p_end = list()
        p_sections_end = list()

        for i in range(number_of_hyp_end):
            srez = sentences[i:i + 1]
            tmp_text = ''
            for j in srez:
                tmp_text += j + ' '
            tmp_vect = text2vect(tmp_text, vectors_model)
            tmp_p = classification_model.predict(tmp_vect.reshape(1, 300))[0][2]
            p_end.append(tmp_p)
            p_sections_end.append((i, i + 1))

        if len(p_end) != 0:
            end_border = p_sections_end[np.argmax(np.array(p_end))]
            end_srez = sentences[end_border[0]:end_border[1]]
        else:
            end_srez = ''

        #print(end_srez)

        intro_text = ''
        for i in intro_srez:
            intro_text += i + ' '

        main_text = ''
        for i in main_srez:
            main_text += i + ' '

        end_text = ''
        for i in end_srez:
            end_text += i + ' '

        #f1 = open('output.txt', 'w')
        #f1.write(str(intro_srez))
        #f1.write('\n\n\n')
        #f1.write(str(main_srez))
        #f1.write('\n\n\n')
        #f1.write(str(end_srez))

        return intro_text, main_text, end_text

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


def main():
    sess = tf.Session()
    K.set_session(sess)
    clear_output()
    tmp_check_exist()
    model = load_model()
    files_path = load_input_files_list()
    type_of_document = check_file_type(files_path[0])
    if type_of_document == 'docx':
        text = getText(INPUT_PATH + files_path[0])
        f = open(INPUT_PATH+files_path[0].split('.')[0]+'.txt', 'w')
        f.write(text)
        f.close()
        os.remove(INPUT_PATH + files_path[0])
        files_path[0] = files_path[0].split('.')[0]+'.txt'
    intro_text, main_text, end_text = get_part_of_text()
    for file_path in files_path:
        tmp_files_list = get_files_of_a_thousand_words(file_path)
        label = model.predict(tmp_files_list)
        #copy_to_the_appropriate_directory(file_path, label)
    clear_tmp()
    clear_input()
    #print('Success!')
    return label, intro_text, main_text, end_text


if __name__ == '__main__':
    main()