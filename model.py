from keras.models import model_from_json
import pickle

INPUT_PATH = './input_files/'
OUTPUT_PATH = './output_files/'
TEMP_FILE_FOLDER_PATH = './tmp/'


class ModelClass:
    def __init__(self):
        json_file = open('./model_files/json_model.json', "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights('./model_files/model.h5')
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
        with open('./model_files/TfidfVectorizer.pickle', 'rb') as f:
            self.vect = pickle.load(f)

    def predict(self, files_list):
        labels = []
        for file in files_list:
            f = open(TEMP_FILE_FOLDER_PATH+file, 'r')
            text = f.read()
            text_vector = self.vect.transform([text])
            label = self.model.predict_classes(text_vector)[0]
            labels.append(label)

        r = labels.count(0)
        e = labels.count(1)
        u = labels.count(2)
        if u <= e <= r or e <= u <= r:
            return 0
        elif u <= r <= e or r <= u <= e:
            return 1
        elif r <= e <= u or e <= r <= u:
            return 2
