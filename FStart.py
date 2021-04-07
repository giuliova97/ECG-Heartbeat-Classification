import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.engine.saving import model_from_json
import scipy.signal

def extract_feat(image, begin, end):
    x_list, y_list = [], [0]
    for x in np.arange(begin, end, 1):
        x_list.append(x-begin)
        for y in np.arange(0, 700, 1):
            #if np.all(image[y][x] == (0, 0, 0)):
            if np.all(image[y][x] == (255, 255, 255)):
                y_list.append(700-y)
                break
            if y==699:
                y_list.append(y_list[x-begin])
    y_list.pop(0)

    return y_list

def show_graph(x_list, y_list, width, height):
    plt.figure(figsize = [width, height])
    plt.scatter(x_list, y_list, marker='.', s=5)
    plt.show()
    return

def start(filepath):
    path = filepath

    img = cv2.imread(path)

    imgForText = cv2.resize(img, (700,700), interpolation=cv2.INTER_CUBIC)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")

    (h, s, v) = cv2.split(img)
    s = s*0
    s = np.clip(s,0,255)
    img = cv2.merge([h,s,v])

    img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_HSV2BGR)

    image = cv2.resize(img, (700,700), interpolation=cv2.INTER_CUBIC)

    cv2.imshow('saturated image', image)
    cv2.waitKey()

    x_list, y_list = [], []
    for x in np.arange(0, 700, 1):
        for y in np.arange(0, 700, 1):
            #if np.all(image[y][x] == (0, 0, 0)):
            if np.all(image[y][x] == (255,255,255)):
                x_list.append(x)
                y_list.append(700-y)

    print('Detect peaks without any filters.')
    peaks, _ = scipy.signal.find_peaks(y_list, height = 400)

    print(peaks)

    try:
        extrac = extract_feat(image, peaks[0] - 90, peaks[0] + 96)
        json_file = open('best_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights("best-model.h5")
        print("Loaded model from disk")

        loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        train_new = np.reshape(extrac, (186, 1))

        scaler = MinMaxScaler(feature_range=(0, 1))

        train_new1 = scaler.fit_transform(train_new)

        p = np.reshape(train_new1, (1, 186))

        predictions = loaded_model.predict(np.reshape(p, (1, 186, 1)))

        print(np.argmax(predictions, axis=1)[0])

        if str(np.argmax(predictions, axis=1)[0]) == '0':
            output = 'N'
        elif str(np.argmax(predictions, axis=1)[0]) == '1':
            output = 'S'
        elif str(np.argmax(predictions, axis=1)[0]) == '2':
            output = 'V'
        elif str(np.argmax(predictions, axis=1)[0]) == '3':
            output = 'F'
        elif str(np.argmax(predictions, axis=1)[0]) == '4':
            output = 'U'

        print(predictions[0][np.argmax(predictions, axis=1)[0]])

        cv2.putText(imgForText, output, (x_list[peaks[0]], 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0)
        cv2.putText(imgForText, 'prob: ' + str(round(predictions[0][np.argmax(predictions, axis=1)[0]], 2)),
                    (x_list[peaks[0]], 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0)
        cv2.imshow('image with classification', imgForText)
        cv2.waitKey()
    except IndexError:
        print('Picco non trovato')