import sys
import argparse
import time
import cv2
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.engine.saving import model_from_json
import scipy.signal
import os

#Same description as the FeatureExtractionWithSaturation.py script, but with video input

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

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

currentDT = time.localtime()
start_datetime = time.strftime("-%m-%d-%H-%M-%S", currentDT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='input video file name')
    parser.add_argument('--frame_ratio', type=int, default=1, help='analyze every [n] frames')
    parser.add_argument('--process_speed', type=int, default=4,
                        help='Int 1 (fastest, lowest quality) to 4 (slowest, highest quality)')
    parser.add_argument('--end', type=int, default=None, help='Last video frame to analyze')

    args = parser.parse_args()

    frame_rate_ratio = args.frame_ratio
    process_speed = args.process_speed
    ending_frame = args.end

    print('start processing...')

    # Video input
    video_path = 'VideoECG.mp4'
    video_file = video_path

    # Output location
    output_path = 'videos/'
    output_format = '.mp4'
    video_output = output_path + str(start_datetime) + output_format

    # Video reader
    cam = cv2.VideoCapture(video_file)
    input_fps = cam.get(cv2.CAP_PROP_FPS)
    ret_val, orig_image = cam.read()
    video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

    if ending_frame is None:
        ending_frame = video_length

    # Video writer
    output_fps = input_fps / frame_rate_ratio
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, output_fps, (700, 700))

    scale_search = [1, .5, 1.5, 2]  # [.5, 1, 1.5, 2]
    scale_search = scale_search[0:process_speed]

    bool = False

    i = 0  # default is 0
    while(cam.isOpened()) and ret_val is True and i < ending_frame:

        if i % frame_rate_ratio == 0:

            tic = time.time()

            print('Processing frame: ', i)

            img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV).astype("float32")

            imgForText = cv2.resize(orig_image, (700, 700), interpolation=cv2.INTER_CUBIC)

            (h, s, v) = cv2.split(img)
            s = s * 0
            s = np.clip(s, 0, 255)
            img = cv2.merge([h, s, v])

            img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_HSV2BGR)

            image = cv2.resize(img, (700, 700), interpolation=cv2.INTER_CUBIC)

            x_list, y_list = [], []
            for x in np.arange(0, 700, 1):
                for y in np.arange(0, 700, 1):
                    # if np.all(image[y][x] == (0, 0, 0)):
                    if np.all(image[y][x] == (255, 255, 255)):
                        x_list.append(x)
                        y_list.append(700 - y)

            print('Detect peaks without any filters.')
            peaks, _ = scipy.signal.find_peaks(y_list, height=400)

            if len(peaks) != 0:
                try:
                    extrac = extract_feat(image,peaks[0] - 90, peaks[0] + 96)
                    bool = True
                except IndexError:
                    print('Picco non trovato')

                if bool:
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

                    cv2.putText(imgForText, output, (x_list[peaks[0]], 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0)
                    cv2.putText(imgForText, 'prob: ' + str(round(predictions[0][np.argmax(predictions, axis=1)[0]], 2)),
                                (x_list[peaks[0]], 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0)
                    #Video saving
                    out.write(imgForText)

                    toc = time.time()
                    print('processing time is %.5f' % (toc - tic))
                    bool = False

        ret_val, orig_image = cam.read()

        i += 1