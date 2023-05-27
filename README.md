# ECG Analysis
The electrocardiogram (ECG) can be reliably used as a measure to monitor the function of the cardiovascular system. Recently, there has been a great focus on accurate categorization of heartbeats. Early and accurate detection of arrhythmia types is important for detecting heart disease and choosing appropriate treatment for the patient. Among all classifiers, convolutional neural networks (CNNs) have become very popular for ECG classification. This project presents a detailed investigation of pre-processing techniques on ECG datasets, feature extraction techniques and CNN-based classifiers. In particular, the novelty introduced compared with other methods proposed in the literature is to perform ECG classification on video and real-time sources. According to the results, the suggested method is able to make predictions with average accuracy of 98.47%.

##### To install:
[Python](https://www.python.org/downloads/)

##### Preliminary installation requirements:
Installing packages via the Python console
```bash
pip install numpy
pip install pandas
pip install scikit-image
pip install scikit-learn
pip install seaborn
pip install matplotlib
pip install Keras
pip install opencv-python
pip install scipy
pip install tensorflow
pip install h5py
pip install cycler
pip install Pillow
```
Note : the version of the libraries is indicated in the requirements.txt file

---
##### Model training execution procedure: 
The first step is to load the dataset using the Pandas library : 
```bash
train_df = pd.read_csv('mitbih_train.csv', header=None)
```

As described in the paper, re-sampling of the dataset is performed
```bash
                                    Undersampling
df_0 = train_df[train_df[187] == 0].sample(n=20000, random_state=13)

                                    Oversampling
df_1 = resample(train_df[train_df[187] == 1], n_samples=20000,replace=True,
                                           random_state=13)
df_2 = resample(train_df[train_df[187] == 2], n_samples=20000,replace=True,
                                           random_state=13)
df_3 = resample(train_df[train_df[187] == 3], n_samples=20000,replace=True,
                                           random_state=13)
df_4 = resample(train_df[train_df[187] == 4], n_samples=20000,replace=True,
                                           random_state=13)
```
Final Dataset : 
```bash
train_df_new = pd.concat([df_0, df_1, df_2, df_3, df_4])
```
We perform feature selection through the use of the functional dependencies extracted on the dataset : 
```bash
#from 1 to 10
train_df_new = train_df_new.drop([1,4,5,6,8,9,10],axis=1)
#from 11 to 31
train_df_new = train_df_new.drop([15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],axis=1)
#from 32 to 42
train_df_new = train_df_new.drop([33,35,36,37,38,39,41,42],axis=1)
#from 43 to 53
train_df_new = train_df_new.drop([45,46,47,48,49,51,53],axis=1)
#from 54 to 64
train_df_new = train_df_new.drop([56,57,58,59,60,62,64],axis=1)
#from 65 to 75
train_df_new = train_df_new.drop([66,67,69,71,72,74,75],axis=1)
#from 76 to 86
train_df_new = train_df_new.drop([77,78,80,82,83,85,86],axis=1)
#from 87 to 97
train_df_new = train_df_new.drop([89,91,93,94,95,96,97],axis=1)
#from 98 to 108
train_df_new = train_df_new.drop([102,103,104,105,106,107,108],axis=1)
#from 109 to 119
train_df_new = train_df_new.drop([111,112,113,115,116,117,118,119],axis=1)
#from 120 to 130
train_df_new = train_df_new.drop([123,125,126,127,128,129,130],axis=1)
#from 131 to 141
train_df_new = train_df_new.drop([133,134,135,136,138,139,140,141],axis=1)
#from 142 to 152
train_df_new = train_df_new.drop([145,146,147,148,149,150,151,152],axis=1)
#from 153 to 163
train_df_new = train_df_new.drop([155,156,157,159,160,161,162,163],axis=1)
#from 164 to 174
train_df_new = train_df_new.drop([165,167,168,169,170,171,172,173,174],axis=1)
#from 175 to 186
train_df_new = train_df_new.drop([176,177,178,179,181,182,183,184,185,186],axis=1)
```
Thus we reduce the number of features in the dataset from 187 to 55

We go on to divide the dataset into training and testing sets. Specifically, we use 80% of the instances for training and the remaining 20% for testing:
```bash
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.2, random_state = 42)
```
Perform Convolutional Neural Network training, using the architecture specified in the paper:
```bash
model, history = train_model(X_train, y_train, X_test, y_test)
```
Build learning curves: 
```bash
evaluate_model(history, X_test, y_test, model)
```
Finally, the script outputs the accuracy results and shows the confusion matrix

Run the script via the Python console:
```bash
python train.py
```
##### Feature extraction and classification procedure on static video sources: 
We specify the path of the video to be labeled:
```bash
video_path = 'VideoECG.mp4'
```
Given a video as input, we work on the individual frames. Specifically, as described in the paper, we perform the conversion of the original image from the RGB model to the HSB model:
```bash
img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV).astype("float32")
```
Perform desaturation of the image: 
```bash
(h, s, v) = cv2.split(img)
s = s * 0
s = np.clip(s, 0, 255)
img = cv2.merge([h, s, v])
```
Extract the feature vector:
```bash
extrac = extract_feat(image,peaks[0] - 90, peaks[0] + 96)
```
Use the previously constructed model to make the prediction of the class to be associated with the extracted signal: 
```bash
prediction = loaded_model.predict(np.reshape(extrac, (1, 186, 1)))
```
Label the signal with the class and the probability with which the signal belongs to the class:
```bash
cv2.putText(imgForText, prediction, (x_list[peaks[0]], 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0)
cv2.putText(imgForText, 'prob: ' + prob,(x_list[peaks[0]], 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4,0)
```
Run the script via the Python console:
```bash
python VideoMaker.py
```
---
##### Real-time feature extraction and classification procedure:
The same procedure as described above is used, but with real-time integration
Run the script via the Python console:
```bash
python VideoClassification.py
```
##### Script execution procedure via Python console:
To run the different scripts, you can use the main.py script in the following way:

To label an image given as input:
```bash
python main.py 0 <image_path>
```

To label a signal in real-time:
```bash
python main.py 1
```

To label a video given as input:
```bash
python main.py 2 <video_path>
```
