# ProgettoFDSML

# ECG Analysis

##### Installare:
[Python](https://www.python.org/downloads/)

##### Requisiti d'installazione preliminari:
Installare i pacchetti tramite la console di Python.
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
Nota : la versione delle librerie è indicata nel file requirements.txt

---
##### Procedura d'esecuzione dell'addestramento del modello: 
La prima fase consiste nell'effettuare il caricamento del dataset utilizzando la libreria Pandas : 
```bash
train_df = pd.read_csv('mitbih_train.csv', header=None)
```

Come descritto nel paper, viene effettuato il re-sampling del dataset
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
Dataset finale : 
```bash
train_df_new = pd.concat([df_0, df_1, df_2, df_3, df_4])
```
Effettuiamo la feature selection attraverso l'utilizzo delle functional dependencies estratte sul dataset : 
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
Quindi riduciamo così il numero di feature del dataset da 187 a 55

Andiamo a suddividere il dataset in training e test set. In particolare, utilizziamo l'80% delle istanze per il training e il restante 20% per il testing
```bash
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.2, random_state = 42)
```
Andiamo ad effettuare il training della Convolutional Neural Network, utilizzando l'architettura specificata nel paper
```bash
model, history = train_model(X_train, y_train, X_test, y_test)
```
Costruiamo le learning curves: 
```bash
evaluate_model(history, X_test, y_test, model)
```
Infine lo script dà in output i risultati di accuracy e mostra la confusion matrix

Eseguire lo script tramite la console di Python.
```bash
python train.py
```
##### Procedura di feature extraction e classification su fonti video statiche: 
Specifichiamo il path del video da etichettare:
```bash
video_path = 'VideoECG.mp4'
```
Dato in input un video, lavoriamo sui singoli frame. In particolare, come descritto nel paper, effettuiamo la conversione dell'immagine originale dal modello RGB al modello HSB
```bash
img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2HSV).astype("float32")
```
Effettuiamo la desaturazione dell'immagine: 
```bash
(h, s, v) = cv2.split(img)
s = s * 0
s = np.clip(s, 0, 255)
img = cv2.merge([h, s, v])
```
Estraiamo il vettore delle caratteristiche 
```bash
extrac = extract_feat(image,peaks[0] - 90, peaks[0] + 96)
```
Utilizziamo il modello costruito precedentemente per effettuare la previsione della classe da associare al segnale estratto: 
```bash
prediction = loaded_model.predict(np.reshape(extrac, (1, 186, 1)))
```
Etichettiamo il segnale con la classe e la probabilità con cui il segnale appartiene alla classe:
```bash
cv2.putText(imgForText, prediction, (x_list[peaks[0]], 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0)
cv2.putText(imgForText, 'prob: ' + prob,(x_list[peaks[0]], 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4,0)
```
Eseguire lo script tramite la console di Python.
```bash
python VideoMaker.py
```
---
##### Procedura di feature extraction e classification in real-time:
Viene utilizzata la stessa procedura sopra descritta, ma con l'integrazione in real-time
Eseguire lo script tramite la console di Python.
```bash
python VideoClassification.py
```
##### Procedura di esecuzione script tramite console di Python:
Per eseguire i diversi script, è possibile utilizzare lo script main.py nel seguente modo:

Per etichettare un'immagine data in input:
```bash
python main.py 0 <image_path>
```

Per etichettare un segnale in real-time:
```bash
python main.py 1
```

Per etichettare un video dato in input:
```bash
python main.py 2 <video_path>
```
