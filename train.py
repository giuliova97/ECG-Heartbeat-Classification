import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from keras.utils.np_utils import to_categorical
from keras.layers import Convolution1D, MaxPool1D, Dense, Input, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from itertools import cycle, islice
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp


def train_model(X_train, y_train, X_test, y_test):
    # input signal image shape
    im_shape = (X_train.shape[1], 1)

    # Input layer
    inputs_cnn = Input(shape=(im_shape),
                       name='inputs_cnn')

    # Block 1
    conv1_1 = Convolution1D(64, (6), activation='relu',
                            input_shape=im_shape)(inputs_cnn)
    conv1_1 = BatchNormalization()(conv1_1)

    pool1 = MaxPool1D(pool_size=(3), strides=(2),
                      padding='same')(conv1_1)

    # Block 2
    conv2_1 = Convolution1D(64, (3), activation='relu',
                            input_shape=im_shape)(pool1)
    conv2_1 = BatchNormalization()(conv2_1)

    pool2 = MaxPool1D(pool_size=(3), strides=(2),
                      padding='same')(conv2_1)

    # Block 3
    conv3_1 = Convolution1D(64, (3), activation='relu',
                            input_shape=im_shape)(pool2)
    conv3_1 = BatchNormalization()(conv3_1)

    pool3 = MaxPool1D(pool_size=(3), strides=(2),
                      padding='same')(conv3_1)
    # Flatten
    flatten = Flatten()(pool3)

    # Dense Block (Fully Connected layer)
    dense1 = Dense(64, activation='relu')(flatten)
    dense2 = Dense(32, activation='relu')(dense1)

    # Output Block
    output = Dense(5, activation='softmax', name='output')(dense2)

    # compile model
    model = Model(inputs=inputs_cnn, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=8),
                 ModelCheckpoint(filepath='best-model.h5',
                                 monitor='val_loss',
                                 save_best_only=True)]
    # training
    print('Training...')
    history = model.fit(X_train, y_train, epochs=40, batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)

    model_json = model.to_json()
    with open("best_model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("best-model.h5")
    print("Saved model to disk")

    return (model, history)

def evaluate_model(history, X_test, y_test, model):
    scores = model.evaluate((X_test), y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    #Learning curves
    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()

    fig2, ax_loss = plt.subplots()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model - Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.show()
    target_names = [str(i) for i in range(5)]

    y_true = []
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba = model.predict(X_test)
    prediction = np.argmax(prediction_proba, axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)

#Read the dataset from csv file through Pandas library
train_df = pd.read_csv('mitbih_train.csv', header=None)

print(train_df.head())
print(train_df.info())

class_dist = train_df[187].astype(int).value_counts()
print(class_dist)

print(class_dist.mean())

my_colors = list(islice(cycle(['orange', 'r', 'g', 'y', 'k']), None, len(train_df)))

#Distribution of each class before re-sampling
p = train_df[187].astype(int).value_counts().plot(kind='bar', title='Count (target)', color=my_colors);
plt.title('Class Distribution: Before re-sampling')
plt.show()

#Re-sampling

#Under-sampling
df_0 = train_df[train_df[187] == 0].sample(n=20000, random_state=13)
#Over-sampling
df_1 = resample(train_df[train_df[187] == 1], n_samples=20000,replace=True,
                                           random_state=13)
df_2 = resample(train_df[train_df[187] == 2], n_samples=20000,replace=True,
                                           random_state=13)
df_3 = resample(train_df[train_df[187] == 3], n_samples=20000,replace=True,
                                           random_state=13)
df_4 = resample(train_df[train_df[187] == 4], n_samples=20000,replace=True,
                                           random_state=13)
#New dataset after re-sampling
train_df_new = pd.concat([df_0, df_1, df_2, df_3, df_4])


print('Before columns reduction')
print(train_df_new.head())
print(train_df_new.info())

#Feature selection using functional dependencies discovered on dataset

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

#Reshaping of dataset for fixing indexes
train_df_new.columns = range(train_df_new.shape[1])

print('After columns reduction')
print(train_df_new.head())
print(train_df_new.info())

#Distribution of each class after re-sampling
train_df_new[55].astype(int).value_counts().plot(kind='bar', title='Count (target)', color=my_colors);
plt.title('Class Distribution: After re-sampling')
plt.show()

#An example for each class
c = train_df_new.groupby(55, group_keys=False)\
        .apply(lambda train_df_new: train_df_new.sample(1))
print(c)

fig, axes = plt.subplots(5, 1, figsize=(16, 15))

leg = iter(['N', 'S', 'V', 'F', 'U'])
colors = iter(['skyblue', 'red', 'lightgreen', 'orange', 'black'])
for i, ax in enumerate(axes.flatten()):
    ax.plot(c.iloc[i, :54].T, color=next(colors))
    ax.legend(next(leg))
plt.show()

# data preparation : Labels
target_train = train_df_new[55]

#Converts a class vector (integers) to binary class matrix
y_train = to_categorical(target_train)

# data preparation : Features
X_train = train_df_new.iloc[:,:54].values[:,:, np.newaxis]

#Splitting of dataset into training_set(80%) and testing_set(20%)
X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size = 0.2, random_state = 42)

#Training of model
model, history = train_model(X_train, y_train, X_test, y_test)

#Learning curves
evaluate_model(history, X_test, y_test, model)

#Confusion matrix
y_pred = model.predict(X_test)
y_pred_clean = np.zeros_like(y_pred)
for idx, i in enumerate(np.argmax(y_pred,axis=1)):
    y_pred_clean[idx][i] = 1

print(classification_report(y_test, y_pred_clean))

conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred_clean, axis=1), normalize=True)
print(conf_matrix)
cmn = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cmn)

plt.figure(figsize=(10, 7))
sns.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix Correlation-Coefficient')
plt.show()

#ROC curve
lw = 2
n_classes = 5

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(5):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['skyblue', 'red', 'lightgreen', 'orange', 'black'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['skyblue', 'red', 'lightgreen', 'orange', 'black'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


