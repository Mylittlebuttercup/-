# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from IPython.display import display, HTML, Markdown

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense, Flatten 
from sklearn.model_selection import train_test_split
import tensorflow.keras
    
pd.set_option('display.max_columns', None)
pd.set_option('display.min_rows', 5)

# +
dataset_df_body = pd.read_csv("https://raw.githubusercontent.com/ArthurFDLR/pose-classification-kit/master/pose_classification_kit/datasets/BodyPose_Dataset.csv")
labels = list(dataset_df_body['label'].astype('category').cat.categories)


labels_body = dataset_df_body.label.unique()
labels_body 

# +
display(Markdown("### Complete dataset view"))
display(dataset_df_body)

df_body_labels = dataset_df_body.groupby('label')

display(Markdown("### Number of samples per label"))
display(
    pd.DataFrame(
        [df_body_labels.size()],
        columns=labels_body,
        index=["Nbr of entries"],
        )
)
# -



# +
l = np.array(dataset_df_body['label'].astype('category').cat.codes)

plt.plot(l)
# -

hist = plt.hist(l)

dataset_df_body.head()

dataset_df_body.tail()

dataset_df_body.shape

dataset_df_body['category'] = dataset_df_body['label'].astype('category').cat.codes
dataset_df_body

datas = dataset_df_body.drop(['label','accuracy'], axis=1).to_numpy()
datas.shape
X = datas[:, 0:50]
Y = datas[:, 50]
Y

datas.shape

pose = X[np.where(Y == 12)[0]]

pose.shape

posePartPairs={
    "Torso":[1, 8], 
    "Shoulder (right)":[1, 2],
    "Shoulder (left)":[1, 5],
    "Arm (right)":[2, 3],
    "Forearm (right)":[3, 4],
    "Arm (left)":[5, 6],
    "Forearm (left)":[6, 7],
    "Hip (right)":[8, 9],
    "Thigh (right)":[9, 10],
    "Leg (right)":[10, 11],
    "Hip (left)":[8, 12],
    "Thigh (left)":[12, 13],
    "Leg (left)":[13, 14],
    "Neck":[1, 0],
   ##" "Eye (right)":[0, 15],
   ##" "Ear (right)":[15, 17],
   ##" "Eye (left)":[0, 16],
  ##"  "Ear (left)":[16, 18],
  ##"  ##"Foot (left)":[14, 19],
    ##""Toe (left)":[19, 20],
   ##" "Heel (left)":[14, 21],
    ##""Foot (right)":[11, 22],
   ##" "Toe (right)":[22, 23],
   ##" "Heel (right)":[11, 24],
}
color_map = plt.cm.get_cmap("nipy_spectral", len(posePartPairs))

plt.axis('equal')
sample2d = X[200].reshape(-1,2)
for i,p in enumerate(posePartPairs.values()):
     plt.plot(*sample2d[p].T, 'r' )           

color_map = plt.cm.get_cmap("nipy_spectral", len(posePartPairs))
plt.axis('equal')
sample2d = X[200].reshape(-1,2)
for i,p in enumerate(posePartPairs.values()):
     plt.plot(*sample2d[p].T, c=color_map(i))   

# +
plt.axis('equal')

for j in range(200) :
    sample2d = X[j].reshape(-1,2)
    for i,p in enumerate(posePartPairs.values()):
         plt.plot(*sample2d[p].T, c=color_map(i))   
# -

pose = X[np.where(Y == 1)[0]]
plt.axis('equal')
sample2d = pose[0].reshape(-1,2)
for i,p in enumerate(posePartPairs.values()):
     plt.plot(*sample2d[p].T, c=color_map(i) )  

# +
plt.axis('equal')

pose = X[np.where(Y == 0)[0]]

for j in range(80) :
    sample2d = pose[j].reshape(-1,2)
    for i,p in enumerate(posePartPairs.values()):
         plt.plot(*sample2d[p].T, c=color_map(i))   

# +
plt.axis('equal')

pose = X[np.where(Y == 5)[0]]

for j in range(20) :
    sample2d = pose[j].reshape(-1,2)
    for i,p in enumerate(posePartPairs.values()):
         plt.plot(*sample2d[p].T, c=color_map(i))   

# +
Train, Test = train_test_split(datas, test_size=0.1)
print(Train.shape)
print(Test.shape)

plt.plot(Train[0:100, 50])
# -

ax.axis('equal')              
                ax.set_title(labels[r])        

# +
plt.subplots(5,4,figsize=(15,15))
plt.subplots_adjust(wspace = 0.3, hspace = 0.3)

for r in range(20):
    pose = X[np.where(Y == r)[0]]
    for j in range(20) :
        sample2d = pose[j].reshape(-1,2)
        for i,p in enumerate(posePartPairs.values()):
            ax = plt.subplot(5, 4, r+1)                                
            ax.plot(*sample2d[p].T, c=color_map(i))
            ax.axis('equal')              
            ax.axis('off')              
            ax.set_title(labels[r])               
# -

# ## 데이터학습

Train, Test = train_test_split(datas, test_size=0.1)
X = Train[:, 0:50]
Y = Train[:, 50]

model = Sequential()
model.add(Dense(20, input_dim=50, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=200, verbose=1)
X_T = Test[:, 0:50]
Y_T = Test[:, 50]
model.evaluate(X_T, Y_T)

print(X.shape)
print(X_T.shape)
print(Y.shape)
print(Y_T.shape)

model = Sequential()
model.add(Dense(100, activation='relu',input_shape=(50,)))
model.add(Dense(20, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist= model.fit(X, Y, epochs=50, verbose=1)

model.evaluate(X_T, Y_T)

plt.plot(hist.history['accuracy'])

plt.plot(hist.history['loss'])

# +
#features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]  # full-body
#features = [2,3,4,5,6,7,8,9,10,11,12,13,14,15]    #  upper body
#features = [16,17,18,19,20,21,22,23,24,25,26,27]  #  lower body
#features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,22,23,24,25]    # without nee

# 0,4,7,11,14
features = [0,1,8,9,14,15,22,23,28,29]    #  edge

X = Train[:, features]
Y = Train[:, 50]

model = Sequential()
model.add(Dense(100, activation='relu',input_shape=(len(features),)))
model.add(Dense(20, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist= model.fit(X, Y, epochs=200, verbose=1)

X_T = Test[:, features]
Y_T = Test[:, 50]
model.evaluate(X_T, Y_T)
