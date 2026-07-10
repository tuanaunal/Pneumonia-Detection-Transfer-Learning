"""
zatürre sınıflandırma için transfer öğrenme uygulaması
zatürre: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

transfer learning model: densenet121

"""

# import libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator # goruntu verisi yukleme ve data augmentation
from tensorflow.keras.applications import DenseNet121 # onceden egitilmis model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout # model katmanlari
from tensorflow.keras.models import Model # model olusturma
from tensorflow.keras.optimizers import Adam # optimizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # callbacklar

import matplotlib.pyplot as plt # goruntu gosterme
import numpy as np # sayisal islemler icin
import os # dosya islemleri icin
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # karisiklik matrisi ve gorsellestirme

# load data


# data augmentation


# basic visualization


# transfer learning modelin tanimlanmasi: densenet121


# modelin derlenmesi ve callback ayarlari


# modelin egitilmesi ve sonuclarin degerlendirilmesi