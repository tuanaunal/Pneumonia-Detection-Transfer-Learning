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

# load data and data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale = 1/255.0, # normalization 0-1 arasina getirme
    horizontal_flip = True, # yatayda cevirme
    rotation_range = 10, # +-10 derece dondurme
    brightness_range = [0.8, 1.2], # parlaklik ayari
    validation_split = 0.1 # validation icin %10 ayirma
) # train data = train + validation

test_datagen = ImageDataGenerator(rescale = 1/255.0) # test icin sadece normalization

DATA_DIR = "./chest_xray/chest_xray"
IMG_SIZE = (224, 224) # modelin bekledigi input boyutu
BATCH_SIZE = 64 # batch boyutu
CLASS_MODE = "binary" # ikili siniflandirma

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"), # egitim verisinin bulundugu klasor
    target_size = IMG_SIZE, # GORUNTULERI IMG_SIZE boyutuna yeniden boyutlandirma
    batch_size = BATCH_SIZE, # batch boyutu
    class_mode = CLASS_MODE, # ikili siniflandirma (zaturre yok/var)
    subset = "training", # egitim verisi
    shuffle = True # veriyi karistirma
)

val_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"), # validation verisinin bulundugu klasor
    target_size = IMG_SIZE, 
    batch_size = BATCH_SIZE,
    class_mode = CLASS_MODE, # ikili siniflandirma
    subset = "validation", # validation verisi
    shuffle = False # validation verisi sirali olmalidir
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"), # test verisinin bulundugu klasor
    target_size = IMG_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = CLASS_MODE, # ikili siniflandirma
    shuffle = False # test verisi sirali olmalidir
)

# basic visualization
class_names = list(train_gen.class_indices.keys()) # sinif isimleri [normal, pneumonia]
images, labels = next(train_gen) # bir batch veri al

plt.figure(figsize=(10, 4))
for i in range(4):
    ax = plt.subplot(1, 4, i+1)
    ax.imshow(images[i])
    ax.set_title(class_names[int(labels[i])])
    ax.axis("off")
plt.tight_layout()
plt.show()

# transfer learning modelin tanimlanmasi: densenet121
base_model = DenseNet121(
    weights = "imagenet", # onceden egitilmis modelin agirliklari
    include_top = False, # son katmanlari dahil etme
    input_shape = (*IMG_SIZE, 3) # input boyutu (224, 224, 3)
)
base_model.trainable = False # base modeli dondur yani base model train edilmeyecek

x = base_model.output # base modelin ciktisi
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation = "relu")(x) # 128 nöronlu gizli katman
x = Dropout(0.5)(x) # dropout katmani
pred = Dense(1, activation = "sigmoid")(x) # 1 nöronlu cikti katmani (ikili siniflandirma)

model = Model(inputs = base_model.input, outputs = pred) # modeli tanimla

# modelin derlenmesi ve callback ayarlari


# modelin egitilmesi ve sonuclarin degerlendirilmesi