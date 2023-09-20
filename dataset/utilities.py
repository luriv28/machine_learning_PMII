
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import tensorflow as tf
import os 
from skimage import io
from PIL import Image
from tensorflow.keras import backend as K
  
#creando un generador de datos personalizado:

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, ids , mask, image_dir = './', batch_size = 16, img_h = 256, img_w = 256, shuffle = True):

    self.ids = ids
    self.mask = mask
    self.image_dir = image_dir
    self.batch_size = batch_size
    self.img_h = img_h
    self.img_w = img_w
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Get the number of batches per epoch'

    return int(np.floor(len(self.ids)) / self.batch_size)

  def __getitem__(self, index):
    'Generate a batch of data'

    #generar índice de longitud de tamaño de lote
    indexes = self.indexes[index* self.batch_size : (index+1) * self.batch_size]

    #obtener el ImageId correspondiente a los índices creados anteriormente según el tamaño del lote
    list_ids = [self.ids[i] for i in indexes]

    #obtener el MaskId correspondiente a los índices creados anteriormente según el tamaño del lote
    list_mask = [self.mask[i] for i in indexes]

    #generar datos para X (características) e y (etiqueta)
    X, y = self.__data_generation(list_ids, list_mask)

    #retornar los datos
    return X, y

  def on_epoch_end(self):
    'Se utiliza para actualizar los índices después de cada época, tanto al principio como al final de cada época.'
    
    #obtener la matriz de índices según el marco de datos de entrada  
    self.indexes = np.arange(len(self.ids))

    #si la mezcla aleatoria es verdadera, mezcla los índices 
    if self.shuffle:
      np.random.shuffle(self.indexes)

  def __data_generation(self, list_ids, list_mask):
    'generate the data corresponding the indexes in a given batch of images'

    # crear matrices vacías de formas (batch_size,height,width,depth) 
    #La profundidad es 3 para la entrada y la profundidad se toma como 1 para la salida porque la máscara consta solo de 1 canal.
    X = np.empty((self.batch_size, self.img_h, self.img_w, 3))
    y = np.empty((self.batch_size, self.img_h, self.img_w, 1))

    #iterar a través de las filas del marco de datos, cuyo tamaño es igual al tamaño del lote
    for i in range(len(list_ids)):
      #path de la imagen
      img_path = './' + str(list_ids[i])
      
      #path de la mascara
      mask_path = './' + str(list_mask[i])
      
      #leer la imagen original y la imagen de máscara correspondiente
      img = io.imread(img_path)
      mask = io.imread(mask_path)

      #Cambiar el tamaño y encubrirlas a la matriz de tipo float64
      img = cv2.resize(img,(self.img_h,self.img_w))
      img = np.array(img, dtype = np.float64)
      
      mask = cv2.resize(mask,(self.img_h,self.img_w))
      mask = np.array(mask, dtype = np.float64)

      #estandarizando 
      img -= img.mean()
      img /= img.std()
      
      mask -= mask.mean()
      mask /= mask.std()
      
      #Agregar imagen a la matriz vacía
      X[i,] = img
      
      #ampliando la dimensión de la imagen desde (256,256) a (256,256,1)
      y[i,] = np.expand_dims(mask, axis = 2)
    
    #normalizar y
    y = (y > 0).astype(int)

    return X, y






def prediction(test, model, model_seg):
  '''
  Función de predicción que toma el marco de datos que contiene ImageID como entrada y realiza 2 tipos de predicción en la imagen
  Inicialmente, la imagen pasa a través de la red de clasificación que predice si la imagen tiene defectos o no, si el modelo
  está 99% seguro de que la imagen no tiene ningún defecto, entonces la imagen se etiqueta como sin defectos, si el modelo no está seguro, pasa la imagen al
  red de segmentación, nuevamente verifica si la imagen tiene defecto o no, si tiene defecto, luego se encuentra el tipo y la ubicación del defecto
  '''

  #directorio
  directory = os.path.abspath('dataset')

  #Creando una lista vacía para almacenar los resultados
  mask = []
  image_id = []
  has_mask = []

  #iterando a través de cada imagen en los datos de prueba
  for i in test.image_path:

    path = directory + str(i)
    path = os.path.join(directory, str(i))

    #leer la imagen
    img = io.imread(path)

    #Normalizando la imagen
    img = img * 1./255.

    #Remodelando la imagen
    img = cv2.resize(img,(256,256))

    #Convertir la imagen en una matriz
    img = np.array(img, dtype = np.float64)
    
    #remodelando la imagen de256,256,3 a 1,256,256,3
    img = np.reshape(img, (1,256,256,3))

    #haciendo predicciones sobre la imagen
    is_defect = model.predict(img)

    #Si el tumor no está presente, agregamos los detalles de la imagen a la lista.
    if np.argmax(is_defect) == 0:
      image_id.append(i)
      has_mask.append(0)
      mask.append('No mask')
      continue

    #Leer la imagen
    img = io.imread(path)

    #Creando una matriz vacía de formas 1,256,256,1
    X = np.empty((1, 256, 256, 3))

    #Cambiar el tamaño de la imagen y encubrirlas a matriz de tipo float64
    img = cv2.resize(img,(256,256))
    img = np.array(img, dtype = np.float64)

    #Estandarizando la imagen
    img -= img.mean()
    img /= img.std()

    #convertir la forma de la imagen de 256,256,3 a 1,256,256,3
    X[0,] = img

    #hacer predicción
    predict = model_seg.predict(X)

    #si la suma de los valores predichos es igual a 0 entonces no hay tumor
    if predict.round().astype(int).sum() == 0:
        image_id.append(i)
        has_mask.append(0)
        mask.append('No mask')
    else:
    #Si la suma de los valores de los píxeles es mayor que 0, entonces hay un tumor.
        image_id.append(i)
        has_mask.append(1)
        mask.append(predict)


  return image_id, mask, has_mask
        




'''
Necesitamos una función de pérdida personalizada para entrenar este ResUNet. 
Entonces, hemos usado la función de pérdida tal como está en https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py

@article{focal-unet,
  title={Una nueva función de pérdida focal de Tversky con Atención U-Net mejorada para la segmentación de lesiones},
  author={Abraham, Nabila and Khan, Naimul Mefraz},
  journal={arXiv preprint arXiv:1810.07842},
  year={2018}
}

'''
def tversky(y_true, y_pred, smooth = 1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)