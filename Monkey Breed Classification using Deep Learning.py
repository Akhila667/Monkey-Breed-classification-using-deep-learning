#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf  #tf >2.0
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


tf.__version__


# In[3]:


tf.test.is_gpu_available()


# In[ ]:





# In[7]:


train_datagen= ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                    zoom_range=0.2,
                                  vertical_flip=True,
                                  horizontal_flip=True)


# In[9]:


# Training Set
training_set = train_datagen.flow_from_directory(r'D:\Python37\Projects\Monkey Breed Classification using Deep Learning\train',
                                                 target_size=(224,224),class_mode='categorical',batch_size=16
                                         )


# In[10]:


# Test Set
test_datagen= ImageDataGenerator(rescale=1./255)

test_set= test_datagen.flow_from_directory(r'D:\Python37\Projects\Monkey Breed Classification using Deep Learning\validation',
                                          target_size=(224,224),class_mode='categorical',batch_size=16)


# In[ ]:





# # Lets make CNN

# In[14]:


#Initalising the CNN
cnn=tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32,padding="same",kernel_size=3,activation='relu',input_shape=[224,224,3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,padding="same",kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))
cnn.add(tf.keras.layers.Dense(units=10,activation='softmax'))



# In[15]:


cnn.summary()


# In[ ]:





# # Lets Train

# In[20]:


cnn.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=cnn.fit(x = training_set, validation_data = test_set, epochs = 10)


# In[21]:


history.model.save(r'D:\Python37\Projects\Monkey Breed Classification using Deep Learning\models\model-10.h5')


# In[ ]:





# In[22]:


model=tf.keras.models.load_model(r'D:\Python37\Projects\Monkey Breed Classification using Deep Learning\models\model-10.h5')


# In[23]:


model.summary()


# In[ ]:





# In[27]:


import matplotlib.pyplot as plt
acc_train=history.history['accuracy']
acc_val=history.history['val_accuracy']
epochs=range(1,11)
plt.plot(epochs,acc_train,'g',label='Training Accuracy')
plt.plot(epochs,acc_val,'r',label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()


# In[28]:


import matplotlib.pyplot as plt
loss_train=history.history['loss']
loss_val=history.history['val_loss']
epochs=range(1,11)
plt.plot(epochs,loss_train,'g',label='Training loss')
plt.plot(epochs,loss_val,'r',label='Validation loss')
plt.title("Training vs Validation loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.show()


# In[ ]:





# # Testing

# In[50]:


import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing import image


# In[51]:


txt=pd.read_csv(r'D:\Python37\Projects\Monkey Breed Classification using Deep Learning\monkey_labels.txt')


# In[52]:


txt


# In[74]:


test_image=r'D:\Python37\Projects\Monkey Breed Classification using Deep Learning\train\n5\n5022.jpg'
open_image=Image.open(test_image)
test_image=image.load_img(test_image,target_size=(224,224))
test_image=image.img_to_array(test_image)
test_image=test_image/255
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
result=np.argmax(result)
Name=txt.iloc[result]
Name=Name.iloc[2]
plt.imshow(open_image)
plt.title(Name)
plt.show()


# In[ ]:




