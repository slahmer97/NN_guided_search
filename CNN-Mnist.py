#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# In[ ]:


from tensorflow.keras.datasets import mnist

# In[64]:


(X_train,y_train),(X_test,y_test)=mnist.load_data()

# In[65]:


plt.imshow(X_train[0],cmap='Greys')

# In[66]:


y_train[0]

# In[67]:


from tensorflow.keras.utils import to_categorical

# In[68]:


y_train.shape

# In[71]:


y_cat_train=to_categorical(y_train,num_classes=10)

# In[72]:


y_cat_test=to_categorical(y_test,num_classes=10)

# In[73]:


y_cat_train[0]

# In[74]:


X_train=X_train/250

# In[75]:


X_test=X_test/250

# In[78]:


#batch_size,height,width,color_channels
X_train=X_train.reshape(60000,28,28,1)

# In[79]:


X_test=X_test.reshape(10000,28,28,1)

# In[82]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten

# In[87]:


model=Sequential()
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))

#output MultiClass ==> SoftMax 
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# In[88]:


from tensorflow.keras.callbacks import EarlyStopping

# In[89]:


early_stop=EarlyStopping(monitor='val_loss',patience=1)

# In[90]:


model.fit(x=X_train,y=y_cat_train,epochs=10,validation_data=(X_test,y_cat_test),callbacks=[early_stop])

# In[99]:


accuracy=pd.DataFrame(model.history.history)[['accuracy','val_accuracy']]

# In[104]:


losses=pd.DataFrame(model.history.history)[['loss','val_loss']]

# In[105]:


accuracy.plot()

# In[106]:


losses.plot()

# In[109]:


model.metrics_names

# In[111]:


model.evaluate(X_test,y_cat_test,verbose=0)

# In[112]:


from sklearn.metrics import confusion_matrix,classification_report

# In[113]:


predictions=model.predict_classes(X_test)

# In[114]:


print(classification_report(y_test,predictions))

# In[115]:


confusion_matrix(y_test,predictions)

# In[117]:


import seaborn as sns;

# In[118]:


sns.heatmap(confusion_matrix(y_test,predictions),annot=True)

# In[123]:


model.predict_classes(X_train[0].reshape(1,28,28,1))
