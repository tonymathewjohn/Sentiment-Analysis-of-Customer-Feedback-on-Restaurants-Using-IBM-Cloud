
#import Libraries

import numpy as np
import pandas as pd

#First import the dataset in the dataset variable
#data_review variable contains the reveiws and ratings 
dataset = pd.read_csv("zomato.csv")
data_review = dataset['reviews_list']

print(dataset['reviews_list'][0])

x = []
y = []

# here we tokenize the rating string and the review string
#loop over all the rows
for row_num in range(0,51717):
    # split the revie text at '()
    lst = data_review[row_num].split("('")
    for i in lst:
        if len(i) > 5:
            if i.find("',") != -1:
                single_rev = i.split("',")
                if len(single_rev[0]) > 2:
                    x.append(single_rev[0])
                if len(single_rev[1]) > 2:    
                    y.append(single_rev[1])


#Import the Libraries
import re
import nltk
nltk.download("stopwords") 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# to store the final rating
rating_final = []
# to store cleaned revies
review_final = []

# the rating string contains words and numbers 
# so we tokenize the numbers only from it and change into float
# for rartings below 2.5 we store the rating as poor
# for ratings between 2.5 and 3.5 the rating as average
# for ratings more than 3.5 tha rating stored as good

for loop in range(0,40000):
    data_x = x[loop]
    data_x = re.sub('[a-zA-Z]', " ", data_x)
    data_x = data_x.split()
    data_x = ''.join(data_x)
    data_x = float(data_x)
    if data_x < 2.5:
        rating_final.append("poor") #poor
    elif data_x >= 2.5 and data_x <= 3.5 :    
        rating_final.append("average") # average
    elif data_x > 3.5:
        rating_final.append("good") #good
        
# label encode the Ratings and OneHotEncode for the classification        

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
rating_final = le.fit_transform(rating_final)

rating_final = np.array(rating_final)
rating_final = np.expand_dims(rating_final, axis=1)
        
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
rates = one.fit_transform(rating_final).toarray()


# Here the unnecessary stop words from the reviews lists
# Stemming operations are also done here 

for loop in range(0,40000) : 
    data_y = y[loop]
    data_y = re.sub('[^a-zA-Z]', " ", data_y)
    data_y = data_y.lower()
    data_y = data_y.split()
    data_y = [ps.stem(word) for word in data_y if not word in set(stopwords.words('english'))]
    data_y = ' '.join(data_y)
    review_final.append(data_y)
    
# count vectorize the reviews according to the unique words    
                    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 20000)
x_final = cv.fit_transform(review_final).toarray()


# saving the vectorizer which would be used as dictionary.
import pickle
pickle.dump(cv, open('cv.pkl','wb'))

# Split the data into test and train sets

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final,rates, test_size = 0.2, random_state = 0)    

# adding the neuron layers 
# the units in the input layers is equal to the number of unique words
# taken three deeper layers of 2000 units each
# Relu as activation in the hidden layers
# the output layer has 3 units as the one hot encoding has 3 columns
# the classification is in categorical

from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(units = 13264, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 2000, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 2000, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 2000, init = 'random_uniform', activation = 'relu'))
model.add(Dense(units = 3, init = 'random_uniform', activation = 'softmax'))
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = 128,epochs = 200)

# testing the prediction

y_pred = model.predict(x_test)

text =  "The food is okay. average place "
text = re.sub('[^a-zA-Z]', ' ',text)
text = text.lower()
text = text.split()
text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
text = ' '.join(text)

y_p = model.predict(cv.transform([text]))

# saving the model

model.save("zomato_2_analysis.h5")
