"""
Created on Thu Jun 28 14:50:38 2018
@author: Andrew Lopez - SOFWERX: Group 4 interns
"""
import pandas as pd
from matplotlib import style
style.use('ggplot')
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#parse data
filename = 'test_data.csv'
names = ['_id', 'Sensor Type', 'Sensor State', 'intrusion']
df = pd.read_csv(filename, names=names)
df = df[df['_id'] != '_id']

#Extract data features from '_id' to represent time
df['_id'] = pd.to_datetime(df['_id'])
df['hour'] = [d.hour for d in df['_id']]
df['minute'] = [d.minute for d in df['_id']]
df['month'] = [d.month for d in df['_id']]
df['day'] = [d.day for d in df['_id']]

#convert intrusion data to datatype int64
df = df[pd.notnull(df['intrusion'])] #remove NaN
df['intrusion'] = [int(d) for d in df['intrusion']]

#represent non-numeric data with binary elements
x = df['Sensor Type']
Y = df['Sensor State']
def binarize(Z):
    encoded = preprocessing.LabelEncoder()
    encoded.fit(Z)
    newEncode = encoded.transform(Z)
    newEncode = newEncode.reshape(-1, 1)
    #One Hot Encoder takes the enumerated data and turns it to binary
    OHE = preprocessing.OneHotEncoder(sparse=False)
    #print (OHE.fit_transform(newEncode))
    Z = OHE.fit_transform(newEncode)
    return Z
x = binarize(x)
Y = binarize(Y)

#put binarized data into DataFrames
columnsType=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
types = pd.DataFrame(x, columns=columnsType)
columnsState=['1', '2', '3', '4', '5', '6']
states = pd.DataFrame(Y, columns=columnsState)


#join all formatted data. Remove NaNs
df = df[['intrusion', 'month', 'day', 'hour', 'minute']]
df = df.join(types)
df = df.join(states)
df = df[pd.notnull(df['6'])]

#create traingning and testing data
X = df.iloc[:, 1:21].values #sets independent variables (time, Sensor Type, Sensor State)
y = df.iloc[:, 0:1].values #sets dependent variable (intrusion status)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) #creates training and testing data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#trains the K-Nearest Neighbors Algorithm
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
c = classifier.fit(X_train, y_train)


#tests the K-Nearest Neighbors Algorithm
y_pred = c.predict(X_test)
print()
print("Testing data point:")
print(X_test[1])
print("Prediction: " + str(y_pred[1]))
print("Actual: " + str(y_test[1]))
print()
cm = confusion_matrix(y_test, y_pred) #makes confusion matrix
result = c.score(X_test, y_test) #returns the accuracy of the algorithm
report = classification_report(y_test, y_pred) #makes a classification report
print("Classification Report:")
print(report)

def confusionMatrixString(cm): #function to return the confusion matrix as a readable string
    print("Confusion Matrix:")
    print("True Positive; False Positive; True Negative; False Negative")
    c = str(cm[0][0]) + "            " + str(cm[0][1]) + "              " + str(cm[1][1]) + "             " + str(cm[1][0])
    return c

def accuracyString(r): #function to return the accuracy as a redable string
    r = r*100
    r = round(r, 3)
    a = "Accuracy: " + str(r)
    return a

print(confusionMatrixString(cm))
print()
print(accuracyString(result))
