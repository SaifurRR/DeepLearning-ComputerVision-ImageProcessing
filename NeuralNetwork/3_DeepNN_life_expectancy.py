#import libraries
import pandas as pd #load data
from sklearn.model_selection import train_test_split #train test split
from sklearn.compose import ColumnTransformer # Scaling
from sklearn.preprocessing import StandardScaler, Normalizer# scaling type
from tensorflow.keras.models import Sequential #Sequential NN model
from tensorflow.keras.layers import InputLayer, Dense #Input Layer
from tensorflow.keras.optimizers import Adam #optimizer

#load data and inspect:

# check dataset - stsummary
dataset=pd.read_csv('3_life_expectancy.csv')
#print(dataset.head())
# drop column: axis=1
dataset=dataset.drop(['Country'], axis=1)
# labels
labels=dataset.iloc[:,-1]
# features
features=dataset.iloc[:, 0:-1]

#data Pre-processing:

# one hot encoding (convert categorical column)
features=pd.get_dummies(features)
pd.set_option('display.max_columns', None)
# train_test_split
features_train, features_test, labels_train, labels_test=train_test_split(features,labels, test_size=0.25, random_state=42)
# numerical featuress
numerical_features=features.select_dtypes(include=['int64', 'float64'])
# numerical columns
numerical_columns=numerical_features.columns
# scaling: StandardScaler
ct=ColumnTransformer([('only numeric', StandardScaler(),list(numerical_columns))], remainder='passthrough')
# fit-transform: training data
features_train_scaled=ct.fit_transform(features_train)
# transform: test data
features_test_scaled=ct.transform(features_test)

#Build Model:
my_model=Sequential()
#input layer
input=InputLayer(input_shape=(features.shape[1],)) # input_shape:(features, #samples)
# add inut layer
my_model.add(input)
# hidden layer
my_model.add(Dense(64, activation='relu')) #64 hidden units, activation: 'relu'
# o/p layer
my_model.add(Dense(1)) #o/p: 1 neuron (for regression)
#print(my_model.summary())

# Initializing Optimizer and Compiling the Model:
opt=Adam(learning_rate=0.01) # Adam optimizer
my_model.compile(loss='mse', metrics=['mae'], optimizer=opt) # compile method : config. method for training, 'mae' will be monitored as performance metrics
#print(my_model.summary())

#Fit and evaluate the model
my_model.fit(features_train_scaled,labels_train,epochs=40,batch_size=1)  #fit
res_mse, res_mae=my_model.evaluate(features_test_scaled,labels_test) #evaluate
print(f'mse: {res_mse}', f'mae: {res_mae}')





