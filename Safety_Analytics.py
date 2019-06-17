
# coding: utf-8

# In[ ]:


# Importing all the datasets for features
import pandas as pd
df_0 = pd.read_csv('part-00000-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
df_1 = pd.read_csv('part-00001-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
df_2 = pd.read_csv('part-00002-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
df_3 = pd.read_csv('part-00003-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
df_4 = pd.read_csv('part-00004-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
df_5 = pd.read_csv('part-00005-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
df_6 = pd.read_csv('part-00006-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
df_7 = pd.read_csv('part-00007-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
df_8 = pd.read_csv('part-00008-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')
df_9 = pd.read_csv('part-00009-e6120af0-10c2-4248-97c4-81baf4304e5c-c000.csv')

# Concat all the dataframes
frames = [df_0,df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9]
result = pd.concat(frames)

# Standarization of all the variables
result['Accuracy'] = ((result['Accuracy'])-min(result['Accuracy']))/(max(result['Accuracy'])-min(result['Accuracy']))
result['Bearing'] = ((result['Bearing'])-min(result['Bearing']))/(max(result['Bearing'])-min(result['Bearing']))
result['acceleration_x'] = ((result['acceleration_x'])-min(result['acceleration_x']))/(max(result['acceleration_x'])-min(result['acceleration_x']))
result['acceleration_y'] = ((result['acceleration_y'])-min(result['acceleration_y']))/(max(result['acceleration_y'])-min(result['acceleration_y']))
result['acceleration_z'] = ((result['acceleration_z'])-min(result['acceleration_z']))/(max(result['acceleration_z'])-min(result['acceleration_z']))
result['gyro_x'] = ((result['gyro_x'])-min(result['gyro_x']))/(max(result['gyro_x'])-min(result['gyro_x']))
result['gyro_y'] = ((result['gyro_y'])-min(result['gyro_y']))/(max(result['gyro_y'])-min(result['gyro_y']))
result['gyro_z'] = ((result['gyro_z'])-min(result['gyro_z']))/(max(result['gyro_z'])-min(result['gyro_z']))
result['second'] = ((result['second'])-min(result['second']))/(max(result['second'])-min(result['second']))
result['Speed'] = ((result['Speed'])-min(result['Speed']))/(max(result['Speed'])-min(result['Speed']))

# Creating a new dataset that has a unique value of all the features for a unique BookingID by taking the mean of all the trips
# for a distinct BookingID

len_ = len(result['bookingID'].unique())
result_1 = pd.DataFrame()
result_1['bookingID'] = [0]*len_
result_1['Accuracy'] = [float(0)]*len_
result_1['Bearing'] = [float(0)]*len_
result_1['acceleration_x'] = [float(0)]*len_
result_1['acceleration_y'] = [float(0)]*len_
result_1['acceleration_z'] = [float(0)]*len_
result_1['gyro_x'] = [float(0)]*len_
result_1['gyro_y'] = [float(0)]*len_
result_1['gyro_z'] = [float(0)]*len_
result_1['second'] = [float(0)]*len_
result_1['Speed'] = [float(0)]*len_
for i in range(len_):
    id_ = result['bookingID'].unique()[i]
    result_1['bookingID'][i] = id_
    result_1['Accuracy'][i] = result[result['bookingID']==id_]['Accuracy'].mean()
    result_1['Bearing'][i] = result[result['bookingID']==id_]['Bearing'].mean()
    result_1['acceleration_x'][i] = result[result['bookingID']==id_]['acceleration_x'].mean()
    result_1['acceleration_y'][i] = result[result['bookingID']==id_]['acceleration_y'].mean()
    result_1['acceleration_z'][i] = result[result['bookingID']==id_]['acceleration_z'].mean()
    result_1['gyro_x'][i] = result[result['bookingID']==id_]['gyro_x'].mean()
    result_1['gyro_y'][i] = result[result['bookingID']==id_]['gyro_y'].mean()
    result_1['gyro_z'][i] = result[result['bookingID']==id_]['gyro_z'].mean()
    result_1['second'][i] = result[result['bookingID']==id_]['second'].mean()
    result_1['Speed'][i] = result[result['bookingID']==id_]['Speed'].mean()
    
# Merging the label with each BookingID
df_lebel = pd.read_csv('part-00000-e9445087-aa0a-433b-a7f6-7f4c19d78ad6-c000.csv')
df_new = pd.merge(result_1,df_lebel, on = 'bookingID')

# Feature Engineering: Making the new features by using the above features

#Net Acceleration^2
df_new['net_accel'] = df_new['acceleration_x']*df_new['acceleration_x'] + df_new['acceleration_y']*df_new['acceleration_y'] +df_new['acceleration_z']*df_new['acceleration_z']

#Net Gyro^2
df_new['net_gyro'] = df_new['gyro_x']*df_new['gyro_x'] + df_new['gyro_y']*df_new['gyro_y'] +df_new['gyro_z']*df_new['gyro_z']

# Other Variables
df_new['acc*bea'] = df_new['Accuracy']*df_new['Bearing']
df_new['acc/bea'] = df_new['Accuracy']/df_new['Bearing']

df_new['acc*sp'] = df_new['Accuracy']*df_new['Speed']
df_new['acc/sp'] = df_new['Accuracy']/df_new['Speed']

df_new['acc*acc_x'] = df_new['Accuracy']*df_new['acceleration_x']
df_new['acc/acc_x'] = df_new['Accuracy']/df_new['acceleration_x']

df_new['acc*acc_y'] = df_new['Accuracy']*df_new['acceleration_y']
df_new['acc/acc_y'] = df_new['Accuracy']/df_new['acceleration_y']

df_new['acc*acc_z'] = df_new['Accuracy']*df_new['acceleration_z']
df_new['acc/acc_z'] = df_new['Accuracy']/df_new['acceleration_z']

df_new['acc*gy_x'] = df_new['Accuracy']*df_new['gyro_x']
df_new['acc/gy_x'] = df_new['Accuracy']/df_new['gyro_x']

df_new['acc*gy_y'] = df_new['Accuracy']*df_new['gyro_y']
df_new['acc/gy_y'] = df_new['Accuracy']/df_new['gyro_y']

df_new['acc*gy_z'] = df_new['Accuracy']*df_new['gyro_z']
df_new['acc/gy_z'] = df_new['Accuracy']/df_new['gyro_z']

df_new['acc*gyro_net'] = df_new['Accuracy']*df_new['gyro_net']
df_new['acc/gyro_net'] = df_new['Accuracy']/df_new['gyro_net']

df_new['acc*net_accel'] = df_new['Accuracy']*df_new['net_accel']
df_new['acc/net_accel'] = df_new['Accuracy']/df_new['net_accel']

df_new['acc*second'] = df_new['Accuracy']*df_new['second']
df_new['acc/second'] = df_new['Accuracy']/df_new['second']

df_new['bea*acc_x'] = df_new['Bearing']*df_new['acceleration_x']
df_new['bea/acc_x'] = df_new['Bearing']/df_new['acceleration_x']

df_new['bea*acc_y'] = df_new['Bearing']*df_new['acceleration_y']
df_new['bea/acc_y'] = df_new['Bearing']/df_new['acceleration_y']

df_new['bea*acc_z'] = df_new['Bearing']*df_new['acceleration_z']
df_new['bea/acc_z'] = df_new['Bearing']/df_new['acceleration_z']

df_new['bea*gy_x'] = df_new['Bearing']*df_new['gyro_x']
df_new['bea/gy_x'] = df_new['Bearing']/df_new['gyro_x']

df_new['bea*gy_y'] = df_new['Bearing']*df_new['gyro_y']
df_new['bea/gy_y'] = df_new['Bearing']/df_new['gyro_y']

df_new['bea*gy_z'] = df_new['Bearing']*df_new['gyro_z']
df_new['bea/gy_z'] = df_new['Bearing']/df_new['gyro_z']

df_new['bea*net_accel'] = df_new['Bearing']*df_new['net_accel']
df_new['bea/net_accel'] = df_new['Bearing']/df_new['net_accel']

df_new['bea*sec'] = df_new['Bearing']*df_new['second']
df_new['bea/sec'] = df_new['Bearing']/df_new['second']

df_new['acc_x*acc_y'] = df_new['acceleration_x']*df_new['acceleration_y']
df_new['acc_x/acc_y'] = df_new['acceleration_x']/df_new['acceleration_y']

df_new['acc_x*acc_z'] = df_new['acceleration_x']*df_new['acceleration_z']
df_new['acc_x/acc_z'] = df_new['acceleration_x']/df_new['acceleration_z']

df_new['acc_y*acc_z'] = df_new['acceleration_y']*df_new['acceleration_z']
df_new['acc_y/acc_z'] = df_new['acceleration_y']/df_new['acceleration_z']

df_new['acc_x*gy_x'] = df_new['acceleration_x']*df_new['gyro_x']
df_new['acc_x/gy_x'] = df_new['acceleration_x']/df_new['gyro_x']

df_new['acc_x*gy_y'] = df_new['acceleration_x']*df_new['gyro_y']
df_new['acc_x/gy_y'] = df_new['acceleration_x']/df_new['gyro_y']

df_new['acc_x*gy_z'] = df_new['acceleration_x']*df_new['gyro_z']
df_new['acc_x/gy_z'] = df_new['acceleration_x']/df_new['gyro_z']

df_new['acc_x*sec'] = df_new['acceleration_x']*df_new['second']
df_new['acc_x/sec'] = df_new['acceleration_x']/df_new['second']

df_new['acc_y*gy_x'] = df_new['acceleration_y']*df_new['gyro_x']
df_new['acc_y/gy_x'] = df_new['acceleration_y']/df_new['gyro_x']

df_new['acc_y*gy_y'] = df_new['acceleration_y']*df_new['gyro_y']
df_new['acc_y/gy_y'] = df_new['acceleration_y']/df_new['gyro_y']

df_new['acc_y*gy_z'] = df_new['acceleration_y']*df_new['gyro_z']
df_new['acc_y/gy_z'] = df_new['acceleration_y']/df_new['gyro_z']

df_new['acc_y*sec'] = df_new['acceleration_y']*df_new['second']
df_new['acc_y/sec'] = df_new['acceleration_y']/df_new['second']

df_new['acc_z*gy_x'] = df_new['acceleration_z']*df_new['gyro_x']
df_new['acc_z/gy_x'] = df_new['acceleration_z']/df_new['gyro_x']

df_new['acc_z*gy_y'] = df_new['acceleration_z']*df_new['gyro_y']
df_new['acc_z/gy_y'] = df_new['acceleration_z']/df_new['gyro_y']

df_new['acc_z*gy_z'] = df_new['acceleration_z']*df_new['gyro_z']
df_new['acc_z/gy_z'] = df_new['acceleration_z']/df_new['gyro_z']

df_new['acc_z*sec'] = df_new['acceleration_z']*df_new['second']
df_new['acc_z/sec'] = df_new['acceleration_z']/df_new['second']

df_new['gy_z*gy_y'] = df_new['acceleration_z']*df_new['gyro_y']
df_new['acc_z/gy_y'] = df_new['acceleration_z']/df_new['gyro_y']

df_new['acc_z*gy_z'] = df_new['acceleration_z']*df_new['gyro_z']
df_new['acc_z/gy_z'] = df_new['acceleration_z']/df_new['gyro_z']

df_new['acc_z*sec'] = df_new['acceleration_z']*df_new['second']
df_new['acc_z/sec'] = df_new['acceleration_z']/df_new['second']

df_new['gy_x*gy_y'] = df_new['gyro_x']*df_new['gyro_y']
df_new['gy_x/gy_y'] = df_new['gyro_x']/df_new['gyro_y']

df_new['gy_x*gy_z'] = df_new['gyro_x']*df_new['gyro_z']
df_new['gy_x/gy_z'] = df_new['gyro_x']/df_new['gyro_z']

df_new['gy_x*net_accel'] = df_new['gyro_x']*df_new['net_accel']
df_new['gy_x/net_accel'] = df_new['gyro_x']/df_new['net_accel']

df_new['gy_x*sec'] = df_new['gyro_x']*df_new['second']
df_new['gy_x/sec'] = df_new['gyro_x']/df_new['second']

df_new['gy_y*gy_z'] = df_new['gyro_y']*df_new['gyro_z']
df_new['gy_y/gy_z'] = df_new['gyro_y']/df_new['gyro_z']

df_new['gy_y*net_accel'] = df_new['gyro_y']*df_new['net_accel']
df_new['gy_y/net_accel'] = df_new['gyro_y']/df_new['net_accel']

df_new['gy_y*sec'] = df_new['gyro_y']*df_new['second']
df_new['gy_y/sec'] = df_new['gyro_y']/df_new['second']

df_new['gy_z*net_accel'] = df_new['gyro_z']*df_new['net_accel']
df_new['gy_z/net_accel'] = df_new['gyro_z']/df_new['net_accel']

df_new['gy_z*sec'] = df_new['gyro_z']*df_new['second']
df_new['gy_z/sec'] = df_new['gyro_z']/df_new['second']

df_new['gyro_net*net_accel'] = df_new['gyro_net']*df_new['net_accel']
df_new['gyro_net/net_accel'] = df_new['gyro_net']/df_new['net_accel']

df_new['gyro_net*second'] = df_new['gyro_net']*df_new['second']
df_new['gyro_net/second'] = df_new['gyro_net']/df_new['second']

df_new['gyro_net*accuracy'] = df_new['gyro_net']*df_new['Accuracy']
df_new['gyro_net/accuracy'] = df_new['gyro_net']/df_new['Accuracy']

df_new['gyro_net*bearing'] = df_new['gyro_net']*df_new['Bearing']
df_new['gyro_net/bearing'] = df_new['gyro_net']/df_new['Bearing']

df_new['net_accel*second'] = df_new['net_accel']*df_new['second']
df_new['net_accel/second'] = df_new['net_accel']/df_new['second']

# Getting the Feature Importance of all the features(given or constructed)
X = df_new.drop(['bookingID','label'],axis=1)
Y = df_new['label']
dtrain = xgb.DMatrix(X, label=Y)
watchlist = [(dtrain, 'train')]
param = {'max_depth': 6, 'learning_rate': 0.03}
num_round = 200
bst = xgb.train(param, dtrain, num_round, watchlist)
bst.get_score(importance_type='gain')

# Sorting the variables according to their importance
import operator
sorted_x = sorted(bst.get_score(importance_type='gain').items(), key=operator.itemgetter(1))

# Taking the top 15 variables for modelling
X = df_new[[sorted_x[-1][0],sorted_x[-2][0],sorted_x[-2][0],sorted_x[-3][0],sorted_x[-4][0],sorted_x[-5][0],sorted_x[-6][0],sorted_x[-7][0],
      sorted_x[-8][0],sorted_x[-9][0],sorted_x[-10][0],sorted_x[-11][0],sorted_x[-12][0],sorted_x[-13][0],sorted_x[-14][0],
      sorted_x[-15][0]]]

Y = df_new['label']

# Splitting the data for training and testing
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 40)

# Using the Light GBM to train the data using specific parameters
import lightgbm as lgb
d_train = lgb.Dataset(x_train, label=y_train)
params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 70
params['max_depth'] = 10
clf = lgb.train(params, d_train, 110)

# Predicting the values
y_pred=clf.predict(x_test)
#convert into binary values
for i in range(len(y_pred)):
    if y_pred[i]>=.3:       # setting threshold to .3
        y_pred[i]=1
    else:  
        y_pred[i]=0
        
# Creating an Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Getting the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)

#Roc_Auc Score
from sklearn.metrics import roc_auc_score
roc_auc_score(y_pred, y_test)

