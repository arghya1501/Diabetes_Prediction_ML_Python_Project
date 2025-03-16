```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

## Data Collection And Analysis

PIMA Diabetes Dataset


```python
# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv(r"E:\Project work 2\Machine Larning\Diabetes Prediction\diabetes.csv")
```


```python
# printing the first 5 rows of the dataset
diabetes_dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# number of rows and Columns in this dataset
diabetes_dataset.shape
```




    (768, 9)




```python
# getting the sttistical measures of the data
diabetes_dataset.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
      <td>768.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.845052</td>
      <td>120.894531</td>
      <td>69.105469</td>
      <td>20.536458</td>
      <td>79.799479</td>
      <td>31.992578</td>
      <td>0.471876</td>
      <td>33.240885</td>
      <td>0.348958</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.369578</td>
      <td>31.972618</td>
      <td>19.355807</td>
      <td>15.952218</td>
      <td>115.244002</td>
      <td>7.884160</td>
      <td>0.331329</td>
      <td>11.760232</td>
      <td>0.476951</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.078000</td>
      <td>21.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>99.000000</td>
      <td>62.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.300000</td>
      <td>0.243750</td>
      <td>24.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>117.000000</td>
      <td>72.000000</td>
      <td>23.000000</td>
      <td>30.500000</td>
      <td>32.000000</td>
      <td>0.372500</td>
      <td>29.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.000000</td>
      <td>140.250000</td>
      <td>80.000000</td>
      <td>32.000000</td>
      <td>127.250000</td>
      <td>36.600000</td>
      <td>0.626250</td>
      <td>41.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17.000000</td>
      <td>199.000000</td>
      <td>122.000000</td>
      <td>99.000000</td>
      <td>846.000000</td>
      <td>67.100000</td>
      <td>2.420000</td>
      <td>81.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
diabetes_dataset['Outcome'].value_counts()
```




    Outcome
    0    500
    1    268
    Name: count, dtype: int64



0 --> Non-Diabetic
1 --> Diabetic


```python
diabetes_dataset.groupby('Outcome').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
    </tr>
    <tr>
      <th>Outcome</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.298000</td>
      <td>109.980000</td>
      <td>68.184000</td>
      <td>19.664000</td>
      <td>68.792000</td>
      <td>30.304200</td>
      <td>0.429734</td>
      <td>31.190000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.865672</td>
      <td>141.257463</td>
      <td>70.824627</td>
      <td>22.164179</td>
      <td>100.335821</td>
      <td>35.142537</td>
      <td>0.550500</td>
      <td>37.067164</td>
    </tr>
  </tbody>
</table>
</div>




```python
# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome']
```


```python
print(X)
```

         Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    0              6      148             72             35        0  33.6   
    1              1       85             66             29        0  26.6   
    2              8      183             64              0        0  23.3   
    3              1       89             66             23       94  28.1   
    4              0      137             40             35      168  43.1   
    ..           ...      ...            ...            ...      ...   ...   
    763           10      101             76             48      180  32.9   
    764            2      122             70             27        0  36.8   
    765            5      121             72             23      112  26.2   
    766            1      126             60              0        0  30.1   
    767            1       93             70             31        0  30.4   
    
         DiabetesPedigreeFunction  Age  
    0                       0.627   50  
    1                       0.351   31  
    2                       0.672   32  
    3                       0.167   21  
    4                       2.288   33  
    ..                        ...  ...  
    763                     0.171   63  
    764                     0.340   27  
    765                     0.245   30  
    766                     0.349   47  
    767                     0.315   23  
    
    [768 rows x 8 columns]
    


```python
print(Y)
```

    0      1
    1      0
    2      1
    3      0
    4      1
          ..
    763    0
    764    0
    765    0
    766    1
    767    0
    Name: Outcome, Length: 768, dtype: int64
    

Data Standardization


```python
scaler = StandardScaler()
```


```python
scaler.fit(X)
```






```python
standardized_data = scaler.transform(X)
```


```python
print(standardized_data)
```

    [[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
       1.4259954 ]
     [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
      -0.19067191]
     [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
      -0.10558415]
     ...
     [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
      -0.27575966]
     [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
       1.17073215]
     [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
      -0.87137393]]
    


```python
X=standardized_data
Y=diabetes_dataset['Outcome']
```


```python
print(X)
print(Y)
```

    [[ 0.63994726  0.84832379  0.14964075 ...  0.20401277  0.46849198
       1.4259954 ]
     [-0.84488505 -1.12339636 -0.16054575 ... -0.68442195 -0.36506078
      -0.19067191]
     [ 1.23388019  1.94372388 -0.26394125 ... -1.10325546  0.60439732
      -0.10558415]
     ...
     [ 0.3429808   0.00330087  0.14964075 ... -0.73518964 -0.68519336
      -0.27575966]
     [-0.84488505  0.1597866  -0.47073225 ... -0.24020459 -0.37110101
       1.17073215]
     [-0.84488505 -0.8730192   0.04624525 ... -0.20212881 -0.47378505
      -0.87137393]]
    0      1
    1      0
    2      1
    3      0
    4      1
          ..
    763    0
    764    0
    765    0
    766    1
    767    0
    Name: Outcome, Length: 768, dtype: int64
    


```python
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify = Y, random_state=2)
```


```python
print(X.shape, X_train.shape, X_test.shape)
```

    (768, 8) (614, 8) (154, 8)
    

Training the Model


```python
classifier = svm.SVC(kernel = 'linear')
```


```python
#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)
```







## Model Evaluation

Accuracy Score


```python
# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
```


```python
print('Accuracy score of the training data :',training_data_accuracy)
```

    Accuracy score of the training data : 0.7866449511400652
    


```python
# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```


```python
print('Accuracy score of the test data :',test_data_accuracy)
```

    Accuracy score of the test data : 0.7727272727272727
    

Making a Predictive System


```python
import_data =(4,110,92,0,0,37.6,0.191,30)

#changing the input_data to numpy array
import_data_as_numpy_array = np.asarray(import_data)

#reshape the array as we are predicting for one instance
import_data_reshaped = import_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(import_data_reshaped)
print(std_data)

pradiction = classifier.predict(std_data)
print(pradiction)

if( pradiction[0] ==0):
    print("The person is not Diabetic")
else:
    print("The person is Diabetic")
```

    [[ 0.04601433 -0.34096773  1.18359575 -1.28821221 -0.69289057  0.71168975
      -0.84827977 -0.27575966]]
    [0]
    The person is not Diabetic
    

    C:\Users\DELL\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\utils\validation.py:2739: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
      warnings.warn(
    


```python

```


```python

```
