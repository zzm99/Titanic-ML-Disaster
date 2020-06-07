# Titanic: Machine Learning from Disaster

The competition is simple: use machine learning to create a model that predicts which passengers survived the Titanic shipwreck.

根据乘客的个人信息和存活情况，运用机器学习工具来预测哪些乘客在悲剧中幸存下来。这是一个典型的二分类问题。

```python
import numpy as np 
import pandas as pd 
from pandas import DataFrame, Series

# /kaggle/input/titanic/train.csv
# /kaggle/input/titanic/gender_submission.csv
# /kaggle/input/titanic/test.csv

data_train = pd.read_csv("/kaggle/input/titanic/train.csv")
data_train.columns
data_train.info()
data_train.describe()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB


PassengerId Survived    Pclass  Age SibSp   Parch   Fare
count   891.000000  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean    446.000000  0.383838    2.308642    29.699118   0.523008    0.381594    32.204208
std 257.353842  0.486592    0.836071    14.526497   1.102743    0.806057    49.693429
min 1.000000    0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25% 223.500000  0.000000    2.000000    20.125000   0.000000    0.000000    7.910400
50% 446.000000  0.000000    3.000000    28.000000   0.000000    0.000000    14.454200
75% 668.500000  1.000000    3.000000    38.000000   1.000000    0.000000    31.000000
max 891.000000  1.000000    3.000000    80.000000   8.000000    6.000000    512.329200
```

PassengerId => 乘客ID

Pclass => 乘客舱级(1/2/3等舱位)

Name => 乘客姓名

Sex => 性别

Age => 年龄

SibSp => 堂兄弟/妹个数

Parch => 父母与小孩个数

Ticket => 船票信息

Fare => 票价

Cabin => 舱位编号

Embarked => 登船港口（3个港口)

Survived => 幸存情况（1为幸存）

乘客舱级，性别，登船港口，幸存情况，堂兄妹个数、父母与小孩个数是离散属性

年龄，票价为连续值属性

乘客ID, 乘客姓名，船票信息，舱位编号等属性具有唯一值

年龄、舱位编号 有缺失值

