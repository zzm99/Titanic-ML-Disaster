import numpy as np 
import pandas as pd 

data_train = pd.read_csv("train.csv")

# Cabin 的缺失值填充函数
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin']='YES'
    df.loc[(df.Cabin.isnull()),'Cabin']='NO'
    return df

data_train=set_Cabin_type(data_train)

# 使用 RandomForestClassifier 填补缺失的年龄属性
from sklearn.ensemble import RandomForestRegressor
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中,因为逻辑回归算法输入都需要数值型特征
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    age_known=age_df.loc[age_df.Age.notnull()].values   # 要用sklearn的模型，输入参数要把Series和DataFrame 转成nparray 
    age_unknown=age_df.loc[age_df.Age.isnull()].values
    #将有值的年龄样本 训练数据，然后用训练好的模型用于 预测 缺失的年龄样本的值
    #训练数据的特征集合
    X=age_known[:,1:]
    #训练数据的目标
    y=age_known[:,0]
    
    #构建随机森林回归模型器
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    # 拟合数据,训练模型
    rfr.fit(X,y)
    # 用训练好的模型 预测 有缺失值的年龄属性
    predictedAges=rfr.predict(age_unknown[:,1:])
    # 得到的预测结果，去填补缺失数据
    df.loc[df.Age.isnull(),'Age']=predictedAges
    return df,rfr

data_train,rfr=set_missing_ages(data_train)   

dummies_Cabin = pd.get_dummies(data_train.Cabin,prefix='Cabin') #ptefix 是前缀，因子花之后的字段名为 前缀_类名

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1) #将哑编码的内容拼接到data_train后
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)  # 把编码前的字段删除

# 这里用StandardScaler类对
from sklearn.preprocessing import StandardScaler
# 创建一个定标器
scaler=StandardScaler()
# 拟合数据 
#---fit和transform为两个动作，可用fit_transform 合并完成
#df['Age_Scale']=scaler.fit_transform(df.Age.values.reshape(-1,1))  # 若为单个特征，需要reshape为（-1,1）

#--但是由于test和train 需要用同一个fit出来的参数，所以需要记录fit参数，用于test数据的标准化，因此分开计算
Age_Scale_parame=scaler.fit(df.Age.values.reshape(-1,1))
#df['Age_Scale']=scaler.transform(df.Age.values.reshape(-1,1))
df['Age_Scale']=scaler.fit_transform(df.Age.values.reshape(-1,1),Age_Scale_parame)

Fare_Scale_parame=scaler.fit(df.Fare.values.reshape(-1,1))
df['Fare_Scale']=scaler.fit_transform(df.Age.values.reshape(-1,1),Fare_Scale_parame)
df.drop(['Age', 'Fare'], axis=1, inplace=True)

from sklearn.linear_model import LogisticRegression
df_train=df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*').values #用正则表达式把需要的字段过滤出来
# 训练特征
df_train_feature=df_train[:,1:]
#训练目标
df_train_label=df_train[:,0]
#构建逻辑回归分类器
clf=LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
#拟合数据
clf.fit(df_train_feature,df_train_label)

data_test = pd.read_csv("test.csv")
# 缺失值填充
data_test.loc[data_test.Fare.isnull(),'Fare']=0
data_test= set_Cabin_type(data_test)
age_data = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
age_test=age_data[age_data.Age.isnull()].values
# age的缺失值填充也用前训练数据 的age值fit计算的模型,所以可以直接预测
predictedAges = rfr.predict(age_test[:,1:])
data_test.loc[data_test.Age.isnull(),'Age'] = predictedAges

# 类目特征因子化
dummies_Cabin = pd.get_dummies(data_test.Cabin, prefix= 'Cabin')
dummies_Sex = pd.get_dummies(data_test.Sex, prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test.Pclass, prefix= 'Pclass')
dummies_Embarked = pd.get_dummies(data_test.Embarked, prefix= 'Embarked')

#归一化也用训练数据fit出来的参数进行转化
data_test['Age_scaled'] = scaler.fit_transform(data_test['Age'].values.reshape(-1, 1), Age_Scale_parame)
data_test['Fare_scaled'] = scaler.fit_transform(data_test['Fare'].values.reshape(-1, 1), Fare_Scale_parame)

# 拼接处理后数据以及删除处理前数据
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Age','Fare'], axis=1, inplace=True)

df_test=df_test.values
# df_test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')  #可用正则表达式取删选数据
predict_result=clf.predict(df_test[:,1:])
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predict_result.astype(np.int32)})
result.to_csv("gender_submission.csv", index=False)


