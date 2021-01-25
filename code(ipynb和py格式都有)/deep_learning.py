#%%import data and library
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.python.keras.layers import LSTM, Dense
os.chdir(r'C:\Users\90930\Desktop\美赛建模\2020年')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from seaborn import color_palette

plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
font = {'family' : 'serif',   #将所有字体改变为粗体 serif
}
matplotlib.rc('font', **font)
import missingno
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
#读取数据
hair = pd.read_csv("hair_dryer.tsv", sep="\t", )
microwave = pd.read_csv("microwave.tsv", sep="\t")
pac = pd.read_csv("pacifier.tsv", sep="\t")
#赋予标志位
hair['flag']=1
pac['flag']=2
microwave['flag']=3


#%分别对三个图片进行画图
for total,name in zip([hair,microwave,pac],['hair','microwave','pac']):
    missingno.matrix(total)   #缺失值判断
    plt.show()
    total['review_date'] = pd.to_datetime(total['review_date'],format="%m/%d/%Y")  #也就是得到了date的时间
    total['Year'] = total['review_date'].dt.year #转换为1年

    total['month'] = total['review_date'].apply(lambda x:datetime.strptime(x.strftime('%Y-%m'),"%Y-%m")) #转换为年与月组合格式  ##得到总表



#%%得到评论平均长度


#%%画出饼状图

#total ratio
# for total,name in zip([hair,microwave,pac],['hair','microwave','pac']):
#     star_rating_ratio =  total.groupby(['star_rating'])['verified_purchase'].count()
#     star_rating_ratio_good =  total[total['helpful_votes']>2].groupby(['star_rating'])['verified_purchase'].count()
#     print(star_rating_ratio_good)     #
#     labels='one star','two star','three star','four star','five star'
#     colors='#55efc4','#81ecec','#74b9ff','#fd79a8','#a29bfe'
#     explode=0,0,0,0,0.1
#     plt.pie(star_rating_ratio, colors=colors,explode=explode, autopct='%1.1f%%', shadow=True,startangle=50)
#     plt.title("the "+name+" star ratings ratio",fontsize=15,loc='left')
#     plt.legend(labels=labels,loc='upper right',)
#     plt.axis('equal')
#     plt.show()
#
# #helpfulness_ratio
#     labels='one star','two star','three star','four star','five star' #定义label值
#     colors='#55efc4','#81ecec','#74b9ff','#fd79a8','#a29bfe'
#     explode=0,0,0,0,0.1
#     plt.pie(star_rating_ratio_good, colors=colors,explode=explode, autopct='%1.1f%%', shadow=True,startangle=50)
#     plt.title("the star ratings ratio of "+name+" (helpful)",fontsize=15,loc='left')
#     plt.legend(labels=labels,loc='upper right',)
#     plt.axis('equal')
#     plt.show()

#%%计算reputation score

#计算当天weight有效值在helpgul中的比例
for total in [hair,pac,microwave]:
    #得到所有的个数
    #将数据转化为年-日的样子
    star_rating_group = total.groupby(['flag','review_date','star_rating'])['review_body'].count().reset_index()
    total['helpful_votes_sum'] = total.groupby(['review_date'])['helpful_votes'].transform("sum") #使用sum函数
    #得到helpful_votes的比例
    total['helpful_votes_ratio'] = total['helpful_votes'] / total['helpful_votes_sum']   #使用一些ratio获得
    #填充NAN值为1
    total['helpful_votes_ratio'] = total['helpful_votes_ratio'].fillna(0)
    #计算星级评分
    total['rating_score'] = total['star_rating'].apply(lambda x:x-3)
    #计算声誉评分
    total['reputation_score'] = total.apply(lambda x:x['rating_score']*(1+x['helpful_votes_ratio']),axis=1)
    #集中求每个声誉的总和
    reputation_score_group = total.groupby('review_date')['reputation_score'].sum().reset_index() #也就是声誉总和的东西           #
    m=200
    list1=list()
    for i in range(m,reputation_score_group.index[-1]):
        sum1=0
        for j in range(i-199,i):
            time_set = reputation_score_group['review_date']
            score = reputation_score_group['reputation_score']
            day= (datetime.strptime(str(time_set[i])[0:10],"%Y-%m-%d") - datetime.strptime(str(time_set[j])[0:10],"%Y-%m-%d")).days
            sum1+=(1.0/day)*score[j]        #
        list1.append(sum1)
    result=pd.Series(list1)
    print(result)
    result.plot()
    plt.show()


#%% become a Supervised learning
    def timetimeseries_to_supervised(data,lag=1):
        df = pd.DataFrame(data)
        #create many columns
        colums = [data.shift(i) for i in range(1, lag+1)]  #
        colums = pd.DataFrame(colums).T
        df = pd.concat([colums,df], axis=1)
        return df

    X = result.values
    X = X.reshape(len(X), 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(X)
    scaled_X = scaler.transform(X)      # scaler.transfrom()  标准化
    scaled_series = pd.Series(scaled_X[:, 0]) #transfrom to orginal
    train  = timetimeseries_to_supervised(scaled_series,1)
    train = train.fillna(0) #
    train = train.values
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    print(X)
    model = keras.Sequential()
    model.add(LSTM(units=50, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')      #
    model.fit(X, y, epochs=10, batch_size=1, verbose=1, shuffle=False)
    y_pred = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred)
    print(y_pred)

