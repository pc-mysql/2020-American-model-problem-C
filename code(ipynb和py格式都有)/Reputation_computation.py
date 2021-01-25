#计算reputation然后进行可视化
from datetime import datetime

import numpy as np
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
hair['review_date'] = pd.to_datetime(hair['review_date'],format="%m/%d/%Y")  #也就是得到了date的时间
pac['review_date'] = pd.to_datetime(pac['review_date'],format="%m/%d/%Y")   #得到pac的时间
microwave['review_date'] = pd.to_datetime(microwave['review_date'],format="%m/%d/%Y")
hair['flag']=1
pac['flag']=2
microwave['flag']=3
#将日期转换为具体的年
hair['Year'] = hair['review_date'].dt.year
pac['Year'] = pac['review_date'].dt.year
microwave['Year'] = microwave['review_date'].dt.year  #
#转换为具体的月

hair['month'] = hair['review_date'].apply(lambda x:datetime.strptime(x.strftime('%Y-%m'),"%Y-%m"))
pac['month'] = pac['review_date'].apply(lambda x:datetime.strptime(x.strftime('%Y-%m'),"%Y-%m"))
microwave['month'] = microwave['review_date'].apply(lambda x:datetime.strptime(x.strftime('%Y-%m'),"%Y-%m"))


#计算当天weight有效值在helpgul中的比例
for total in [hair,pac,microwave]:
    # total = pd.concat([hair,pac,microwave],axis=0)  ##得到总表
    #得到所有的个数
    total['review_date'] = total['review_date'].apply(lambda x:datetime.strptime(x.strftime('%Y-%m-%d'),"%Y-%m-%d"))  #将数据转化为年-日的样子
    total_group = total.groupby(['flag','review_date'])['star_rating'].count() #计算每个评论中每个东西的个
    star_rating_group = total.groupby(['flag','review_date','star_rating'])['review_body'].count().reset_index()
    #/total_group
    #使用type()
    time_set= total['review_date'][total['review_date'].duplicated()==False] #找到所有False的数据
    time_set= sorted(time_set.to_list(),reverse=True)
    time_dict = {}
    for i in time_set:
        time_dict[i] =(datetime.strptime(str(i)[0:10],"%Y-%m-%d") - datetime.strptime(str(time_set[-1])[0:10],"%Y-%m-%d")).days

    time_diff = list(time_dict.values()) #打印每个值
    time_diff = np.array(time_diff[:-1])-np.array(time_diff[1:])
    time_range=range(0,(datetime.strptime(str(time_set[0])[0:10],"%Y-%m-%d") - datetime.strptime(str(time_set[-1])[0:10],"%Y-%m-%d")).days,1)
    total['helpful_votes_sum'] = total.groupby(['review_date'])['helpful_votes'].transform("sum") #使用sum函数
    #得到helpful_votes的比例
    total['helpful_votes_ratio'] = total['helpful_votes'] / total['helpful_votes_sum']   #使用一些ratio获得
    total['helpful_votes_ratio'] = total['helpful_votes_ratio'].fillna(1)
    #计算星级评分
    total['rating_score'] = total['star_rating'].apply(lambda x:x-3)
    print(total['star_rating'].value_counts())
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



