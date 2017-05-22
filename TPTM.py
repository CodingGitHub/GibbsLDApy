#!/usr/bin/env python
# encoding: utf-8
'''
@author: Jasperyang
@license: (C) Copyright 2013-2017, Jasperyang Corporation Limited.
@contact: yiyangxianyi@gmail.com
@software: GibbsLDA
@file: TPTM.py
@time: 5/18/2017 11:00 PM
@desc:   这个部分是功能性拓展，基于danmuLDA的实现部分
        1. 读取B站格式弹幕
        2. 生成各类矩阵 delta_s,delta_c等等
        3. 更新 lambda_s和 x_u_c_t 的函数
        4. 生成给原始LDA的alpha向量
'''

import eulerlib
import numpy as np
import math
import pandas as pd
from scipy import stats as st  # 用户正态函数的pdf
import threadpool
import time
import pickle
from Utils import *

'''
TPTM 的工具包

    updated : 取单个元素 iloc 改成 iat ，速度较快
                        loc  改成 at              -----> 主要加快了更新 x_u_c_t 的速度
'''
class TPTM(object):

    # 定义变量 默认数据 全部的类变量
    user = {}
    comments = []
    split_num = 10
    pool = None
    total_topic = 10
    iteration = 0
    yita = 0.005
    eur = None
    nw = None # model 传进来的矩阵
    gamma_s = 0.5  # 我自己设的
    gamma_c = 0.3  # 论文中做实验得到的最好的值
    sigma_s = 0.1  # 应该是每个片段的都不一样，但是这里我认为其实每个片段的topic分布没有统计可能性，不合理，都设成一样的了

    user_num = 0
    comments_num = 0
    comments = None
    user = None
    stopwords = None
    shots = None
    delta_c = None
    delta_s = None
    result_s = None
    result_c = None
    M_pre_c = None
    M_pre_s = None
    comment_all = None
    comment_all_sort = None
    user_sigma = None
    pi_c = None
    x_u_c_t = None
    column = None
    word2id = None
    word_fre = None
    user_ct = None
    alpha_c = None

    ut = None# Utils

    '''无参初始化，这是不能用的'''
    def __init__(self):
        self.pool = threadpool.ThreadPool(100 + 2)
        # 总的topic个数
        self.total_topic = 10
        self.yita = 0.05 / pow(2, (self.iteration % 10))
        print('danmu project init')
        # do nothing

    # @tested
    '''
    Api_name : init
    Argc :  splitnum (视频片段分割个数)
            total_topic (主题个数)
            pool_num (线程池的容量)
            nw_metric (单词对应topic的分布的矩阵)
            ut (Utils)
            gamma_s
            gamma_c
            sigma_s
    Func :  danmu 初始化
    '''
    def __init__(self,split_num,total_topic,pool_num,eur_maxnum,nw_metric,ut,gamma_s=0.5,gamma_c=0.3,sigma_s=0.1):
        self.user = {}
        self.comments = []
        self.split_num = split_num
        # 定义容量
        self.pool = threadpool.ThreadPool(pool_num + 2)
        # 总的topic个数
        self.total_topic = total_topic
        self.yita = 0.05 / pow(2, (self.iteration % 10))
        # 欧拉函数的定义
        self.eur = eulerlib.numtheory.Divisors(eur_maxnum)  # maxnum
        self.nw = nw_metric
        self.gamma_s = gamma_s
        self.gamma_c = gamma_c  # 一般设置 0.3
        self.sigma_s = sigma_s  # 应该是每个片段的都不一样，但是这里我认为其实每个片段的topic分布没有统计可能性，不合理，都设成一样的了，自己定一个0.1吧

        self.ut = ut
        self.user = self.ut.user
        self.comments = self.ut.comments
        self.user_num = self.ut.user_num
        self.comments_num = self.ut.comments_num
        self.shots = self.ut.shots
        self.com = self.ut.com
        self.stopwords = self.ut.stopwords
        self.word_fre = self.ut.word_fre

        # 下面这两个原本是在 preprocessing.py 中生成

        # 每一个用户的user-topic分布
        # sigma_u_t 是每个用户对于每一个topic的sigma值
        # 从文件里面读取每一个用户的每一个topic的sigma值
        # 每一行一个用户 (顺序就是下面生成的 user_ 中的顺序)
        self.user_sigma = pd.read_csv('data/sigma_u_t.csv')
        self.user_sigma = self.user_sigma.drop(['Unnamed: 0'], 1)
        self.user_sigma.fillna(0.1)

        # word2id 读取
        self.word2id = pd.read_csv('test_data/wordmap.txt', sep=' ')  # 读取单词对应id的表
        self.column = list(self.word2id)[0]  # 这个是因为第一行是单词的个数，会变成index，下面转换成字典后出现二级索引，所以做了处理
        self.word2id = self.word2id.to_dict()
        print('danmu project init')


    # @tested
    '''
    Api_name : delta
    Argc : i (index)
           j (index)
           result (可传入 result_c 或 result_s)
           delta (可传入 delta_c 或 delta_s)
    Func : 利用result生成delta，delta就是shot与shot之间的余弦距离
           # 计算 delta 的函数，通过不同的输入矩阵计算输出不同的矩阵
    '''
    def delta(self,i, j, result, delta):
        numerator = np.sum(result[i].time * result[j].time)
        denominator = pow(np.sum(pow(result[i].time, 2)), 0.5) * pow(np.sum(pow(result[j].time, 2)), 0.5)
        if denominator != 0:
            cos = numerator / denominator
        else:
            cos = 0
        delta[i][j] = cos

    # @tested
    '''
    Api_name : MkFakeSigma
    Argc : None
    Func : 制造一个假的 user-topic 分布
    '''
    def MkFakeSigma(self):
        user_num = len(self.user)
        f = open('data/sigma_u_t.csv', 'w')
        f.write(',user')
        for i in range(self.total_topic):
            f.write(',topic' + str(i))
        f.write('\n')
        for key in self.user.keys():
            f.write(',' + key)
            for j in range(self.total_topic):
                f.write(',0.1')
            f.write('\n')


    # @tested
    '''
    Api_name : preprocessing
    Argc : danmu_dir
    Func : 读取弹幕文件并分词处理，之后生成所需数据，这是重要的预处理
    生成数据：
             user : dict
             shots : list
             com : list
             lambda_s : array
             delta_c : array
             delta_s : array
             result_s : list
             result_c : list
             M_pre_c : array
             M_pre_s : array
             comment_all : dataframe
             comment_all_sort : dataframe
             pi_c : array
             x_u_c_t : dataframe
             user_ct : dataframe
    '''
    def preprocessing(self):

        # 计算每一个shot里面的所有的单词的词频 ------->   缺点：执行速度实在太慢了，后期需要修改 , 这一部分要执行十五分钟左右
        self.result_s = []
        for i in range(self.split_num):
            shot_word_fre = self.word_fre.copy()
            shot_word_fre['time'] = 0
            for x in self.shots[i]:
                for word in x:
                    index = shot_word_fre[self.word_fre[0] == word.encode('utf-8')].index
                    shot_word_fre.ix[index,'time'] = shot_word_fre.ix[index,'time'] + 1
            shot_word_fre = shot_word_fre.drop(1, 1)
            self.result_s.append(shot_word_fre)

        # 计算每一个comment的词频向量  -----------> 现在的办法是每个 comment 都有一个完整的词向量，便于后面的计算，问题是这样很占内存资源
        # 不按照每一个shot分片后内部的comment之间的delta计算，所有的comment进行计算
        self.result_c = []
        for i in range(self.split_num):
            for j in range(len(self.shots[i])):
                shot_word_fre = self.word_fre.copy()
                shot_word_fre['time'] = 0
                for x in self.shots[i][j]:
                    for word in x:
                        index = shot_word_fre[self.word_fre[0] == word.encode('utf-8')].index
                        shot_word_fre.ix[index, 'time'] = shot_word_fre.ix[index, 'time'] + 1
                shot_word_fre = shot_word_fre.drop(1, 1)
                self.result_c.append(shot_word_fre)

        # 这部分计算也十分耗时，我改成多线程了 ,速度很快了
        # 计算delta<s,_s> : 这里用的是词频向量 余弦值    -----> 下三角矩阵，后面方便计算
        # 从后面的shot往前计算
        self.delta_s = np.zeros((self.split_num, self.split_num))
        seq = range(self.split_num)
        # 修改 time 的数据类型 to float64
        for shot in self.result_s:
            shot.time = shot.time.astype('float64')



        start_time = time.time()  # 下面的多线程开始执行的时间
        seq.reverse()
        for i in seq:
            for j in range(i):
                lst_vars = [i, j, self.result_s, self.delta_s]
                func_var = [(lst_vars, None)]
                requests = threadpool.makeRequests(self.delta, func_var)
                [self.pool.putRequest(req) for req in requests]
        self.pool.wait()
        print('calculate delta_s %d second' % (time.time() - start_time))

        # 计算delta<c,_c> : 这里用的是词频向量 余弦值    -----> 下三角矩阵，后面方便计算
        # 从后往前
        # 这里是不按照每个shot分开然后计算里面的comment
        seq = range(len(self.result_c))
        # 修改 time 的数据类型 to float64
        for i in seq:
            self.result_c[i].time = self.result_c[i].time.astype('float64')

        # list存储
        self.delta_c = np.zeros((len(self.result_c), len(self.result_c)))

        start_time = time.time()  # 下面的多线程开始执行的时间
        for i in seq:
            for j in range(i):
                lst_vars = [i, j, self.result_c, self.delta_c]
                func_var = [(lst_vars, None)]
                requests = threadpool.makeRequests(self.delta, func_var)
                [self.pool.putRequest(req) for req in requests]
        self.pool.wait()
        print('calculate delta_c %d second' % (time.time() - start_time))


        # 利用上面的用户对应评论的字典 make 一个 dataframe
        user_ = pd.DataFrame()
        temp1 = []
        temp2 = []
        for key in self.user.keys():
            for i in range(len(self.user[key])):
                temp1.append(key)
                temp2.append(self.user[key][i])
        user_['user'] = temp1
        user_['comment'] = temp2

        # 处理得到一个大表，里面包括所有评论以及评论的人，和每个人对应的所有的topic的sigma值
        # 这里处理之后好像有点问题，有些用户没有，下面我直接就都填充0.1了
        comment_per_shot = []
        for i in range(self.split_num):
            temp = pd.DataFrame(self.com[i])
            u = []
            tem = pd.DataFrame()
            for j in range(len(temp)):
                user_id = user_[user_.comment == temp[0][j]].iat[0][0]
                u.append(user_id)
                a = self.user_sigma[self.user_sigma.user == user_id].iloc[:, 1:]
                tem = [tem, a]
                tem = pd.concat(tem)
            tem = tem.reset_index().drop(['index'], 1)
            temp['user'] = pd.DataFrame(u)
            temp = temp.join(tem)
            comment_per_shot.append(temp)

        # 有了上面的矩阵后，计算论文中提到的 M_pre_s 以及 M_pre_c
        # 需要两个衰减参数 gamma_s 以及 gamma_c
        # M_pre_s 比较好计算，M_pre_c 比较复杂一点，因为涉及到了每一个shot
        self.M_pre_s = np.zeros((self.split_num, self.total_topic))  # 行：shot个数    列：topic个数
        self.lambda_s = np.zeros((self.split_num, self.total_topic))


        # 先初始化 M_pre_s[0] 以及 lambda_s[0]
        mu = 0  # 初始的 M_pre_s[0] 都是0
        s = np.random.normal(mu, self.sigma_s, self.total_topic)  # 不知道这个做法对不对，用正态生成x坐标，再用pdf去生成y值
        self.lambda_s[0] = st.norm(mu, self.sigma_s).pdf(s)

        # 从 第1的开始
        for i in range(1, self.split_num):
            for topic in range(self.total_topic):  # 先循环topic
                numerator = 0
                denominator = 0
                for j in range(i):
                    numerator += np.exp(-self.gamma_s * self.delta_s[i][j]) * self.lambda_s[j][topic]
                    denominator += np.exp(-self.gamma_s * self.delta_s[i][j])
                self.M_pre_s[i][topic] = numerator / denominator
                s = np.random.normal(self.M_pre_s[i][topic], self.sigma_s, 1)
                self.lambda_s[i][topic] = st.norm(self.M_pre_s[i][topic], self.sigma_s).pdf(s)

        # 所有的 comment 的一个 dataframe ,comment-user_id-topic0,1,2...99 ，后面的topic分布是user_id的
        self.comment_all = pd.concat(comment_per_shot).reset_index().drop('index', 1)
        # 给那些没有topic分布的用户填充0.1 ----> 缺失值（就是生成用户的topic分布表没有生成全）
        self.comment_all = self.comment_all.fillna(0.1)
        self.comment_all = self.comment_all.rename(columns={0: 'comment'})


        # 生成 pi_c 和 M_pre_c 不同于上面，因为这里是对每个shot的面的comment进行操作
        # 先初始化 M_pre_c[0] 和 第0个 shot 里的第一个 comment 对应的 pi_c[0]
        self.M_pre_c = np.zeros((len(self.comment_all), self.total_topic))  # 行：shot个数    列：topic个数
        self.pi_c = np.zeros((len(self.comment_all), self.total_topic))
        self.alpha_c = np.zeros((len(self.comment_all), self.total_topic)) # 初始化 alpha_c
        for i in range(self.total_topic):
            self.pi_c[0][i] = self.lambda_s[0][i] * self.comment_all.iat[0][i + 2] + self.M_pre_c[0][i]

        start = 0  # shot 之间的位移
        for q in range(self.split_num):
            if q == 0:
                for i in range(1, len(self.com[q])):
                    for topic in range(self.total_topic):  # 先循环topic
                        numerator = 0
                        denominator = 0
                        for j in range(i):
                            numerator += np.exp(-self.gamma_c * self.delta_c[i][j]) * self.pi_c[j][topic]
                            denominator += np.exp(-self.gamma_c * self.delta_c[i][j])
                        self.M_pre_c[i][topic] = numerator / denominator
                        self.pi_c[i][topic] = self.lambda_s[q][topic] * self.comment_all.iat[i][topic + 2] + self.M_pre_c[i][topic]
                start += len(self.com[q])
            else:
                for i in range(start, start + len(self.com[q])):
                    for topic in range(self.total_topic):  # 先循环topic
                        numerator = 0
                        denominator = 0
                        for j in range(i):
                            numerator += np.exp(-self.gamma_c * self.delta_c[i][j]) * self.pi_c[j][topic]
                            denominator += np.exp(-self.gamma_c * self.delta_c[i][j])
                        self.M_pre_c[i][topic] = numerator / denominator
                        self.pi_c[i][topic] = self.lambda_s[q][topic] * self.comment_all.iat[i][topic + 2] + self.M_pre_c[i][topic]
                start += len(self.com[q])

        # 将 comment_all 升级成一个新的大表 comment_all_sort 结构为 {comment,user_id,user_id的topic,该comment属于的shot的topic分布},有了这个表，后面的处理会很方便
        a1 = pd.concat([self.comment_all, pd.DataFrame(self.M_pre_c)], axis=1)
        temp = []
        for i in range(self.split_num):
            for j in range(len(self.shots[i])):
                t = pd.DataFrame(self.lambda_s)[i:i + 1]
                t['shot'] = i
                t['com'] = j
                temp.append(t)
        a2 = pd.concat(temp)
        a2 = a2.reset_index().drop('index', 1)
        self.comment_all = pd.concat([a1, a2], axis=1)  # comment_all 不 sort 版的留给更新 lambda_s 用
        self.comment_all_sort = self.comment_all.sort_values('user')  # 按照 user 排序

        # 生成 user-topic 分布的 dataframe
        self.x_u_c_t = np.zeros((len(self.comment_all_sort), self.total_topic))
        for i in range(len(self.comment_all_sort)):
            for topic in range(self.total_topic):
                s = np.random.normal(mu, self.comment_all_sort.iat[i,topic + 2], 1)
                self.x_u_c_t[i][topic] = st.norm(mu, self.comment_all_sort.iat[i,topic + 2]).pdf(s)
        user_id = self.comment_all_sort.drop_duplicates('user')['user'].reset_index().drop('index', 1)
        self.x_u_c_t = user_id.join(pd.DataFrame(self.x_u_c_t))

        # 每个人评论的次数的dataframe
        self.user_ct = self.comment_all_sort.groupby('user').count()['topic0']

        # 保存重要数据，矩阵和dataframe到 data/progress文件夹里
        properties = open('data/progress/properties.xml','w')
        properties.write('user_num:' + str(self.user_num) + '\n')
        properties.write('comments_num:' + str(self.comments_num) + '\n')
        properties.write('iteration:' + str(self.iteration) + '\n')
        output = open('data/progress/comments.pkl', 'wb')
        pickle.dump(self.comments,output,-1)
        output = open('data/progress/user.pkl','wb')
        pickle.dump(self.user,output)
        output = open('data/progress/shots.pkl', 'wb')
        pickle.dump(self.shots,output,-1)
        output = open('data/progress/com.pkl', 'wb')
        pickle.dump(self.com,output,-1)
        np.save('data/progress/lambda_s.npy',self.lambda_s)
        np.save('data/progress/delta_c.npy',self.delta_c)
        np.save('data/progress/delta_s.npy',self.delta_s)
        output = open('data/progress/result_s.pkl','wb')
        pickle.dump(self.result_s,output,-1)
        output = open('data/progress/result_c.pkl','wb')
        pickle.dump(self.result_c,output,-1)
        np.save('data/progress/M_pre_s.npy',self.M_pre_s)
        np.save('data/progress/M_pre_c.npy', self.M_pre_c)
        self.comment_all.to_csv('data/progress/comment_all.csv',encoding='utf-8')
        self.comment_all_sort.to_csv('data/progress/comment_all_sort.csv',encoding='utf-8')
        np.save('data/progress/pi_c.npy',self.pi_c)
        self.x_u_c_t.to_csv('data/progress/x_u_c_t.csv',encoding='utf-8')
        self.user_ct.to_csv('data/progress/user_ct.csv',encoding='utf-8')


    # @tested
    '''
    Api_name : load_stage_data
    Argc : None
    Func : lgt函数
    '''
    def load_stage_data(self):
        print('loading')
        pro = []
        properties = open('data/progress/properties.xml')
        for line in properties.readlines():
            pro.append(line.split(':')[1])
        self.user_num = int(pro[0][:-1])
        self.comments_num = int(pro[1][:-1])
        self.iteration = int(pro[2][:-1])
        input = open('data/progress/comments.pkl','rb')
        self.comments = pickle.load(input)
        input = open('data/progress/user.pkl','rb')
        self.user = pickle.load(input)
        input = open('data/progress/shots.pkl','rb')
        self.shots = pickle.load(input)
        # self.com = pickle.load(input)
        self.lambda_s = np.load('data/progress/lambda_s.npy')
        self.delta_c = np.load('data/progress/delta_c.npy')
        self.delta_s = np.load('data/progress/delta_s.npy')
        input = open('data/progress/result_s.pkl','rb')
        self.result_s = pickle.load(input)
        input = open('data/progress/result_c.pkl','rb')
        self.result_c = pickle.load(input)
        self.M_pre_s = np.load('data/progress/M_pre_s.npy')
        self.M_pre_c = np.load('data/progress/M_pre_c.npy')
        self.comment_all = pd.read_csv('data/progress/comment_all.csv')
        self.comment_all = self.comment_all.drop('Unnamed: 0',1)
        self.comment_all_sort = pd.read_csv('data/progress/comment_all_sort.csv')
        self.comment_all_sort = self.comment_all_sort.drop('Unnamed: 0', 1)
        self.pi_c = np.load('data/progress/pi_c.npy')
        self.x_u_c_t = pd.read_csv('data/progress/x_u_c_t.csv')
        self.x_u_c_t = self.x_u_c_t.drop('Unnamed: 0', 1)
        self.user_ct = pd.read_csv('data/progress/user_ct.csv')
        print('loading finished!')


    # @tested
    '''
    Api_name : lgt
    Argc : y
    Func : lgt函数
    '''
    def lgt(self, y):
        return math.log(1 + math.exp(y))

    # @tested
    '''
    Api_name : dlgt
    Argc : y
    Func : lgt求导
    '''
    def dlgt(self, y):
        return 1 / ((1 + math.exp(y)) * np.log(10))


    # @tested
    '''
    Api_name : calculate_lambda_s
    Argc : shot
           start
    Func : 计算关于输入shot的lambda值
           # 线程函数 --> 计算 yita_lambda_s
    '''
    def calculate_lambda_s(self,shot,start):
        for topic in range(self.total_topic):
            result = 0
            lam_s = self.lambda_s[shot][topic]
            for comment in range(len(self.shots[shot])):
                x_u = self.comment_all.iat[comment + start, topic + self.total_topic + 2]
                m_pre_c = self.M_pre_c[comment + start][topic]
                t1 = x_u * self.dlgt(x_u * lam_s + m_pre_c)
                t2 = []
                for t in range(self.total_topic):
                    t2.append(self.lgt(self.comment_all.iat[comment + start, t + 2] * lam_s + self.M_pre_c[comment + start][t]))
                t2 = sum(t2)
                t3 = t2
                t2 = self.eur.phi(t2)
                t3 = self.eur.phi(t3 + len(self.shots[shot][comment]))
                n_tc = 0
                for word in self.shots[shot][comment]:
                    word = word.encode('utf-8')
                    if word != ' ':
                        try:
                            num = self.word2id[self.column][word]
                            n_tc += self.nw[num][topic]
                        except Exception as e:          # 这里会出现找不到的问题，一般就是最后一段的一些单词，我处理过了，又出现问题了，还是得再找找
                            print('Exception:'+str(e))
                t4 = self.eur.phi(self.lgt(x_u * lam_s + m_pre_c) + n_tc)
                t5 = self.eur.phi(self.lgt(x_u * lam_s + m_pre_c))
            result += t1 * (t2 - t3 + t4 - t5)
            self.lambda_s[shot][topic] = self.lambda_s[shot][topic] - self.yita * (-(lam_s + self.M_pre_s[shot][topic]) / (lam_s * lam_s) + result)

    # @tested
    '''
    Api_name : calculate_x_u_c_t
    Argc : i
           start
    Func : 计算关于输入每个用户的topic分布
           # x_u_c_t 的更新代码
           # 注意 ：这里的 comment_all 已经排过序了，和上面的不一样
    '''
    def calculate_x_u_c_t(self,i, start):
        for topic in range(self.total_topic):
            result = 0
            for j in range(start, start + self.user_ct.iloc[i]):
                lambda_s_t = self.comment_all_sort.iat[j,topic + self.total_topic + self.total_topic + 2]
                m_pre_c_t = self.comment_all_sort.iat[j,topic + self.total_topic + 2]
                x_u = self.x_u_c_t.iat[i,topic + 1]
                t1 = lambda_s_t * self.dlgt(x_u * lambda_s_t + m_pre_c_t)
                t2 = []
                for k in range(self.total_topic):
                    t2.append(self.lgt(self.comment_all_sort.iat[j,k + 2] * self.comment_all_sort.iat[j,k + self.total_topic + self.total_topic + 2] + self.comment_all_sort.iat[j,k + self.total_topic + 2]))
                t3 = self.eur.phi(sum(t2) + len(self.shots[int(self.comment_all_sort.ix[j, ['shot']])][int(self.comment_all_sort.ix[j, ['com']])]))
                t2 = self.eur.phi(sum(t2))
                n_tc = 0
                for word in self.shots[int(self.comment_all_sort.ix[j, ['shot']])][int(self.comment_all_sort.ix[j, ['com']])]:
                    word = word.encode('utf-8')
                    if word != ' ':
                        try:
                            num = self.word2id[self.column][word]
                            n_tc += self.nw[num][topic]
                        except Exception as e:
                            print('Exception:' + str(e))
                t4 = self.eur.phi(self.lgt(x_u * lambda_s_t + m_pre_c_t) + n_tc)
                t5 = self.eur.phi(self.lgt(x_u * lambda_s_t + m_pre_c_t))
                result += t1 * (t2 - t3 + t4 - t5)
            self.x_u_c_t.iat[i,topic + 1] = self.x_u_c_t.iat[i,topic + 1] - self.yita * (-x_u / (self.comment_all_sort.iat[j,topic + 2] * self.comment_all_sort.iat[j,topic + 2]) + result)

    # @tested
    '''
     Api_name : update_lambda_s
     Argc : iteration (现阶段迭代的次数)
     Func : 更新所有shot的lambda_s分布
     '''
    def update_lambda_s(self,iteration):
        self.iteration = iteration
        self.yita = 0.005 / pow(2, (self.iteration % 10))
        start_time = time.time()  # 下面的多线程开始执行的时间
        start = 0  # 初始化，用于控制在哪一个shot里面
        for shot in range(len(self.shots)):
            lst_vars = [shot, start]
            func_var = [(lst_vars, None)]
            start += len(self.shots[shot])  # start 增加位移，移动一个shot
            requests = threadpool.makeRequests(self.calculate_lambda_s, func_var)
            [self.pool.putRequest(req) for req in requests]
        self.pool.wait()
        # 保存每一轮迭代会改变的数据
        properties = open('data/progress/properties.xml','w')
        properties.write('user_num:' + str(self.user_num) + '\n')
        properties.write('comments_num:' + str(self.comments_num) + '\n')
        properties.write('iteration:' + str(self.iteration) + '\n')
        np.save('data/progress/lambda_s.npy', self.lambda_s)
        np.save('data/progress/M_pre_s.npy',self.M_pre_s)
        np.save('data/progress/M_pre_c.npy', self.M_pre_c)
        self.x_u_c_t.to_csv('data/progress/x_u_c_t.csv',encoding='utf-8')
        print('updating lambda_s %d second' % (time.time() - start_time))


    # @tested
    '''
     Api_name : update_x_u_c_t
     Argc : iteration (现阶段迭代的次数)
     Func : 更新所有用户的topic分布
     '''
    def update_x_u_c_t(self,iteration):
        self.iteration = iteration
        self.yita = 0.05 / pow(2, (self.iteration % 10))
        start_time = time.time()  # 下面的多线程开始执行的时间
        start = 0  # 初始化，用于控制在哪一个shot里面
        for i in range(len(self.user_ct)):
            lst_vars = [i, start]
            func_var = [(lst_vars, None)]
            start += self.user_ct.iloc[i]  # start 增加位移，移动一个shot
            requests = threadpool.makeRequests(self.calculate_x_u_c_t, func_var)
            [self.pool.putRequest(req) for req in requests]
        self.pool.wait()
        # 保存每一轮迭代会改变的数据
        properties = open('data/progress/properties.xml','w')
        properties.write('user_num:' + str(self.user_num) + '\n')
        properties.write('comments_num:' + str(self.comments_num) + '\n')
        properties.write('iteration:' + str(self.iteration) + '\n')
        np.save('data/progress/lambda_s.npy', self.lambda_s)
        np.save('data/progress/M_pre_s.npy',self.M_pre_s)
        np.save('data/progress/M_pre_c.npy', self.M_pre_c)
        self.x_u_c_t.to_csv('data/progress/x_u_c_t.csv',encoding='utf-8')
        print('updating x_u_c_t %d second' % (time.time() - start_time))

    # @tested
    '''
     Api_name : update_Mpre_s
     Argc : None
     Func : 更新 M_pre_s
    '''
    def update_Mpre_s(self,iteration):
        self.iteration = iteration
        self.yita = 0.05 / pow(2, (self.iteration % 10))
        for i in range(0, self.split_num):
            for topic in range(self.total_topic):  # 先循环topic
                numerator = 0
                denominator = 0
                for j in range(i):
                    numerator += np.exp(-self.gamma_s * self.delta_s[i][j]) * self.lambda_s[j][topic]
                    denominator += np.exp(-self.gamma_s * self.delta_s[i][j])
                self.M_pre_s[i][topic] = numerator / denominator

    # @tested
    '''
     Api_name : update_Mpre_c
     Argc : None
     Func : 更新 M_pre_c
    '''
    def update_Mpre_c(self,iteration):
        self.iteration = iteration
        self.yita = 0.05 / pow(2, (self.iteration % 10))
        start = 0  # shot 之间的位移
        for q in range(self.split_num):
            for i in range(start, start + len(self.com[q])):
                for topic in range(self.total_topic):  # 先循环topic
                    numerator = 0
                    denominator = 0
                    for j in range(i):
                        numerator += np.exp(-self.gamma_c * self.delta_c[i][j]) * self.pi_c[j][topic]
                        denominator += np.exp(-self.gamma_c * self.delta_c[i][j])
                    self.M_pre_c[i][topic] = numerator / denominator

    # @tested
    '''
     Api_name : Get_alpha_c
     Argc : None
     Func : 获取 alpha_c 的值（对于每一条评论的topic分布）
    '''
    def Get_alpha_c(self):
        self.alpha_c = self.pi_c.copy()
        for i in range(self.comments_num):
            for j in range(self.total_topic):
                self.alpha_c[i][j] = self.lgt(self.alpha_c[i][j])
        return self.alpha_c