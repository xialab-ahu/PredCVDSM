#-*- coding = utf-8 -*-
# @time :2021/4/20 21:48
# @Auther :ma
# @file : trainRF
import json
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, matthews_corrcoef, \
    accuracy_score, roc_auc_score, precision_recall_curve,auc
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from threading import Thread
from concurrent.futures import  ThreadPoolExecutor,ProcessPoolExecutor
from multiprocessing import Pool
import os, time, random
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as sk_model_selection

def metricsScores(y_true, pred_proba, thres=0.50):
    y_pred = [(0. if item < thres else 1.) for item in pred_proba]  # pred_proba > thres
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()  # 混淆矩阵tn, fp, fn, tp
    Sen = recall_score(y_true, y_pred)  # 召回率、灵敏度
    Spe = tn / (tn + fp)  # 特异性
    Pre = precision_score(y_true, y_pred)  # 精确度
    F1 = f1_score(y_true, y_pred)  # F1-scores
    MCC = matthews_corrcoef(y_true, y_pred)  # 马修斯相关系数
    ACC = accuracy_score(y_true, y_pred)  # 准确度
    AUC = roc_auc_score(y_true, pred_proba)  # ROC曲线下围面积
    precision_prc, recall_prc, _ = precision_recall_curve(y_true, pred_proba)  # P-R曲线中x,y轴的数组
    PRC = auc(recall_prc, precision_prc)  # P-R曲线下围面积
    metrics_list = [Sen, Spe, Pre, F1, MCC, ACC, AUC, PRC, tn, fp, fn, tp, thres]
    return metrics_list
def dataRead( posPath, negPath, testPosPath=None, testNegPath=None, testfrequence=6, frequence=6, header=None,
             testHeader=None, ):
    # 训练集
    p_data = pd.read_csv(posPath.format(frequence), header=header,
                         delimiter='\t', na_values=['nan', 'na', 'Na', ''])  # ,0.0
    n_data = pd.read_csv(negPath.format(frequence), header=header,
                         delimiter='\t', na_values=['nan', 'na', 'Na',''])  # ,0.0
    Xtrain = np.vstack((n_data, p_data))
    Ytrain = np.hstack((np.zeros(len(n_data)), np.ones(len(p_data))))
    # 测试集
    if testPosPath != None and testNegPath != None:
        p_data = pd.read_csv(testPosPath.format(testfrequence), header=testHeader,
                             delimiter='\t', na_values=['nan', 'na', 'Na'])  # ,0.0
        n_data = pd.read_csv(testNegPath.format(testfrequence), header=testHeader,
                             delimiter='\t', na_values=['nan', 'na', 'Na'])  # ,0.0
        Xtest = np.vstack((n_data, p_data))
        Ytest = np.hstack((np.zeros(len(n_data)), np.ones(len(p_data))))
        # print(p_data)
        return Xtrain, Ytrain, Xtest, Ytest, frequence
    else:
        return Xtrain, Ytrain, frequence
def fillMeanMaxMin(preXtrain,testData,fillMethod="mean"):
    '''
    allowed_fillMethod = ["mean", "median", "most_frequent", "constant"]
    '''
        # 训练集数据均值填充，使用训练集数据填充验证集
    imp_mean = SimpleImputer(strategy=fillMethod)
    Xtrain = imp_mean.fit_transform(preXtrain)
    Xtest = imp_mean.transform(testData)
    # 归一化处理
    maxmin = preprocessing.MinMaxScaler()
    Xtrain = maxmin.fit_transform(Xtrain)
    Xtest = maxmin.transform(Xtest)
    return Xtrain,Xtest
def trainFillMean(preXtrain,fillMethod="mean"):
    '''
    allowed_fillMethod = ["mean", "median", "most_frequent", "constant"]
    '''
    #填充
    pipe = Pipeline(steps=[('missing_values', SimpleImputer(missing_values=np.nan, strategy=fillMethod, )),
                           ('minmax_scaler', MinMaxScaler())])
    Xtrain = pipe.fit_transform(preXtrain)
    return Xtrain

def delete70NaN(Xtrain,columnNumber = 40,threshold = 0.4,):
    '''
    forty column
    threshol 80%
    '''
    list = []
    Xtrain = pd.DataFrame(Xtrain)
    Xtrain.columns =[i for i in range(0,columnNumber)]
    for i in range(0, columnNumber):
        num = Xtrain[i].isna().sum()
        proportion = num / len(Xtrain[i])
        if proportion < threshold:
            list.append(i)
    afterTrain = Xtrain.loc[:, list]
    return afterTrain,list
def model(Xtrain,Ytrain,Xtest ,Ytest,kwargs ,savepred = None):
    '''
    allowed_modelname = [XGBClassifier,RandomForestClassifier]
    '''

    model_true = kwargs
    model_true.fit(Xtrain,Ytrain, )
    ypred = model_true.predict_proba(Xtest)[:,1]
    if savepred != None:
        # ypred = round(ypred, 4)
        # np.savetxt(saveYpred,ypred,fmt="%d")
        temp_ypred = ypred
        temp_ypred =  pd.DataFrame(temp_ypred)
        temp_ypred.to_csv(savepred,sep='\t',index = None)
    score = metricsScores(Ytest,ypred)
    return score

def fold(model,X_train, Y_train, cv = 10 ):
    metrics_list = []
    kf = StratifiedKFold(n_splits=cv, random_state=1, shuffle=True)
    for train_index, test_index in kf.split(X_train,Y_train):
        Xtrain_cross, Xtest_cross = X_train.loc[train_index], X_train.loc[test_index]
        Ytrain_cross, Ytest_cross = Y_train.loc[train_index], Y_train.loc[test_index]
        model.fit(Xtrain_cross, Ytrain_cross)
        y_pred = model.predict_proba(Xtest_cross)[:, 1]
        metrics_list.append(metricsScores(Ytest_cross, y_pred))
    metrics_list.append(np.mean(metrics_list, axis=0))
    metrics_list = pd.DataFrame(metrics_list)

    return metrics_list.iloc[-1]

def selectFeature(train,label,model):
    #特征重要性排序
    #可更改其他方式
    clf = ExtraTreesClassifier(random_state=1)
    clf = clf.fit(train, label)
    feature_order = clf.feature_importances_
    feature_order = pd.DataFrame(feature_order).T  # 0  0.116739  0.051316  0.395378  0.436567
    feature_order = feature_order.sort_values(by=0, axis=1,ascending=False)
    with open('./AAAA_featureSeletct', "a+", encoding='utf-8') as f:
        f.write('特征重要性排序')
        f.write(str(feature_order))
        f.write('\n')
    #贪心算法确定特征子集
    auc = 0
    importFeature = []
    for i in feature_order.columns:
        importFeature.append(i)
        train = pd.DataFrame(train)
        featureSet = train.loc[:,importFeature]
        CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)#8389058127314367
        lastAuc = auc
        print('lastAuc',lastAuc)
        auc = sk_model_selection.cross_val_score(model, featureSet, label,scoring='roc_auc', cv=CV, n_jobs=-1).mean()
        auc = round(auc,5)
        print('acc',auc)
        with open('./AAAA_featureSeletct',"a+",encoding='utf-8' ) as f:

            f.write('不同特征子集auc')
            f.write(str(importFeature))
            f.write('\t')
            f.write(str(auc))
            f.write('\n')
        if lastAuc-0.01 > auc:
            importFeature.remove(i)
            break
    return importFeature,auc
def plot_featureSet(auc_mean, metrics, plot_name='SFS'):
    import matplotlib.pyplot as plt
    from pathlib import Path


    fig = plt.figure(figsize=(16, 9))
    x = range(1, len(auc_mean) + 1)
    plt.plot(x, auc_mean, 'ro-', color='#4169E1', alpha=1, label=metrics)
    plt.xticks(np.arange(min(x), max(x) + 1, 3.0))
    plt.grid(ls='--')
    plt.legend()
    plt.title(plot_name, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.ylabel('Performance', fontsize=14)

    # 保存文件
    Path("./Plot").mkdir(exist_ok=True)
    plt.savefig(f"./Plot/{metrics}-{plot_name}.png", dpi=300, format="png")
    plt.show()
if __name__ == '__main__':
    # for itern in range(4,8):
        #各种参数
        #cross_meterics_index = ['RandomForestClassifier','AdaBoostClassifier','XGBClassifier','DecisionTreeClassifier','MLPClassifier','LogisticRegression']# 输出行名
        cross_meterics_index = ['RandomForestClassifier']# [AdaBoostClassifier(random_state=1), RandomForestClassifier(random_state=2), XGBClassifier(random_state=1)]
        #testnumber = 1
        predSavePath = '../pro_value/RF_pred'
        # test = 'COSMIC_test1'
        # test = 'ICGC_test2'
        mutationNumber = 6
        posPath = '../train/train_pos_closeby.vcf'
        negPath = '../train/train_neg_closeby.vcf'
        testNegPath = '../test/test_neg_closeby_process_cs_quchong_closeby_match_features.txt'
        testPosPath = '../test/test_pos_closeby_process_cs_quchong_closeby_match_features.txt'
        #mrmr选择的列名（特征选择后所需的列）
        ##mrmrCol = [5,18,23,25,24,31,22,21,26,33,37,35,38,32,36,17,34,20,0,19]  # 40列的列名
        #mrmrCol = [27, 2, 25, 28, 20, 22, 26, 7, 31, 29, 0, 30, 1, 19, 24, 23, 18]  # 非40列的列名,去除缺失值后列名
        mrmrCol = [33, 2, 31, 34, 20, 22, 32, 7, 39, 35, 0, 38, 1, 19, 24, 23, 18] #非40列的列名,去除缺失值
        #交叉验证折数
        cv = 10
        #去除缺失值的比例
        threshold = 0.4
        #数据读取
        X_train, Y_train, X_test, Y_test, mutationNumber = dataRead(posPath,negPath, testPosPath,testNegPath,mutationNumber,mutationNumber)
        #删除超70%缺失值的属性
        X_train,listcol = delete70NaN(X_train,threshold = threshold)
        X_test = pd.DataFrame(X_test)
        X_test = X_test.loc[:,listcol]
        #mrmr选择的列
        X_train = pd.DataFrame(X_train)
        X_train = X_train.loc[:,mrmrCol]
        X_test = X_test.loc[:,mrmrCol]
        #均值填充和归一化
        X_train,X_test=fillMeanMaxMin(X_train,X_test)
        # #输出均值填充结果
        # temp_save_data_train = pd.DataFrame(X_train)
        # temp_save_data_test = pd.DataFrame(X_test)
        # temp_save_Y_train  = pd.DataFrame(Y_train)
        # temp_save_Y_test  = pd.DataFrame(Y_test)
        # temp_all_data = pd.concat([temp_save_Y_train,temp_save_data_train],axis=1)
        # temp_all_data_test = pd.concat([temp_save_Y_test,temp_save_data_test],axis=1)
        # temp_all_data.to_csv('./allData-Train'+str(mutationNumber)+'.txt', sep=',',header=None,index=None)
        # temp_all_data_test.to_csv('./allData-test'+str(testnumber)+'.txt',sep=',',header=None,index=None)
        # 模型预测
        start = time.time()
        # 修改模型注意修改上边index的名称
        #modelList = [RandomForestClassifier(random_state=1),AdaBoostClassifier(random_state=1),XGBClassifier(random_state=1),DecisionTreeClassifier(random_state=1),MLPClassifier(random_state=1),LogisticRegression(random_state=1,max_iter =500)]# [AdaBoostClassifier(random_state=1), RandomForestClassifier(random_state=2), XGBClassifier(random_state=1)]
        modelList = [RandomForestClassifier(random_state=1)]# [AdaBoostClassifier(random_state=1), RandomForestClassifier(random_state=2), XGBClassifier(random_state=1)]
        # 注意决策树返回的数01标签值，而不是概率
        predictionScore = pd.DataFrame()
        cross_meterics_score = []

        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        Y_train = pd.DataFrame(Y_train)
        for i in modelList:
            temp_X_train = X_train
            temp_X_test = X_test
            #训练集
            meterics_cross = fold(i,temp_X_train,Y_train,cv)
            cross_meterics_score.append(meterics_cross)
            #测试集
            ypred = model(temp_X_train, Y_train, temp_X_test, Y_test,i,savepred = predSavePath)#numpy转化为pandas报错  先搁置 一会解决
            ypred = pd.DataFrame(ypred).T
            predictionScore = pd.concat([predictionScore, ypred], axis=0, )
        cross_meterics_score = pd.DataFrame(cross_meterics_score)
        # print( 'predictionScore',predictionScore)
        # print( 'cross_meterics_score',cross_meterics_score)
        predictionScore.columns =   ['Sen', 'Spe', 'Pre', 'F1', 'MCC', 'ACC', 'AUC', 'PRC', 'tn', 'fp', 'fn', 'tp', 'thres']
        cross_meterics_score.columns =   ['Sen', 'Spe', 'Pre', 'F1', 'MCC', 'ACC', 'AUC', 'PRC', 'tn', 'fp', 'fn', 'tp', 'thres']
        cross_meterics_score.index = cross_meterics_index
        predictionScore.index = cross_meterics_index
        print(predictionScore)
        predictionScore.to_csv('./test_Score.txt', sep='\t')
        cross_meterics_score.to_csv('./validScore.txt', sep='\t',columns =   ['Sen', 'Spe', 'Pre', 'F1', 'MCC', 'ACC', 'AUC', 'PRC', 'tn', 'fp', 'fn', 'tp', 'thres'])

        end = time.time()
        print('未使用进程池时间花费{}'.format((end - start)))  # 时间花费5.807056665420532






