import csv
import time
import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

File_SC = "./CoAQDsrc/附件1.csv"
File_SB = "./CoAQDsrc/附件2.csv"
File_TMAS = "./CoAQDsrc/TimeMatchAndSub.csv"
File_TMASaj = "./CoAQDsrc/TimeMatchAndSubaj.csv"
File_AJ = "./CoAQDsrc/adjust.csv"
Time_sc = 6
Time_sb = 11
TimeInterval = 5 * 60
countLimit = False  # 限制数据量，测试用
feature_cols = [U'风速', U'压强', U'降水量', U'温度', U'湿度']
Y = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']


def TimeMatchAndSub(File_clu, File_subed, File_sub):
    with open(File_clu, 'w', newline='') as f_sub:
        writer_sub = csv.writer(f_sub)
        with open(File_sub, 'r') as f_sb:
            reader_sb = csv.reader(f_sb)
            for row in reader_sb:
                writer_sub.writerow(row)
                break
        with open(File_subed, 'r') as f_sc:
            reader_sc = csv.reader(f_sc)
            next(reader_sc)  # 去掉表头
            count = 0
            OptimizeSP = 0
            mean = [0] * len(Y)
            for row_sc in reader_sc:
                TimeCol_sc = row_sc[Time_sc]  # 提取时间列
                timeStamp_sc = int(time.mktime(time.strptime(TimeCol_sc, "%Y/%m/%d %H:%M")))  # 转为时间戳
                with open(File_sub, 'r') as f_sb:
                    reader_sb = csv.reader(f_sb)
                    next(reader_sb)  # 去掉表头
                    for SP in range(OptimizeSP):  # 起始点优化
                        next(reader_sb)
                    for row_sb in reader_sb:
                        OptimizeSP += 1
                        TimeCol_sb = row_sb[Time_sb]  # 提取时间列
                        timeStamp_sb = int(time.mktime(time.strptime(TimeCol_sb, "%Y/%m/%d %H:%M")))  # 转为时间戳
                        if timeStamp_sb > timeStamp_sc + TimeInterval:  # 时间匹配失败优化
                            break
                        if (timeStamp_sb - TimeInterval) <= timeStamp_sc <= (timeStamp_sb + TimeInterval):  # 时间匹配
                            data = [float(row_sc[index]) - float(row_sb[index]) for index in range(len(Y))]
                            mean = [mean[im] + data[im] for im in range(len(Y))]
                            for ix in range(len(Y), len(row_sb)):  # 计算差值后合并
                                data.append(row_sb[ix])
                            print(count, row_sc[Time_sc])
                            print(count, row_sb[Time_sb])
                            writer_sub.writerow(data)
                            count += 1
                            break
                    if countLimit and count >= countLimit:
                        break
            for i in range(len(Y)):
                mean[i] = mean[i] / (count + 1)
                print(mean[i])


def LRmode(data, pare):
    from sympy.physics.quantum.tests.test_circuitplot import mpl
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 设置可用的字体
    mpl.rcParams['axes.unicode_minus'] = False
    sns.pairplot(data, x_vars=feature_cols, y_vars=pare, height=5, aspect=0.8, kind='reg')  # 数据可视化
    plt.show()


def predict(data, pare):
    print(pare)
    X = data[feature_cols]
    y = data[pare]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)  # 拆分2：1的训练集和测试集
    linreg = LinearRegression()  # 训练
    linreg.fit(X_train, y_train)
    print(linreg.intercept_)  # 基准系数
    List = zip(feature_cols, linreg.coef_)  # 特征量与相关系数对应zip
    for i in List:
        print(i)
    y_pred = linreg.predict(X_test)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_err = np.sqrt(sum_mean / int(y_test.shape[0]))  # 计算均方根误差
    print("RMSE ERR:", sum_err)
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel('the number of ' + pare)
    plt.ylabel('value of ' + pare)
    plt.show()
    return linreg.intercept_, linreg.coef_


def adjust(srcpath, tarpath, inte, coef):
    with open(tarpath, 'w', newline='') as f_tar:
        writer_aj = csv.writer(f_tar)
        with open(srcpath, 'r') as f_src:
            reader_sb = csv.reader(f_src)
            next(reader_sb)
            for row_sb in reader_sb:
                data = []
                for index in range(len(Y)):  # 结果list
                    ajcoe = inte[index]
                    for ix in range(len(feature_cols)):  # 特征系数list
                        ajcoe += float(coef[index][ix]) * float(row_sb[len(Y) + ix])  # 系数乘特征
                    data.append(float(row_sb[index]) + ajcoe)
                for ix in range(len(Y), len(row_sb)):  # 校准后合并
                    data.append(row_sb[ix])
                writer_aj.writerow(data)


if __name__ == '__main__':
    if not os.path.exists(File_TMAS):
        TimeMatchAndSub(File_TMAS, File_SC, File_SB)  # 时间匹配然后计算参数差值，IO耗时大，只需要运行一次
    dataTMAS = pd.read_csv(File_TMAS)
    inte = []
    coef = []
    for i in range(len(Y)):
        LRmode(dataTMAS, Y[i])
        one, two = predict(dataTMAS, Y[i])
        inte.append(one)
        coef.append(two)
    if not os.path.exists(File_AJ):
        adjust(File_SB, File_AJ, inte, coef)
        TimeMatchAndSub(File_TMASaj, File_SC, File_AJ)  # 时间匹配然后计算校准后的差值，IO耗时大
