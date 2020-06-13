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
Time_sc = 6
Time_sb = 11
TimeInterval = 5 * 60
countLimit = False  # 限制数据量，测试用
feature_cols = [U'风速', U'压强', U'降水量', U'温度', U'湿度']
Y = ['PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3']


def TimeMatchAndSub():
    with open(File_TMAS, 'w', newline='') as f_sub:
        writer_sub = csv.writer(f_sub)
        with open(File_SB, 'r') as f_sb:
            reader_sb = csv.reader(f_sb)
            for row in reader_sb:
                writer_sub.writerow(row)
                break
        with open(File_SC, 'r') as f_sc:
            reader_sc = csv.reader(f_sc)
            next(reader_sc)  # 去掉表头
            count = 0
            OptimizeSP = 0
            for row_sc in reader_sc:
                TimeCol_sc = row_sc[Time_sc]  # 提取时间列
                timeStamp_sc = int(time.mktime(time.strptime(TimeCol_sc, "%Y/%m/%d %H:%M")))  # 转为时间戳
                with open(File_SB, 'r') as f_sb:
                    reader_sb = csv.reader(f_sb)
                    next(reader_sb)  # 去掉表头
                    for i in range(OptimizeSP):  # 起始点优化
                        next(reader_sb)
                    for row_sb in reader_sb:
                        OptimizeSP += 1
                        TimeCol_sb = row_sb[Time_sb]  # 提取时间列
                        timeStamp_sb = int(time.mktime(time.strptime(TimeCol_sb, "%Y/%m/%d %H:%M")))  # 转为时间戳
                        if timeStamp_sb > timeStamp_sc + 2 * TimeInterval:  # 时间匹配失败优化
                            break
                        if (timeStamp_sb - TimeInterval) <= timeStamp_sc <= (timeStamp_sb + TimeInterval):  # 时间匹配
                            data = [float(row_sc[0]) - float(row_sb[0]), float(row_sc[1]) - float(row_sb[1]),
                                    float(row_sc[2]) - float(row_sb[2]),
                                    float(row_sc[3]) - float(row_sb[3]), float(row_sc[4]) - float(row_sb[4]),
                                    float(row_sc[5]) - float(row_sb[5])]
                            for i in range(6, len(row_sb)):  # 计算差值后合并
                                data.append(row_sb[i])
                            print(count, row_sc[Time_sc])
                            print(count, row_sb[Time_sb])
                            writer_sub.writerow(data)
                            count += 1
                            break
                    if countLimit and count >= countLimit:
                        break


def LRmode(data, pare):
    from sympy.physics.quantum.tests.test_circuitplot import mpl
    mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
    mpl.rcParams['axes.unicode_minus'] = False
    sns.pairplot(data, x_vars=feature_cols, y_vars=pare, size=3, aspect=0.8, kind='reg')
    plt.show()


def predict(data, pare):
    print(pare)
    X = data[feature_cols]
    y = data[pare]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    print(linreg.intercept_)
    List = zip(feature_cols, linreg.coef_)
    for i in List:
        print(i)
    y_pred = linreg.predict(X_test)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / int(y_test.shape[0]))
    print(sum_erro)


if __name__ == '__main__':
    if not os.path.exists(File_TMAS):
        TimeMatchAndSub()  # 时间匹配然后计算参数差值，IO耗时大，只需要运行一次
    data = pd.read_csv(File_TMAS)
    for i in Y:
        LRmode(data, i)
        predict(data, i)
