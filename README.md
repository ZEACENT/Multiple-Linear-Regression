# Multiple-Linear-Regression
### 多元线性回归 挖掘天气情况对空气质量侦测精度的影响
```
空气污染对生态环境和人类健康危害巨大，通过对“两尘四气”（PM2.5、PM10、CO、NO2、SO2、O3）浓度的实时监测可以及时掌握空气质量，对污染源采取相应措施。虽然国家监测控制站点（国控点）对“两尘四气”有监测数据，且较为准确，但因为国控点的布控较少，数据发布时间滞后较长且花费较大，无法给出实时空气质量的监测和预报。某公司自主研发的微型空气质量检测仪花费小，可对某一地区空气质量进行实时网格化监控，并同时监测温度、湿度、风速、气压、降水等气象参数。
由于所使用的电化学气体传感器在长时间使用后会产生一定的零点漂移和量程漂移，非常规气态污染物（气）浓度变化对传感器存在交叉干扰，以及天气因素对传感器的影响，在国控点近邻所布控的自建点上，同一时间微型空气质量检测仪所采集的数据与该国控点的数据值存在一定的差异，因此，需要利用国控点每小时的数据对国控点近邻的自建点数据进行校准。
```
附件 1.CSV 和附件 2.CSV 分别提供了一段时间内某个国控点每小时的数据和该国控点近邻的一个自建点数据（相应于国控点时间且间隔在 5 分钟内），各变量单位见附件3。请建立数学模型研究下列问题：
1. 对自建点数据与国控点数据进行探索性数据分析。
2. 对导致自建点数据与国控点数据造成差异的因素进行分析。
3. 利用国控点数据，建立数学模型对自建点数据进行校准。
