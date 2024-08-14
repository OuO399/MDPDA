# MDPDA

## 执行方式
对于传统分类器，不需要使用glove，使用dbn，深度分类器使用glove

因此传统分类器先用data_process_by_version.py处理数据，进行变异；再使用train_test_by_version.py进行dbn训练与预测，分出训练集与测试集；最后使用classify_by_version.py进行分类器的预测。

深度分类器则使用对应的glove文件即可。

---
## 文件夹
data_source  需要找到对应项目的具体代码文件放在其中，或者直接使用压缩包中的数据。

其中部分项目可以查看以下地址，寻找对应的tag版本或者自行寻找：

https://github.com/apache/ant

https://github.com/apache/camel

https://github.com/apache/xalan-java

GloVe  存放使用的GloVe模型

MDPDA/dbn  存放使用的DBN模型

PROMISE/promise_data  存放使用项目的数据集