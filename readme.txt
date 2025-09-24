Heriot:
 Heriot-Watt email: cx4005@hw.ac.uk
 Heriot-Watt user name: cx4005
 Heriot-Watt ID: H00511994
passwd: usual pw+ heriot
选课信息：https://myhwu.hw.ac.uk/HWSAS8/bwkkspgr.showpage?page=HW_STU_SUMMARY_MAIN
查看课表 https://curriculum.hw.ac.uk/programmedetails/F2Z7-ARI?termcode=202526&campuscode=1DU
查看单个课程的时间：https://timetable.hw.ac.uk/WebTimetables/LiveDU/default.aspx

worldquant iqc  
Numerai: og stock market data competition https://numer.ai/
Synth: volatility modelling on bittensor (approx $50,000 per week in rewards) https://www.synthdata.co/
NeurIPS: more ml focussed https://neurips.cc/
Kaggle: they had a great comp in the past with Jane Street https://www.kaggle.com/

Numerai Tournament
QuantConnect Alpha Competition
WorldQuant BRAIN

student id: H00511994

################################实施步骤########################################
BTC预测四分之一长度的走势 整张图形在4h级别   约360根蜡烛图组成，每根蜡烛图 1分钟！
1分钟级别的图形噪声太大，用5分钟作为基础时间预测，  5分钟间隔，38小时-456个蜡烛图(短一点可以选18小时)作为总图形，预测接下来1-2小时走势， 价格波动大约在 0.2%， 

问题：456个蜡烛图输入太多了，简化为15分钟级别，    152个蜡烛图预测接下来4-8个蜡烛图的走势

用15分钟图预测4小时趋势，  152 个蜡烛图(38小时)预测接下来16个蜡烛图的走势


特征选择：
特征选择
可以先只用原始价量（Open/High/Low/Close/Volume…），再逐步加入衍生特征（如过去 N 根收益、波动率、成交量 z-score、K 线形态统计），观察验证集 macro-F1 与混淆矩阵变化。

********
先学明白cnn，不然无法优化
******************************************************************************

| 指标                 | 含义                       | 解释                                                 |
| ------------------ | ------------------------ | -------------------------------------------------- |
| **precision（精确率）** | 预测为该类别的样本中，真实属于该类别的比例    | `precision = TP / (TP + FP)`<br>→ 预测的“准”           |
| **recall（召回率）**    | 该类别真实样本中，被模型正确预测出来的比例    | `recall = TP / (TP + FN)`<br>→ 预测的“全”              |
| **f1-score（F1分数）** | precision 与 recall 的调和平均 | $F1 = \frac{2\cdot P \cdot R}{P+R}$<br>→ 平衡“准”和“全” |
| **support**        | 该类别在测试集中的样本数量            | 样本基数，便于看类别规模                                       |

| 名称                     | 含义                                            | 说明                                 |
| ---------------------- | --------------------------------------------- | ---------------------------------- |
| **accuracy（准确率）**      | 全部样本中预测正确的比例                                  | $\frac{\text{所有正确预测}}{\text{总样本}}$ |
| **macro avg（宏平均）**     | 先在每个类别算 precision/recall/F1，再对类别**等权平均**      | 不考虑类别数量差别，适合看模型对小类是否公平             |
| **weighted avg（加权平均）** | 先在每类算 precision/recall/F1，再按 support（样本数）加权平均 | 样本多的类别影响更大，接近整体表现                  |

*********************************接下来的工作*****************************
优化模型(长期)
搭建回测(紧急)
尝试不同时间尺度
尝试不同品类
