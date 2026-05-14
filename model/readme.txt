***************Model******************
•	Convolution Neural Networks
•	Recurrent Neural Networks (RNN)
•	Deep Autoencoders (unsupervised learning)
•	GAN (2014)
•	Deep Forrest (2017)
•	Transformer (GPT3, 2020)


--------------------Problem---------------------------
*早停指标参考va_loss 还是va_macroF1？怎么选择
    在验证既用交易 proxy 指标来判断？
*unbalanced class

*tr_loss 训练集损失（training loss）
*va_loss 验证集损失（validation loss）
*va_macroF1 precision + recall 的调和平均,0 = 完全乱猜,1 = 完美分类.
    经验参考：
    随机分类（多分类）：≈ 0.2~0.25
    勉强有用：>0.3
    有一定信号：>0.35
    比较强：>0.45
过拟合：va_loss ≫ tr_loss，va_macroF1 没跟着 tr_loss 的下降而上升，模型学到的是 训练集的模式 / 噪声
将幅度信息部分反映在损失函数，尝试能否让模型学到更多信息
