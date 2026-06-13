\# SR-CNN Sock Shop Anomaly Detection



本项目复现了 SR-CNN 异常检测算法，并将其应用于 Sock Shop 微服务系统的 CPU 使用率异常检测任务。



\## 项目内容



\- 使用 Spectral Residual 提取时间序列异常特征

\- 使用一维 CNN 学习正常 CPU 使用率模式

\- 对网络延迟、Pod Kill、CPU Stress 等故障场景进行检测

\- 输出异常检测结果图



\## 文件说明



\- `sr\_cnn\_sockshop\_demo.ipynb`：完整训练与测试代码

\- `data/`：训练集和测试集

\- `models/`：训练好的 SR-CNN 模型权重

\- `results/`：实验输出图像



\## 运行方法



```bash

pip install -r requirements.txt

jupyter notebook

