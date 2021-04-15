# RS_deepLearn
遥感分割论文中的代码
注意这里分为两组代码 ubuntu 与 windows两组，均从master分支拉取
2021 4.15 23:49
梳理存在的问题
第三章部分存在的问题
+ SLIC 超像素算法结构
+ quick shift算法结构
+ 探索如何利用超像素优化神经网络的输出
第四章的关键问题
+ 边界损失函数的构建问题
+ 超像素信息与神经网络的融合结构
第五章的关键问题
+ 解耦结构的实现

需要训练的数据集与模型
|  数据集   | fcn  | unet| segnet | superPixel-encodingNet | SPE-decouple-Net|
|  ----  | ----  |  ----  |  ----  |  ----  |  ----  |
| RGB123  | ok | ok | model_err | designing | designing |
| RGB124  | ok | ok | model_err | designing | designing |
| RGB134  | ok | ok | model_err | designing | designing |
| RGB234  | ok | ok | model_er | designing | designing |
| 指数数据集  | training | training | model_er | designing | designing |
| allbands  | wait | wait | model_err | designing | designing |

- [ ] Mercury
- [x] Venus
- [x] Earth (Orbit/Moon)
- [x] Mars
- [ ] Jupiter
