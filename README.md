# RS_deepLearn
遥感分割论文中的代码
注意这里分为两组代码 ubuntu 与 windows两组，均从master分支拉取
2021 4.15 23:49
梳理存在的问题

#### 第三章
- [x] 基于RGB数据集原始模型的训练
- [ ] SLIC 超像素算法实现
- [ ] quick shift算法结构
- [ ] 利用超像素优化神经网络的输出
- [ ] 原始网络模型数据集的训练结果对比
- [ ] 原始超像素算法与新构建超像素算法之间的对比

#### 第四章
- [ ] 边界损失函数的构建问题
- [ ] 超像素信息与神经网络的融合结构(初步定名 superpixel-encoding-Net)
- [ ] 超像素网络结果与原始模型训练结果对比

#### 第五章
- [ ] 解耦结构的实现
- [ ] 解耦网络模型训练与之前模型训练结果对比

#### 需要训练的数据集与模型
|  数据集   | fcn  | unet| segnet | superPixel-encodingNet | SPE-decouple-Net|
|  ----  | ----  |  ----  |  ----  |  ----  |  ----  |
| RGB123  | ok | ok | model_err | designing | designing |
| RGB124  | ok | ok | model_err | designing | designing |
| RGB134  | ok | ok | model_err | designing | designing |
| RGB234  | ok | ok | model_er | designing | designing |
| 指数数据集  | training | training | model_er | designing | designing |
| allbands  | wait | wait | model_err | designing | designing |

