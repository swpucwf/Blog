[TOC]



# OCR

## 1.简介

### 1.1 OCR技术难点

![img](https://ai-studio-static-online.cdn.bcebos.com/a56831fbf0c449fe9156a893002cadfe110ccfea835b4d90854a7ce4b1df2a4f)

1. **海量数据要求OCR能够实时处理。** 
2. **端侧应用要求OCR模型足够轻量，识别速度足够快。**

### 1.2. OCR前沿算法

#### 1.2.1 文本检测

首先，**文本检测的任务是定位出输入图像中的文字区域**。

1. 一类方法将文本检测视为目标检测中的一个特定场景，基于通用目标检测算法进行改进适配，如TextBoxes基于一阶段目标检测器SSD算法，调整目标框使之适合极端长宽比的文本行，CTPN则是基于Faster RCNN架构改进而来。

2. 考虑文本检测与目标检测在目标信息以及任务本身区别，如文本一般长宽比较大，往往呈“条状”，文本行之间可能比较密集，弯曲文本等，又衍生了很多专用于文本检测的算法，如EAST、PSENet、DBNet等等。

算法分类：**基于回归**和**基于分割**的两大类文本检测算法

1. 基于回归：规则形状文本检测效果较好，但是对不规则形状的文本检测效果会相对差一些。
2. 基于分割：在各种场景、对各种形状文本的检测效果都可以达到一个更高的水平，但缺点就是后处理一般会比较复杂，因此常常存在速度问题，并且无法解决重叠文本的检测问题。

![img](https://ai-studio-static-online.cdn.bcebos.com/4f4ea65578384900909efff93d0b7386e86ece144d8c4677b7bc94b4f0337cfb)

![image-20240407231308383](https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/imagesimage-20240407231308383.png)

#### 1.2.2 文本识别

任务：**识别出图像中的文字内容**，一般输入来自于文本检测得到的文本框截取出的图像文字区域。

文本识别一般可以根据待识别文本形状分为**规则文本识别**和**不规则文本识别**两大类。

1. 规则文本主要指印刷字体、扫描文本等，文本大致处在水平线位置；不规则文本往往不在水平位置，存在弯曲、遮挡、模糊等问题。

2. 不规则文本场景具有很大的挑战性，也是目前文本识别领域的主要研究方向。

规则文本识别的算法根据解码方式的不同可以大致分为基于CTC和Sequence2Sequence两种，将网络学习到的序列特征 转化为 最终的识别结果 的处理方式不同。基于CTC的算法以经典的CRNN为代表。

![img](https://ai-studio-static-online.cdn.bcebos.com/403ca85c59d344f88d3b1229ca14b1e90c5c73c9f1d248b7aa94103f9d0af597)

不规则文本的识别算法：

- 如STAR-Net等方法通过加入TPS等矫正模块，将不规则文本矫正为规则的矩形后再进行识别；
- RARE等基于Attention的方法增强了对序列之间各部分相关性的关注；
- 基于分割的方法将文本行的各字符作为独立个体，相比与对整个文本行做矫正后识别，识别分割出的单个字符更加容易；
- 基于Transformer的文本识别算法

![img](https://ai-studio-static-online.cdn.bcebos.com/0fa30c3789424473ad9be1c87a4f742c1db69e3defb64651906e5334ed9571a8)

#### 1.2.3 文档结构化识别

获取的往往是结构化的信息，如身份证、发票的信息格式化抽取，表格的结构化识别等等，多在快递单据抽取、合同内容比对、金融保理单信息比对、物流业单据识别等场景下应用。

OCR结果+后处理是一种常用的结构化方案，但流程往往比较复杂，并且后处理需要精细设计，泛化性也比较差。在OCR技术逐渐成熟、结构化信息抽取需求日益旺盛的背景下，版面分析、表格识别、关键信息提取等关于智能文档分析

- **版面分析**

版面分析（Layout Analysis）主要是对文档图像进行内容分类，类别一般可分为纯文本、标题、表格、图片等。现有方法一般将文档中不同的板式当做不同的目标进行检测或分割，如Soto Carlos[16]在目标检测算法Faster R-CNN的基础上，结合上下文信息并利用文档内容的固有位置信息来提高区域检测性能；Sarkar Mausoom[17]等人提出了一种基于先验的分割机制，在非常高的分辨率的图像上训练文档分割模型，解决了过度缩小原始图像导致的密集区域不同结构无法区分进而合并的问题。

![img](https://ai-studio-static-online.cdn.bcebos.com/dedb212e8972497998685ff51af7bfe03fdea57f6acd450281ad100807086e1a)

- **表格识别**

表格识别（Table Recognition）的任务就是将文档里的表格信息进行识别和转换到excel文件中。文本图像中表格种类和样式复杂多样，例如不同的行列合并，不同的内容文本类型等，除此之外文档的样式和拍摄时的光照环境等都为表格识别带来了极大的挑战。这些挑战使得表格识别一直是文档理解领域的研究难点。

表格识别的方法种类较为丰富，早期的基于启发式规则的传统算法，如Kieninger等人提出的T-Rect等算法，一般通过人工设计规则，连通域检测分析处理；近年来随着深度学习的发展，

开始涌现一些基于CNN的表格结构识别算法，如Siddiqui Shoaib Ahmed等人提出的DeepTabStR，Raja Sachin等人提出的TabStruct-Net等；

此外，随着图神经网络（Graph Neural Network）的兴起，也有一些研究者尝试将图神经网络应用到表格结构识别问题上，基于图神经网络，将表格识别看作图重建问题，如Xue Wenyuan等人提出的TGRNet；

基于端到端的方法直接使用网络完成表格结构的HTML表示输出，端到端的方法大多采用Seq2Seq方法来完成表格结构的预测，如一些基于Attention或Transformer的方法，如TableMaster[22

![img](https://ai-studio-static-online.cdn.bcebos.com/22ca5749441441e69dc0eaeb670832a5d0ae0ce522f34731be7d609a2d36e8c1)

![img](https://ai-studio-static-online.cdn.bcebos.com/a9a3c91898c84f03b382583859526c4b451ace862dbc4a15838f5dde4d0ea657)

- **关键信息提取**

关键信息提取（Key Information Extraction，KIE）是Document VQA中的一个重要任务，主要从图像中提取所需要的关键信息，如从身份证中提取出姓名和公民身份号码信息，这类信息的种类往往在特定任务下是固定的，但是在不同任务间是不同的。

KIE通常分为两个子任务进行研究：

- SER: 语义实体识别 (Semantic Entity Recognition)，对每一个检测到的文本进行分类，如将其分为姓名，身份证。如下图中的黑色框和红色框。
- RE: 关系抽取 (Relation Extraction)，对每一个检测到的文本进行分类，如将其分为问题和的答案。然后对每一个问题找到对应的答案。如下图中的红色框和黑色框分别代表问题和答案，黄色线代表问题和答案之间的对应关系。

一般的KIE方法基于命名实体识别(Named Entity Recognition,NER)来研究，但是这类方法只利用了图像中的文本信息，缺少对视觉和结构信息的使用，因此精度不高。在此基础上，近几年的方法都开始将视觉和结构信息与文本信息融合到一起，按照对多模态信息进行融合时所采用的的原理可以将这些方法分为下面四种：

- 基于Grid的方法
- 基于Token的方法
- 基于GCN的方法
- 基于End to End 的方法

## 2. OCR文本检测

## 2.1 文本检测

文本在图像中的表现形式可以视为一种‘目标‘，通用的目标检测的方法也适用于文本检测，从任务本身上来看：

- 目标检测：给定图像或者视频，找出目标的位置（box），并给出目标的类别；
- 文本检测：给定输入图像或者视频，找出文本的区域，可以是单字符位置或者整个文本行位置；

![img](https://ai-studio-static-online.cdn.bcebos.com/af2d8eca913a4d5a968945ae6cac180b009c6cc94abc43bfbaf1ba6a3de98125)

目标检测和文本检测同属于“定位”问题。但是文本检测无需对目标分类，并且文本形状复杂多样。

当前所说的文本检测一般是自然场景文本检测，其难点在于：

1. 自然场景中文本具有多样性：文本检测受到文字颜色、大小、字体、形状、方向、语言、以及文本长度的影响；
2. 复杂的背景和干扰；文本检测受到图像失真，模糊，低分辨率，阴影，亮度等因素的影响；
3. 文本密集甚至重叠会影响文字的检测；
4. 文字存在局部一致性，文本行的一小部分，也可视为是独立的文本；

![img](https://ai-studio-static-online.cdn.bcebos.com/072f208f2aff47e886cf2cf1378e23c648356686cf1349c799b42f662d8ced00)

针对以上问题，衍生了很多基于深度学习的文本检测算法，解决自然场景文字检测问题，这些方法可以分为基于回归和基于分割的文本检测方法。

## 2.2 文本检测方法介绍

近些年来基于深度学习的文本检测算法层出不穷，这些方法大致可以分为两类：

1. 基于回归的文本检测方法
2. 基于分割的文本检测方法

![img](https://ai-studio-static-online.cdn.bcebos.com/22314238b70b486f942701107ffddca48b87235a473c4d8db05b317f132daea0)

### 2.1 基于回归的文本检测

基于回归文本检测方法和目标检测算法的方法相似，文本检测方法只有两个类别，图像中的文本视为待检测的目标，其余部分视为背景。

#### 2.1.1 水平文本检测

早期基于深度学习的文本检测算法是从目标检测的方法改进而来，支持水平文本检测。比如Textbox算法基于SSD算法改进而来，CTPN根据二阶段目标检测Fast-RCNN算法改进而来。

在TextBoxes算法根据一阶段目标检测器SSD调整，将默认文本框更改为适应文本方向和宽高比的规格的四边形，提供了一种端对端训练的文字检测方法，并且无需复杂的后处理。

- 采用更大长宽比的预选框
- 卷积核从3x3变成了1x5，更适合长文本检测
- 采用多尺度输入

![img](https://ai-studio-static-online.cdn.bcebos.com/3864ccf9d009467cbc04225daef0eb562ac0c8c36f9b4f5eab036c319e5f05e7)

CTPN基于Fast-RCNN算法，扩展RPN模块并且设计了基于CRNN的模块让整个网络从卷积特征中检测到文本序列，二阶段的方法通过ROI Pooling获得了更准确的特征定位。但是TextBoxes和CTPN只支持检测横向文本。



## 3 . 文本识别

### 3.1 背景介绍

- 规则文本识别：主要指印刷字体、扫描文本等，认为文本大致处在水平线位置
- 不规则文本识别： 往往出现在自然场景中，且由于文本曲率、方向、变形等方面差异巨大，文字往往不在水平位置，存在弯曲、遮挡、模糊等问题

### 3.2 文本识别算法分类

在传统的文本识别方法中，任务分为3个步骤，即**图像预处理、字符分割和字符识别**。需要对特定场景进行建模，一旦场景变化就会失效。面对复杂的文字背景和场景变动，基于深度学习的方法具有更优的表现。

![img](https://ai-studio-static-online.cdn.bcebos.com/4d0aada261064031a16816b39a37f2ff6af70dbb57004cb7a106ae6485f14684)

多数现有的识别算法可用如下统一框架表示，算法流程被划分为4个阶段：

![img](https://ai-studio-static-online.cdn.bcebos.com/a2750f4170864f69a3af36fc13db7b606d851f2f467d43cea6fbf3521e65450f)

| 算法类别    | 主要思路                              | 主要论文                        |
| :---------- | :------------------------------------ | :------------------------------ |
| 传统算法    | 滑动窗口、字符提取、动态规划          | -                               |
| ctc         | 基于ctc的方法，序列不对齐，更快速识别 | CRNN, Rosetta                   |
| Attention   | 基于attention的方法，应用于非常规文本 | RARE, DAN, PREN                 |
| Transformer | 基于transformer的方法                 | SRN, NRTR, Master, ABINet       |
| 校正        | 校正模块学习文本边界并校正成水平方向  | RARE, ASTER, SAR                |
| 分割        | 基于分割的方法，提取字符位置再做分类  | Text Scanner， Mask TextSpotter |

### 3.3 规则文本识别

文本识别的主流算法有两种，分别是基于 CTC (Conectionist Temporal Classification) 的算法和 Sequence2Sequence 算法，区别主要在解码阶段。

基于 CTC 的算法是将编码产生的序列接入 CTC 进行解码；基于 Sequence2Sequence 的方法则是把序列接入循环神经网络(Recurrent Neural Network, RNN)模块进行循环解码，两种方式都验证有效也是主流的两大做法。

![img](https://ai-studio-static-online.cdn.bcebos.com/f64eee66e4a6426f934c1befc3b138629324cf7360c74f72bd6cf3c0de9d49bd)

#### 3.3.1 基于CTC的算法

基于 CTC 最典型的算法是CRNN (Convolutional Recurrent Neural Network)[1]，它的特征提取部分使用主流的卷积结构，常用的有ResNet、MobileNet、VGG等。由于文本识别任务的特殊性，输入数据中存在大量的上下文信息，卷积神经网络的卷积核特性使其更关注于局部信息，缺乏长依赖的建模能力，因此仅使用卷积网络很难挖掘到文本之间的上下文联系。为了解决这一问题，CRNN文本识别算法引入了双向 LSTM(Long Short-Term Memory) 用来增强上下文建模，通过实验证明双向LSTM模块可以有效的提取出图片中的上下文信息。最终将输出的特征序列输入到CTC模块，直接解码序列结果。该结构被验证有效，并广泛应用在文本识别任务中。Rosetta是FaceBook提出的识别网络，由全卷积模型和CTC组成。Gao Y等人使用CNN卷积替代LSTM，参数更少，性能提升精度持平。

![img](https://ai-studio-static-online.cdn.bcebos.com/d3c96dd9e9794fddb12fa16f926abdd3485194f0a2b749e792e436037490899b)

#### 3.3.2 Sequence2Sequence 算法

​	Sequence2Sequence 算法是由编码器 Encoder 把所有的输入序列都编码成一个统一的语义向量，然后再由解码器Decoder解码。在解码器Decoder解码的过程中，不断地将前一个时刻的输出作为后一个时刻的输入，循环解码，直到输出停止符为止。一般编码器是一个RNN，对于每个输入的词，编码器输出向量和隐藏状态，并将隐藏状态用于下一个输入的单词，循环得到语义向量；解码器是另一个RNN，它接收编码器输出向量并输出一系列字以创建转换。受到 Sequence2Sequence 在翻译领域的启发， Shi提出了一种基于注意的编解码框架来识别文本,通过这种方式，rnn能够从训练数据中学习隐藏在字符串中的字符级语言模型。

![img](https://ai-studio-static-online.cdn.bcebos.com/f575333696b7438d919975dc218e61ccda1305b638c5497f92b46a7ec3b85243)

### 3.4 不规则文本识别

- 不规则文本识别算法可以被分为4大类：**基于校正的方法；基于 Attention 的方法；基于分割的方法；基于 Transformer 的方法。**

#### 3.4.1 基于校正的方法

基于校正的方法利用一些视觉变换模块，将非规则的文本尽量转换为规则文本，然后使用常规方法进行识别。

RARE模型首先提出了对不规则文本的校正方案，整个网络分为两个主要部分：一个空间变换网络STN(Spatial Transformer Network) 和一个基于Sequence2Squence的识别网络。其中STN就是校正模块，不规则文本图像进入STN，通过TPS(Thin-Plate-Spline)变换成一个水平方向的图像，该变换可以一定程度上校正弯曲、透射变换的文本，校正后送入序列识别网络进行解码。

![img](https://ai-studio-static-online.cdn.bcebos.com/66406f89507245e8a57969b9bed26bfe0227a8cf17a84873902dd4a464b97bb5)

RARE论文指出，该方法在不规则文本数据集上有较大的优势，特别比较了CUTE80和SVTP这两个数据集，相较CRNN高出5个百分点以上，证明了校正模块的有效性。基于此[6]同样结合了空间变换网络(STN)和基于注意的序列识别网络的文本识别系统。基于校正的方法有较好的迁移性，除了RARE这类基于Attention的方法外，STAR-Net[5]将校正模块应用到基于CTC的算法上，相比传统CRNN也有很好的提升。

#### 3.4.2 基于Attention的方法

​	基于 Attention 的方法主要关注的是序列之间各部分的相关性，该方法最早在机器翻译领域提出，认为在文本翻译的过程中当前词的结果主要由某几个单词影响的，因此需要给有决定性的单词更大的权重。在文本识别领域也是如此，将编码后的序列解码时，每一步都选择恰当的context来生成下一个状态，这样有利于得到更准确的结果。R^2AM 首次将 Attention 引入文本识别领域，该模型首先将输入图像通过递归卷积层提取编码后的图像特征，然后利用隐式学习到的字符级语言统计信息通过递归神经网络解码输出字符。在解码过程中引入了Attention 机制实现了软特征选择，以更好地利用图像特征，这一有选择性的处理方式更符合人类的直觉。

![img](https://ai-studio-static-online.cdn.bcebos.com/a64ef10d4082422c8ac81dcda4ab75bf1db285d6b5fd462a8f309240445654d5)

后续有大量算法在Attention领域进行探索和更新，例如SAR[8]将1D attention拓展到2D attention上，校正模块提到的RARE也是基于Attention的方法。实验证明基于Attention的方法相比CTC的方法有很好的精度提升。

![img](https://ai-studio-static-online.cdn.bcebos.com/4e2507fb58d94ec7a9b4d17151a986c84c5053114e05440cb1e7df423d32cb02)

#### 3.4.3 基于分割的方法

​	基于分割的方法是将文本行的各字符作为独立个体，相比与对整个文本行做矫正后识别，识别分割出的单个字符更加容易。它试图从输入的文本图像中定位每个字符的位置，并应用字符分类器来获得这些识别结果，将复杂的全局问题简化成了局部问题解决，在不规则文本场景下有比较不错的效果。然而这种方法需要字符级别的标注，数据获取上存在一定的难度。Lyu等人提出了一种用于单词识别的实例分词模型，该模型在其识别部分使用了基于 FCN(Fully Convolutional Network) 的方法。从二维角度考虑文本识别问题，设计了一个字符注意FCN来解决文本识别问题，当文本弯曲或严重扭曲时，该方法对规则文本和非规则文本都具有较优的定位结果。

![img](https://ai-studio-static-online.cdn.bcebos.com/fd3e8ef0d6ce4249b01c072de31297ca5d02fc84649846388f890163b624ff10)

#### 2.2.4 基于Transformer的方法

​	随着 Transformer 的快速发展，分类和检测领域都验证了 Transformer 在视觉任务中的有效性。如规则文本识别部分所说，CNN在长依赖建模上存在局限性，Transformer 结构恰好解决了这一问题，它可以在特征提取器中关注全局信息，并且可以替换额外的上下文建模模块（LSTM）。

​	一部分文本识别算法使用 Transformer 的 Encoder 结构和卷积共同提取序列特征，Encoder 由多个 MultiHeadAttentionLayer 和 Positionwise Feedforward Layer 堆叠而成的block组成。MulitHeadAttention 中的 self-attention 利用矩阵乘法模拟了RNN的时序计算，打破了RNN中时序长时依赖的障碍。也有一部分算法使用 Transformer 的 Decoder 模块解码，相比传统RNN可获得更强的语义信息，同时并行计算具有更高的效率。

​	SRN 算法将Transformer的Encoder模块接在ResNet50后，增强了2D视觉特征。并提出了一个并行注意力模块，将读取顺序用作查询，使得计算与时间无关，最终并行输出所有时间步长的对齐视觉特征。此外SRN还利用Transformer的Eecoder作为语义模块，将图片的视觉信息和语义信息做融合，在遮挡、模糊等不规则文本上有较大的收益。

​	NRTR 使用了完整的Transformer结构对输入图片进行编码和解码，只使用了简单的几个卷积层做高层特征提取，在文本识别上验证了Transformer结构的有效性。

![img](https://ai-studio-static-online.cdn.bcebos.com/e7859f4469a842f0bd450e7e793a679d6e828007544241d09785c9b4ea2424a2)

## 4.系统与优化策略简介

- 检测模型优化: (1) 采用 CML 协同互学习知识蒸馏策略；(2) CopyPaste 数据增广策略；
- 识别模型优化: (1) PP-LCNet 轻量级骨干网络；(2) U-DML 改进知识蒸馏策略； (3) Enhanced CTC loss 损失函数改进。

优化点：骨干网络、特征金字塔网络、头部结构、学习率策略、模型裁剪

### 4.1 轻量化模型

应选择轻量的骨干网络。随着图像分类技术的发展，MobileNetV1、MobileNetV2、MobileNetV3和ShuffleNetV2系列常用作轻量骨干网络。每个系列都有不同的模型大小和性能表现。[PaddeClas](https://github.com/PaddlePaddle/PaddleClas)提供了20多种轻量级骨干网络。

![img](https://ai-studio-static-online.cdn.bcebos.com/d3855eac989542d49e5dd69e2f09de284ec02fd2c3314f8b9db7491630e0cd14)

### 4.2 学习率策略优化



- Cosine 学习率下降策略

梯度下降算法需要我们设置一个值，用来控制权重更新幅度，我们将其称之为学习率。它是控制模型学习速度的超参数。学习率越小，loss的变化越慢。虽然使用较低的学习速率可以确保不会错过任何局部极小值，但这也意味着模型收敛速度较慢。

因此，在训练前期，权重处于随机初始化状态，我们可以设置一个相对较大的学习速率以加快收敛速度。在训练后期，权重接近最优值，使用相对较小的学习率可以防止模型在收敛的过程中发生震荡。

Cosine学习率策略也就应运而生，Cosine学习率策略指的是学习率在训练的过程中，按照余弦的曲线变化。在整个训练过程中，Cosine学习率衰减策略使得在网络在训练初期保持了较大的学习速率，在后期学习率会逐渐衰减至0，其收敛速度相对较慢，但最终收敛精度较好。

- 学习率预热

模型刚开始训练时，模型权重是随机初始化的，此时若选择一个较大的学习率,可能造成模型训练不稳定的问题，因此**学习率预热**的概念被提出，用于解决模型训练初期不收敛的问题。

学习率预热指的是将学习率从一个很小的值开始，逐步增加到初始较大的学习率。它可以保证模型在训练初期的稳定性。使用学习率预热策略有助于提高图像分类任务的准确性。



### 4.3 模型裁剪策略-FPGM

深度学习模型中一般有比较多的参数冗余，我们可以使用一些方法，去除模型中比较冗余的地方，从而提升模型推理效率。

模型裁剪指的是通过去除网络中冗余的通道（channel）、滤波器（filter）、神经元（neuron）等，来得到一个更轻量的网络，同时尽可能保证模型精度。

相比于裁剪通道或者特征图的方法，裁剪滤波器的方法可以得到更加规则的模型，因此减少内存消耗，加速模型推理过程。

之前的裁剪滤波器的方法大多基于范数进行裁剪，即，认为范数较小的滤波器重要程度较小，但是这种方法要求存在的滤波器的最小范数应该趋近于0，否则我们难以去除。

针对上面的问题，基于**几何中心点的裁剪算法**(Filter Pruning via Geometric Median, FPGM)被提出。FPGM将卷积层中的每个滤波器都作为欧几里德空间中的一个点，它引入了几何中位数这样一个概念，即**与所有采样点距离之和最小的点**。如果一个滤波器的接近这个几何中位数，那我们可以认为这个滤波器的信息和其他滤波器重合，可以去掉。

1. 模型裁剪需要重新训练模型，可以参考[PaddleOCR剪枝教程](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/deploy/slim/prune/README.md)。
2. 如果您需要对自己的模型进行剪枝，需要重新分析模型结构、参数的敏感度，我们通常情况下只建议裁剪相对敏感度低的参数，而跳过敏感度高的参数。
3. 每个卷积层的剪枝率对于裁剪后模型的性能也很重要，用完全相同的裁剪率去进行模型裁剪通常会导致显着的性能下降。
4. 模型裁剪不是一蹴而就的，需要进行反复的实验，才能得到符合要求的模型。

### 4.4 数据增强

常用的数据增强包括旋转、透视失真变换、运动模糊变换和高斯噪声变换等

AutoAugment (Cubuk et al. 2019), RandAugment (Cubuk et al. 2020), CutOut (DeVries and Taylor 2017), RandErasing (Zhong et al. 2020), HideAndSeek (Singh and Lee 2017), GridMask (Chen 2020), Mixup (Zhang et al. 2017) 和 Cutmix (Yun et al. 2019)。

这些数据增广大体分为3个类别：

（1）图像变换类：AutoAugment、RandAugment

（2）图像裁剪类：CutOut、RandErasing、HideAndSeek、GridMask

（3）图像混叠类：Mixup、Cutmix

### 4.5 输入分辨率优化

一般来说，当图像的输入分辨率提高时，精度也会提高。由于方向分类器的骨干网络参数量很小，即使提高了分辨率也不会导致推理时间的明显增加。我们将方向分类器的输入图像尺度从`3x32x100`增加到`3x48x192`，方向分类器的精度从`92.1%`提升至`94.0%`，但是预测耗时仅仅从`3.19ms`提升至`3.21ms`。

### 4.6 模型量化策略-PACT

模型量化是一种将浮点计算转成低比特定点计算的技术，可以使神经网络模型具有更低的延迟、更小的体积以及更低的计算功耗。

模型量化主要分为离线量化和在线量化。其中，离线量化是指一种利用KL散度等方法来确定量化参数的定点量化方法，量化后不需要再次训练；在线量化是指在训练过程中确定量化参数，相比离线量化模式，它的精度损失更小。

PACT(PArameterized Clipping acTivation)是一种新的在线量化方法，可以**提前从激活层中去除一些极端值**。在去除极端值后，模型可以学习更合适的量化参数。

PaddleOCR中提供了适用于PP-OCR套件的量化脚本。具体链接可以参考[PaddleOCR模型量化教程](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/deploy/slim/quantization/README.md)。



## 5. 文本检测训练相关FAQ

**1.1 PaddleOCR提供的文本检测算法包括哪些？**

**A**：PaddleOCR中包含多种文本检测模型，包括基于回归的文本检测方法EAST、SAST，和基于分割的文本检测方法DB，PSENet。


**1.2：请问PaddleOCR项目中的中文超轻量和通用模型用了哪些数据集？训练多少样本，gpu什么配置，跑了多少个epoch，大概跑了多久？**

**A**：对于超轻量DB检测模型，训练数据包括开源数据集lsvt，rctw，CASIA，CCPD，MSRA，MLT，BornDigit，iflytek，SROIE和合成的数据集等，总数据量越10W，数据集分为5个部分，训练时采用随机采样策略，在4卡V100GPU上约训练500epoch，耗时3天。


**1.3 文本检测训练标签是否需要具体文本标注，标签中的”###”是什么意思？**

**A**：文本检测训练只需要文本区域的坐标即可，标注可以是四点或者十四点，按照左上，右上，右下，左下的顺序排列。PaddleOCR提供的标签文件中包含文本字段，对于文本区域文字不清晰会使用###代替。训练检测模型时，不会用到标签中的文本字段。

**1.4 对于文本行较紧密的情况下训练的文本检测模型效果较差？**

**A**：使用基于分割的方法，如DB，检测密集文本行时，最好收集一批数据进行训练，并且在训练时，并将生成二值图像的[shrink_ratio](https://github.com/PaddlePaddle/PaddleOCR/blob/8b656a3e13631dfb1ac21d2095d4d4a4993ef710/ppocr/data/imaug/make_shrink_map.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L37)参数调小一些。另外，在预测的时候，可以适当减小[unclip_ratio](https://github.com/PaddlePaddle/PaddleOCR/blob/8b656a3e13631dfb1ac21d2095d4d4a4993ef710/configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L59)参数，unclip_ratio参数值越大检测框就越大。


**1.5 对于一些尺寸较大的文档类图片， DB在检测时会有较多的漏检，怎么避免这种漏检的问题呢？**

**A**：首先，需要确定是模型没有训练好的问题还是预测时处理的问题。如果是模型没有训练好，建议多加一些数据进行训练，或者在训练的时候多加一些数据增强。
如果是预测图像过大的问题，可以增大预测时输入的最长边设置参数[det_limit_side_len](https://github.com/PaddlePaddle/PaddleOCR/blob/8b656a3e13631dfb1ac21d2095d4d4a4993ef710/tools/infer/utility.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L47)，默认为960。其次，可以通过可视化后处理的分割图观察漏检的文字是否有分割结果，如果没有分割结果，说明是模型没有训练好。如果有完整的分割区域，说明是预测后处理的问题，建议调整[DB后处理参数](https://github.com/PaddlePaddle/PaddleOCR/blob/8b656a3e13631dfb1ac21d2095d4d4a4993ef710/tools/infer/utility.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L51-L53)。


**1.6  DB模型弯曲文本（如略微形变的文档图像）漏检问题?**

**A**: DB后处理中计算文本框平均得分时，是求rectangle区域的平均分数，容易造成弯曲文本漏检，已新增求polygon区域的平均分数，会更准确，但速度有所降低，可按需选择，在相关pr中可查看[可视化对比效果](https://github.com/PaddlePaddle/PaddleOCR/pull/2604)。该功能通过参数 [det_db_score_mode](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/tools/infer/utility.py#L51)进行选择，参数值可选[`fast`(默认)、`slow`]，`fast`对应原始的rectangle方式，`slow`对应polygon方式。感谢用户[buptlihang](https://github.com/buptlihang)提[pr](https://github.com/PaddlePaddle/PaddleOCR/pull/2574)帮助解决该问题。


**1.7 简单的对于精度要求不高的OCR任务，数据集需要准备多少张呢？**

**A**：（1）训练数据的数量和需要解决问题的复杂度有关系。难度越大，精度要求越高，则数据集需求越大，而且一般情况实际中的训练数据越多效果越好。

（2）对于精度要求不高的场景，检测任务和识别任务需要的数据量是不一样的。对于检测任务，500张图像可以保证基本的检测效果。对于识别任务，需要保证识别字典中每个字符出现在不同场景的行文本图像数目需要大于200张（举例，如果有字典中有5个字，每个字都需要出现在200张图片以上，那么最少要求的图像数量应该在200-1000张之间），这样可以保证基本的识别效果。


**1.8 当训练数据量少时，如何获取更多的数据？**

**A**：当训练数据量少时，可以尝试以下三种方式获取更多的数据：（1）人工采集更多的训练数据，最直接也是最有效的方式。（2）基于PIL和opencv基本图像处理或者变换。例如PIL中ImageFont, Image, ImageDraw三个模块将文字写到背景中，opencv的旋转仿射变换，高斯滤波等。（3）利用数据生成算法合成数据，例如pix2pix等算法。


**1.9 如何更换文本检测/识别的backbone？**

A：无论是文字检测，还是文字识别，骨干网络的选择是预测效果和预测效率的权衡。一般，选择更大规模的骨干网络，例如ResNet101_vd，则检测或识别更准确，但预测耗时相应也会增加。而选择更小规模的骨干网络，例如MobileNetV3_small_x0_35，则预测更快，但检测或识别的准确率会大打折扣。幸运的是不同骨干网络的检测或识别效果与在ImageNet数据集图像1000分类任务效果正相关。飞桨图像分类套件PaddleClas汇总了ResNet_vd、Res2Net、HRNet、MobileNetV3、GhostNet等23种系列的分类网络结构，在上述图像分类任务的top1识别准确率，GPU(V100和T4)和CPU(骁龙855)的预测耗时以及相应的117个预训练模型下载地址。

（1）文字检测骨干网络的替换，主要是确定类似与ResNet的4个stages，以方便集成后续的类似FPN的检测头。此外，对于文字检测问题，使用ImageNet训练的分类预训练模型，可以加速收敛和效果提升。

（2）文字识别的骨干网络的替换，需要注意网络宽高stride的下降位置。由于文本识别一般宽高比例很大，因此高度下降频率少一些，宽度下降频率多一些。可以参考[PaddleOCR中MobileNetV3骨干网络的改动](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.3/ppocr/modeling/backbones/rec_mobilenet_v3.py)。


**1.10 如何对检测模型finetune，比如冻结前面的层或某些层使用小的学习率学习？**

**A**：如果是冻结某些层，可以将变量的stop_gradient属性设置为True，这样计算这个变量之前的所有参数都不会更新了，参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/faq/train_cn.html#id4

如果对某些层使用更小的学习率学习，静态图里还不是很方便，一个方法是在参数初始化的时候，给权重的属性设置固定的学习率，参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fluid/param_attr/ParamAttr_cn.html#paramattr

实验发现，直接加载模型去fine-tune，不设置某些层不同学习率，效果也都不错

**1.11 DB的预处理部分，图片的长和宽为什么要处理成32的倍数？**

**A**：和网络下采样的倍数（stride）有关。以检测中的resnet骨干网络为例，图像输入网络之后，需要经过5次2倍降采样，共32倍，因此建议输入的图像尺寸为32的倍数。


**1.12 在PP-OCR系列的模型中，文本检测的骨干网络为什么没有使用SEBlock？**

**A**：SE模块是MobileNetV3网络一个重要模块，目的是估计特征图每个特征通道重要性，给特征图每个特征分配权重，提高网络的表达能力。但是，对于文本检测，输入网络的分辨率比较大，一般是640\*640，利用SE模块估计特征图每个特征通道重要性比较困难，网络提升能力有限，但是该模块又比较耗时，因此在PP-OCR系统中，文本检测的骨干网络没有使用SE模块。实验也表明，当去掉SE模块，超轻量模型大小可以减小40%，文本检测效果基本不受影响。详细可以参考PP-OCR技术文章，https://arxiv.org/abs/2009.09941.


**1.13 PP-OCR检测效果不好，该如何优化？**

A： 具体问题具体分析:
- 如果在你的场景上检测效果不可用，首选是在你的数据上做finetune训练；
- 如果图像过大，文字过于密集，建议不要过度压缩图像，可以尝试修改检测预处理的resize逻辑，防止图像被过度压缩；
- 检测框大小过于紧贴文字或检测框过大，可以调整db_unclip_ratio这个参数，加大参数可以扩大检测框，减小参数可以减小检测框大小；
- 检测框存在很多漏检问题，可以减小DB检测后处理的阈值参数det_db_box_thresh，防止一些检测框被过滤掉，也可以尝试设置det_db_score_mode为'slow';
- 其他方法可以选择use_dilation为True，对检测输出的feature map做膨胀处理，一般情况下，会有效果改善；


## 6. 文本检测预测相关FAQ

**2.1 DB有些框太贴文本了反而去掉了一些文本的边角影响识别，这个问题有什么办法可以缓解吗？**

**A**：可以把后处理的参数[unclip_ratio](https://github.com/PaddlePaddle/PaddleOCR/blob/d80afce9b51f09fd3d90e539c40eba8eb5e50dd6/tools/infer/utility.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L52)适当调大一点，该参数越大文本框越大。


**2.2 为什么PaddleOCR检测预测是只支持一张图片测试？即test_batch_size_per_card=1**

**A**：预测的时候，对图像等比例缩放，最长边960，不同图像等比例缩放后长宽不一致，无法组成batch，所以设置为test_batch_size为1。


**2.3 在CPU上加速PaddleOCR的文本检测模型预测？**

**A**：x86 CPU可以使用mkldnn（OneDNN）进行加速；在支持mkldnn加速的CPU上开启[enable_mkldnn](https://github.com/PaddlePaddle/PaddleOCR/blob/8b656a3e13631dfb1ac21d2095d4d4a4993ef710/tools/infer/utility.py#L105)参数。另外，配合增加CPU上预测使用的[线程数num_threads](https://github.com/PaddlePaddle/PaddleOCR/blob/8b656a3e13631dfb1ac21d2095d4d4a4993ef710/tools/infer/utility.py#L106)，可以有效加快CPU上的预测速度。

**2.4 在GPU上加速PaddleOCR的文本检测模型预测？**

**A**：GPU加速预测推荐使用TensorRT。

- 1. 从[链接](https://paddleinference.paddlepaddle.org.cn/master/user_guides/download_lib.html)下载带TensorRT的Paddle安装包或者预测库。
- 2. 从Nvidia官网下载TensorRT版本，注意下载的TensorRT版本与paddle安装包中编译的TensorRT版本一致。
- 3. 设置环境变量LD_LIBRARY_PATH，指向TensorRT的lib文件夹
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<TensorRT-${version}/lib>
```
- 4. 开启PaddleOCR预测的[tensorrt选项](https://github.com/PaddlePaddle/PaddleOCR/blob/8b656a3e13631dfb1ac21d2095d4d4a4993ef710/tools/infer/utility.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L38)。

**2.5 如何在移动端部署PaddleOCR模型？**

**A**: 飞桨Paddle有专门针对移动端部署的工具[PaddleLite](https://github.com/PaddlePaddle/Paddle-Lite)，并且PaddleOCR提供了DB+CRNN为demo的android arm部署代码，参考[链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.3/deploy/lite/readme.md)。


**2.6 如何使用PaddleOCR多进程预测？**

**A**: 近期PaddleOCR新增了[多进程预测控制参数](https://github.com/PaddlePaddle/PaddleOCR/blob/8b656a3e13631dfb1ac21d2095d4d4a4993ef710/tools/infer/utility.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L111)，`use_mp`表示是否使用多进程，`total_process_num`表示在使用多进程时的进程数。具体使用方式请参考[文档](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.3/doc/doc_ch/inference.md#1-%E8%B6%85%E8%BD%BB%E9%87%8F%E4%B8%AD%E6%96%87ocr%E6%A8%A1%E5%9E%8B%E6%8E%A8%E7%90%86)。

**2.7 预测时显存爆炸、内存泄漏问题？**

**A**: 如果是训练模型的预测，由于模型太大或者输入图像太大导致显存不够用，可以参考代码在主函数运行前加上paddle.no_grad()，即可减小显存占用。如果是inference模型预测时显存占用过高，可以配置Config时，加入[config.enable_memory_optim()](https://github.com/PaddlePaddle/PaddleOCR/blob/8b656a3e13631dfb1ac21d2095d4d4a4993ef710/tools/infer/utility.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L267)用于减小内存占用。另外关于使用Paddle预测时出现**内存泄漏的问题，建议安装paddle最新版本，内存泄漏已修复。**
