# ***# 一．模型介绍：***

## *1.CNN模型的基本结构*

CNN(Convolutional Neural Networks)即卷积神经网络，是一种神经网络，其基本运算方式为卷积。一个简单的CNN由输入层，卷积层，池化层，全连接层组成。

 - **输入层(Input)**：计算机可理解为若干个矩阵。输入的图像数据，以矩阵形式的数据存在，如果输入一张尺寸为(H, W)的彩色图像，则输入层的数据为一个(H×W×3)的矩阵，数值范围为[0, 255]，其中，3表示RGB三个通道，一般称作该输入层为3通道，或者说包含3个feature map，如果是灰度图像，则数据表示为(H×W×1)，1个通道或者1个feature map。
	
 - **卷积层(Conv)**：其实就是对输出的图像的不同局部的矩阵和卷积核矩阵各个位置的元素相乘，然后相加得到。CNN基本运算，这是CNN特有的，由卷积核对输入层图像进行卷积操作以提取图像特征，1个卷积核生成1个通道，即卷积输出的图像通道数与卷积核的个数一致，卷积核的尺寸为(S×S×C×N)，其中C表示卷积核深度，必须与输入层图像的通道数一致，即如果输入图像是3通道，则C为3，如果是1通道，则C为1。在上图Conv2中，因为其输入图像的通道数为12，所以Conv2的卷积核深度为12，N表示卷积核的数量。
	
 - **池化层(pooling)**：就是对输入张量的各个子矩阵进行压缩，主要用于图像下采样，降低图像分辨率(对图像层的通道数没有影响)，减少区域内图像的特征数。常见的池化标准有2个，MAX或者是Average。即取对应区域的最大值或者平均值作为池化后的元素值。
	
 - **全连接层(Fully connected)**：全连接就是将输入的所有节点数据与输出的所有节点数据相连，其结构类似BP神经网络，在CNN中体现为将卷积图像映射至一个n维向量，通过设置多个全连接关系，起到从特征到分类的作用。
	
当然我们也可以灵活使用使用卷积层+卷积层，或者卷积层+卷积层+池化层的组合，这些在构建模型的时候没有限制。但是最常见的CNN都是若干卷积层+池化层的组合。

## 2.算法流程

 - 图像的卷积就是让卷积核(卷积模板)在原图像上依次滑动，在两者重叠的区域中，把对应位置的像素值和卷积模板值相乘，最后累加求和得到新图像(卷积结果)中的一个像素值，卷积核每滑动一次获得一个新值，当完成原图像的全部遍历，便完成原图像的一次卷积。输入图像是32*32*3（3是它的深度（即R、G、B），卷积层是一个5*5*3的filter(感受野)。在实际的运用过程中，通常会使用多层卷积层来得到更深层次的特征图。卷积层是通过一个可调参数的卷积核与上一层特征图进行滑动卷积运算，再加上一个偏置量得到一个净输出，然后调用激活函数得出卷积结果，通过对全图的滑动卷积运算输出新的特征图。
	
 - 激活函数主要引入非线性特性，在卷积过程中，所有的运算都是线性运算和线性叠加，无论网络多复杂，输入输出关系都是线性关系，没有激活函数，网络就无法拟合非线性特性的输入输出关系，常用的激活函数有Sigmoid，Tanh，ReLU等。在CNN中，卷积核的每一次卷积在累加模板内各个位置的乘积后，将累加值输入激活函数，然后将输出值作为卷积结果。
 - 池化层是将输入的特征图用nxn的窗口划分成多个不重叠的区域，然后对每个区域计算出最大值或者均值，使图像缩小了n倍，最后加上偏置量通过激活函数得到抽样数据。
 - 全连接层则是通过提取的特征参数对原始图像进行分类，就是它把特征整合到一起，输出为一个值。图像经过多次卷积和池化后，通过全连层完成分类操作，设卷积后的图像尺寸为(h×w×c)，需分成n类，则全连层的作用为将[h×w×c]的矩阵转换成[n×1]的矩阵。传统的分类方法一般操作为图像预处理，ROI定位，目标定位，特征提取，SVM或BP分类，在基于CNN的分类方法中，可以把卷积和池化操作看作传统方法的图像预处理到特征提取过程，因此CNN的操作结果就是网络自主学习并提取了一个[h×w×c]大小的特征值，然后在全连接层中进行了n目标分类任务。常用的分类方法如下式yl=f(wlxl-1+bl)。式中xl-1为前一层的特征图，通过卷积核抽样提取出来的特征参数；wl为全连接层的权重系数；bl为l层的偏置量。
	
 - 此外，CNN还有反向传播算法，学习的目的是获得对输入准确精炼的描述。影响输出结果的是每层的权重和偏置，因此为了达到目标，需要将输出误差层层传递回去，看每个参数对误差的影响，并因此调整参数。 
通过损失函数计算输出层的δL
从倒数第二层开始，根据下面3种情况进行反向传播计算：
如果当前是全连接层，则有δl =（Wl+1）Tδl+1⊙ σ’（zl）
如果上层是卷积层，则有δl =δl +1*rot180(Wl+1)⊙σ’（zl）
如果上层是池化层，则有δl=upsample(δl +1)

## 3.CNN模型优势

 - 由于CNN（卷积神经网络）是计算机视觉领域非常有效的模型，因此采用CNN模型。CNN之所以比其他模型更适合解决图像问题，因为每一张图像本质都是一个矩阵，而CNN的扫描窗口也是一个小矩形，这样不仅仅可以利用当前像素点，还能利用当前像素点和周围像素点的关系。
 - CNN模型能最大的利用图像的局部信息，图像中有固有的局部模式（比如轮廓、边界，人的眼睛、鼻子、嘴等）可以利用，显然应该将图像处理中的概念和神经网络技术相结合，对于CNN来说，并不是所有上下层神经元都能直接相连，而是通过“卷积核”作为中介。同一个卷积核在所有图像内是共享的，图像通过卷积操作后仍然保留原先的位置关系。一个典型的卷积神经网络结构，到最后一层实际上是一个全连接层，输入层到隐含层的参数个数瞬间降低到了，这使得我们能够用已有的训练数据得到良好的模型，不用担心参数膨胀，不易导致过度拟合和局部最优解问题的出现，适用于图像识别。
 - CNN模型有许多优势，比如
1.共享卷积核
2.对高维数据处理无压力。
3.图像通过卷积操作后仍然保留原先的位置关系
4.可以自动进行特征提取等。

## 4.选择训练模型

 - 本实验中刚开始自己讲CNN模型设计为3个3个卷积——最大池化层，一个mixed层，2个全连接层和输出层组成，但在训练后处理数据结果不是很好，出现了过拟合，之后在网上查找到资料，选择了使用Keras进行迁移学习，而Keras中可以导入的模型有Xception，VGG16，VGG19，ResNet50，InceptionV3，InceptionResNet -V2，MobileNet， 最后选用迁移学习的基础模型为Xception，InceptionV3和ResNet50。Xception，InceptionV3和ResNet50这三个模型对于输入数据都有各自的默认值，比如在输入图片大小维度上，Xception和InceptionV3默认输入图片大小是299*299，ResNet50默认输入图片大小是224*224；在输入数值维度上，Xception和InceptionV3默认输入数值在(-1, 1)范围内。当要输入与默认图片大小不同的图片时，只需传入当前图片大小即可。ResNet50需要对图片进行中心化处理，由于载入的ResNet50模型是在ImageNet数据上训练出来的，所以在预处理中每个像素点都要减去ImageNet均值。

# - 二、模型实现

## 1.数据处理的方法

 - 首先第一步是异常数据清理：我根据网上的资料，得知该数据集有一些非猫或狗的图像，这些图片属于离群数据，可能会影响模型精度，需要移除，其中包括资料给出的cat.4688.jpg，cat.5418.jpg，cat.7377.jpg，cat.7564.jpg，cat.8100.jpg，cat.8456.jpg，cat.10029.jpg，cat.12272.jpg，dog.1259.jpg，dog.1895.jpg，dog.4367.jpg，dog.8736.jpg，dog.9517.jpg，dog.10190.jpg，dog.11299.jpg和我无意中发现的dog.10797.jpg，将这些图片删除后训练集中猫狗都为12492张照片。
 - 之后要进行训练集划分验证集以便模型训练时在验证集上观察评估指标是否达到要求，于是将24984张图片的训练集分割为TRAIN_NUM=11492张图片的训练集和VALIDATION_NUM=1000张图片的验证集，使用np.random.permutation()函数即可，该函数可以返回被打乱的下标。
数据预处理完成，接下来开始准备读取数据，我们知道训练数据集的图片标签包含在图片的名字中，比如cat.1000.jpg，表示该图片标签为cat，在所有的猫图片中是第1000个，所以想要载入图片的话，先要把图片的路径读取出来。先利用os.listdir读取所有训练数据集train_images_all[]和测试集test_images_filepaths[]，之后从训练数据集中选出标签为dog的train_images_dog和标签为cat的train_images_dog。
 - 接着定义一个载入图片并统一尺寸的函数read_image(file_path)。为了不改变图片中物体的尺寸，首先将图片的长边按照某个比例缩小或者扩大到150，然后对图片的短边进行相同比例的缩放，如果图片不为正方形，那么使用黑色像素点将其填充为150*150的正方形图片。
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210526224617825.png)
  ![在这里插入图片描述](https://img-blog.csdnimg.cn/20210526224629801.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1ODA4NzAw,size_16,color_FFFFFF,t_70)


 - 最后将处理后的图片数据保存为TFRecord文件，这样做有诸多好处，图片数据被保存为TFRecord文件后，就是一个Check Point，之后的模型训练可以直接载入TFRecord文件，不需要再对图片的进行预处理；图片数据可以被保存为多个TFRecord文件，使用的时候可以一个一个batch载入（可以乱序载入），只载入训练需要的数据，可以减少内存的使用。我们将训练集分成若干个文件保存，每个文件中保存1024个图片；验证集和测试集保存为一个文件，同时打印信息看到保存文件的进度。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210526224637297.png)

## 2.模型实现的过程

	

 - 在本例的CNN模型实现时，将卷积池化层，展开层，全连接层和输出层分别写成函数conv2d_maxpool，flatten，fully_conn，output，然后将整个神经网络的计算过程包装为一个函数，最后本例中CNN模型设计为3个3个卷积——最大池化层，一个mixed层，2个全连接层和输出层组成。
	
 - 然后定义输入，输出，损失函数，优化方式和使用sigmoid将结果转化为概率。使用队列从文件中读取训练集，每次读取batch_size=500张图片。每训练5个batch，从batch中随机抽取20个数据计算loss和accuracy，从验证集中随机抽取80个数据计算loss和accuracy，以便实时观察模型在训练集和验证集上的分类效果。
 - 之后选用迁移学习的基础模型为Xception，InceptionV3和ResNet50进行迁移学习，首先在网上下载Xception，InceptionV3和ResNet50导出的特征向量，![在这里插入图片描述](https://img-blog.csdnimg.cn/20210526224718971.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1ODA4NzAw,size_16,color_FFFFFF,t_70)
载入预处理的数据之后，先进行一次概率为0.5的dropout，然后直接连接输出层，激活函数为Sigmoid，优化器为Adam，输出一个零维张量，表示某张图片中有狗的概率，分割训练集和验证集，并构建模型，使用处理过的训练集数据进行模型训练，处理过的验证集数据检测模型的分类效果。

## 3.模型实现细节

 - 为了防止过拟合，本项目进行了Dropout处理。Dropout的一种主流解释是进行Dropout后，网络会变得稀疏，最后对测试集进行分类时，相当于多个稀疏网络一起决定结果。
由于最后提交的结果表示图片为狗的概率，因此输出层使用Sigmoid函数处理一下（Sigmoid函数的值域是[0, 1]，常常用其输出表示概率）。
 - 当一个完整的数据集通过了神经网络一次并且返回了一次，这个过程称为一个 epoch。在神经网络中传递完整的数据集一次是不够的，我们需要将完整的数据集在同样的神经网络中传递多次。随着 epoch 数量增加，神经网络中的权重的更新次数也增加，曲线从欠拟合变得过拟合。
 - 学习速率是一个超参数，设置过小会导致收敛较慢，设置过大会导致准确率和loss振荡较大难以收敛。学习速率在0.01时，准确率和loss振幅和振荡频率都比较大，从0.01调整至到0.001后，准确率和loss振荡幅度和频率有明显降低，模型更加稳健，所以在尝试的两个学习速率中，学习速率为0.001时，模型得到最优分数是合理的。
 - 对于自己建立的cnn模型，累计训练了20个epochs之后，最后15个batch，训练batch抽样loss平均值是0.0691，accuracy平均值是98.33%，验证集抽样loss平均值是0.4458，accuracy平均是82.50%。
 	 
 - 而当利用Xception，InceptionV3和ResNet50进行迁移学习时，结果则是一共训练了5个epochs，在第1个epoch的第1个batch训练后，训练集loss为0.6375，验证集loss为0.6346，到最后一个batch，训练集loss为0.0188，验证集loss为 0.0180。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210526224809154.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQ1ODA4NzAw,size_16,color_FFFFFF,t_70)

# 三、实验结果分析

 - 一开始通过自己定义的模型进行训练batch抽样loss平均值是0.0691，accuracy平均值是98.33%，验证集抽样loss平均值是0.4458，accuracy平均是82.50%，过拟合典型的表现为训练集损失远远小于验证集损失,从以上的数据中我们知道CNN模型过拟合。
 - 而当我们之后使用Xception，InceptionV3和ResNet50这三个模型进行迁移学习，验证集loss是0.0108，测试集loss是0.04472。
 - 这是因为我们自定义的卷积神经网络只有14层，而Xception有126层，InceptionV3有159层，ResNet50有168层，更多的层数，不同的卷积核各种各样的的组合，就可以更好的抽取图片中的泛化特征，这样既可以提高分类的准确率，又可以降低模型的过拟合风险Xception，InceptionV3和ResNet50这三个模型如果不进行组合直接进行迁移学习的话，效果肯定也比先前自己搭建的神经网络效果好，但是一般情况下，组合比不组合强，所以就直接进行组合了。这里利用了bagging的思想，通过多个模型处理数据并进行组合，可以有效降低模型的方差，减少过拟合程度，提高分类准确率。
 - 想提高准确率，可以尝试不同的网络架构，改变epoch-size和epoch次数，改变学习率。本例中在模型后还使用numpy.clip()函数做一个截断处理，将所有图片的概率值限制在[0.005, 0.995]之间，将每个预测值限制到了 [0.005, 0.995] 个区间内，这样可以稍微降低loss。

# - 四、学习心得及建议

 - 在本次的猫狗大战实验中，我查阅了许多网上的猫狗大战的资料，也利用多种模型代码进行实验，也利用BAIDU AI Studio进行尝试，利用这些代码建立的CNN模型训练效果不是很好，而且最后输出的值也有很大悬殊，于是最后利用了Xception，InceptionV3和ResNet50三个结合模型来进行训练。
通过这次猫狗大战的实验，我对CNN模型理解更加深刻。
 - 首先，我理解了输入层，卷积层，池化层，全连接层等的意义与作用，接着对于CNN模型的算法流程有了一些了解，例如CNN模型的正向传播算法流程和反向传播算法流程，还有CNN模型相对其他神经网络模型在图像领域的优势，还有认识了Xception，InceptionV3和ResNet50模型。
 - 而对于模型的实现过程，我知道了如何利用os.listdir读取大量一定命名格式的图片，如何对大量数据划分训练集和验证集，如何自己定义的函数read_image(file_path)来将读入的图片统一尺寸，对于要进行多次训练的数据集可以将图片保存为多个TFRecord文件，便于每次读取数据，之后对数据进行训练时也更加理解了多次epoch训练的意义。
 - 而对于最后结果的处理，我也认识到了dropou处理，Sigmoid函数处理，学习速率与epoch的影响，又如利用umpy.clip()函数做一个截断处理，将所有图片的概率值限制在[0.005, 0.995]之间从而降低logloss值的处理。
 - 经过本次实验，我体会到了利用深度学习来进行分类处理的快乐，当看到训练结果输出时会有一种愉快感，因此希望如果有一些其他有趣的这种机器学习的题目，老师可以分享给我们，让我们更加深入的了解各种类型的深度学习知识与模型。


```python
import matplotlib.pyplot as plt
import numpy as np
import os, cv2, math
import tensorflow as tf
from IPython.display import display, Image, HTML

TRAIN_DIR = 'C:/Users/HF/Desktop/machineLearning/cat vs dog data/train/'
TEST_DIR = 'C:/Users/HF/Desktop/machineLearning/cat vs dog data/test/'
IMAGE_SIZE = 150
CHANNELS = 3
PIXEL_DEPTH = 255.0  # 像素最大值
TRAIN_DATA_NUM = 11492 # 狗和猫各自的训练图片数量
VALIDATION_DATA_NUM = 1000 # 狗和猫各自的验证图片数量
TRAIN_NUM = TRAIN_DATA_NUM * 2 # 训练集的图片数量，不包含验证集
VALIDATION_NUM = VALIDATION_DATA_NUM * 2 # 验证集的图片数量
TEST_NUM = 12500 # 测试集数量

# 训练集 + 验证集
train_images_all = [TRAIN_DIR + i for i in os.listdir(TRAIN_DIR)]
train_images_dog = [i for i in train_images_all if 'dog' in i]
train_images_cat = [i for i in train_images_all if 'cat' in i]
train_images_filepaths = train_images_dog + train_images_cat
train_images_labels = np.array([1] * (TRAIN_DATA_NUM + VALIDATION_DATA_NUM) \
                               + [0] * (TRAIN_DATA_NUM + VALIDATION_DATA_NUM)) # 1表示dog，0表示cat
# 测试集
test_images_filepaths = [TEST_DIR + i for i in os.listdir(TEST_DIR)]

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img.shape[0] > img.shape[1]:
        img_size = (IMAGE_SIZE, int(round(float(img.shape[1] / img.shape[0]) * IMAGE_SIZE)))
    else:
        img_size = (int(round(float(img.shape[0] / img.shape[1]) * IMAGE_SIZE)), IMAGE_SIZE)
    
    # OpenCV的图像size都是先水平轴再垂直轴，和一个矩阵的shape刚好相反
    tmp_img = cv2.resize(img, (img_size[1], img_size[0]), interpolation=cv2.INTER_CUBIC)
    ret_img = cv2.copyMakeBorder(tmp_img, 0, IMAGE_SIZE-img_size[0], 0, IMAGE_SIZE-img_size[1], cv2.BORDER_CONSTANT, 0)
    
    return ret_img[:,:,::-1] # 转化为rgb格式

# 载入训练集 + 验证集
train_dataset = np.ndarray((TRAIN_NUM + VALIDATION_NUM , IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.uint8)
for i, image_path in enumerate(train_images_all):
    train_dataset[i] = read_image(image_path)
print("Train shape: {}".format(train_dataset.shape))

# 载入测试集
test_dataset = np.ndarray((TEST_NUM, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.uint8)
for i, image_path in enumerate(test_images_filepaths):
    test_dataset[i] = read_image(image_path)
print("Test shape: {}".format(test_dataset.shape))

%matplotlib inline
plt.imshow (train_dataset[0,:,:,:], interpolation='nearest')
plt.figure ()
plt.imshow (train_dataset[1,:,:,:], interpolation='nearest')
plt.figure ()
plt.imshow (train_dataset[2,:,:,:], interpolation='nearest')
plt.figure ()
plt.imshow (train_dataset[TRAIN_DATA_NUM + VALIDATION_DATA_NUM,:,:,:], interpolation='nearest')
plt.figure ()
plt.imshow (train_dataset[TRAIN_DATA_NUM + VALIDATION_DATA_NUM + 1,:,:,:], interpolation='nearest')
plt.figure ()
plt.imshow (train_dataset[TRAIN_DATA_NUM + VALIDATION_DATA_NUM + 2,:,:,:], interpolation='nearest')

np.random.seed(201712)
# 打乱训练集中的数据
def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation,:,:,:]
    shuffled_labels = labels[permutation]
  
    return shuffled_dataset, shuffled_labels

train_dataset_rand, train_labels_rand = randomize(train_dataset, train_images_labels)
train = train_dataset_rand[:TRAIN_NUM,:,:,:]
train_labels = train_labels_rand[:TRAIN_NUM]
valid = train_dataset_rand[-VALIDATION_NUM:]
valid_labels = train_labels_rand[-VALIDATION_NUM:]
print('Train shape: ', train.shape, train_labels.shape)
print('Valid shape: ', valid.shape, valid_labels.shape)

#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
    
# 把数据写到多个文件中
def save_multiple_tfrecords(images, labels, filename, instances_per_file):
    image_num = labels.shape[0]
    file_num = math.ceil(image_num / instances_per_file) # 计算需要写的文件总数
    
    # 遍历每一张图片
    write_num = instances_per_file
    cur_file_no = -1
    for i in range(image_num):
        # 如果一个文件的图片达到预定的数目，则创建新的文件继续写
        if write_num == instances_per_file:
            write_num = 0
            cur_file_no += 1
            write_filename = filename + '.tfrecords-%.4d-of-%.4d' % (cur_file_no, file_num)
            print('Writing ' + write_filename)
            writer = tf.compat.v1.python_io.TFRecordWriter(write_filename)
        # 写图片到文件
        image_bytes = images[i].tostring()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'label': _int64_feature(labels[i]),
                    'image_raw': _bytes_feature(image_bytes)
            }))
        writer.write(example.SerializeToString())
        write_num += 1
              
    writer.close()
    print('Writing End.')


save_multiple_tfrecords(train, train_labels, 'C:/Users/HF/Desktop/machineLearning/cat vs dog data/data/train', 1024)
save_multiple_tfrecords(valid, valid_labels, 'C:/Users/HF/Desktop/machineLearning/cat vs dog data/data/vaild', VALIDATION_NUM)
save_multiple_tfrecords(test_dataset, np.array([0] * TEST_NUM), 'C:/Users/HF/Desktop/machineLearning/cat vs dog data/data/test', 12500)
# 卷积池化
def conv2d_maxpool(x_tensor, conv_depth, conv_ksize, conv_strides, pool_ksize, pool_strides):
    input_depth = x_tensor.shape[-1].value
    # 卷积
    conv_weights = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1], input_depth, conv_depth], stddev=0.05))
    conv_bias = tf.Variable(tf.zeros(conv_depth))
    conv = tf.nn.conv2d(x_tensor, conv_weights, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    conv_output = tf.nn.relu(tf.nn.bias_add(conv, conv_bias))
    # 池化
    pool_out = tf.nn.max_pool(conv_output, ksize=[1, pool_ksize[0], pool_ksize[1], 1],\
                             strides=[1, pool_strides[0], pool_strides[1], 1], padding='SAME')
    
    return pool_out


# 展开函数，全连接层的数据depth都是1
def flatten(x_tensor):
    nodes = np.prod(x_tensor.shape.as_list()[1:])
    x_reshape = tf.reshape(x_tensor, [-1, nodes])
    
    return x_reshape


# 全连接
def fully_conn(x_tensor, num_outputs):
    height = x_tensor.shape[1].value
    weights = tf.Variable(tf.truncated_normal([height, num_outputs], stddev=0.05))
    bias = tf.Variable(tf.zeros(num_outputs))
    fully_output = tf.nn.relu(tf.add(tf.matmul(x_tensor, weights), bias))
        
    return fully_output


# 输出
def output(x_tensor, num_outputs):
    height = x_tensor.shape[1].value
    weights = tf.Variable(tf.truncated_normal([height, num_outputs], stddev=0.05))
    bias = tf.Variable(tf.zeros(num_outputs))
    
return tf.add(tf.matmul(x_tensor, weights), bias)

def conv_net(x, keep_prob):
    # 3个卷积——最大池化层
    conv1 = conv2d_maxpool(x, conv_depth=64, conv_ksize=[2,2], conv_strides=[1,1], \
                           pool_ksize=[2,2], pool_strides=[1,1])
    conv2 = conv2d_maxpool(conv1, conv_depth=128, conv_ksize=[2,2], conv_strides=[2,2],\
                          pool_ksize=[3,3], pool_strides=[2,2])
    conv3 = conv2d_maxpool(conv2, conv_depth=256, conv_ksize=[4,4], conv_strides=[2,2],\
                          pool_ksize=[3,3], pool_strides=[2,2])
    # 第1个mixed层
    mixed1_branch0 = conv2d_maxpool(conv3, conv_depth=256, conv_ksize=[1,1], conv_strides=[1,1],\
                                   pool_ksize=[1,1], pool_strides=[1,1])
    mixed1_branch1 = conv2d_maxpool(conv3, conv_depth=512, conv_ksize=[3,3], conv_strides=[1,1],\
                                   pool_ksize=[2,2], pool_strides=[1,1])
    mixed1_branch2 = conv2d_maxpool(conv3, conv_depth=512, conv_ksize=[5,5], conv_strides=[1,1],\
                                   pool_ksize=[3,3], pool_strides=[1,1])
    mixed1 = tf.concat([mixed1_branch0, mixed1_branch1, mixed1_branch2], 3) # depth=1280
    
    
    # 2个全连接层，dropout
    flatten_x = flatten(mixed1)
    fully_conn1 = tf.nn.dropout(fully_conn(flatten_x, 640), keep_prob)
    fully_conn2 = tf.nn.dropout(fully_conn(fully_conn1, 320), keep_prob)
    
    # 输出层
    y = output(fully_conn2, 1)
    
    return y

# input
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name='x')
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name='y')
keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

# model
logits = tf.nn.sigmoid(conv_net(x, keep_prob)) # 使用sigmoid将结果转化为概率
logits = tf.identity(logits, name='logits') # 加个名字方便持久化后载入

# loss and optimizer
loss = tf.losses.log_loss(y, logits)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Accuracy
equal = tf.equal(tf.where(tf.greater(logits, 0.5), tf.ones_like(logits), tf.zeros_like(logits)), y)
accuracy = tf.reduce_mean(tf.cast(equal, tf.float32), name='accuracy')

def pre_process_for_train(image):
    image_data = tf.cast(image, tf.float32)
    # 将所有的像素值转化到[-1,1]之间
    image_data = (image_data / PIXEL_DEPTH - 0.5) * 2
    
    return image_data

def print_status(session, feature_batch, label_batch, loss, accuracy, end='\n'):
    cur_loss = session.run(loss, feed_dict={x:feature_batch, y:label_batch, keep_prob:1})
    cur_acc = session.run(accuracy, feed_dict={x:feature_batch, y:label_batch, keep_prob:1})
    print('loss: {:>6,.4f}  acc: {:>6,.4f}'.format(cur_loss, cur_acc), end=end)

# 载入验证集
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
filename = './data/valid.tfrecords-0000-of-0001'
filename_queue = tf.train.string_input_producer([filename]) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)  #返回文件名和文件
valid_features = tf.parse_single_example(
    serialized_example,
    features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })

# 把字节串编码成矩阵，并根据尺寸还原图像
image = tf.decode_raw(valid_features['image_raw'], tf.uint8)
image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, CHANNELS])
label = valid_features['label']

# 处理图像
processed_image = pre_process_for_train(image)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
   
    valid = np.ndarray((VALIDATION_DATA_NUM*2, IMAGE_SIZE, IMAGE_SIZE, CHANNELS), dtype=np.float32)
    valid_label = np.ndarray((VALIDATION_DATA_NUM*2, 1), dtype=np.float32)
    for i in range(VALIDATION_DATA_NUM*2):
        # 处理图像
        valid[i], valid_label[i] = sess.run([processed_image, label])

    coord.request_stop()
    coord.join(threads)

import h5py, math
from keras.layers import Input, Lambda
from keras.applications import inception_v3, resnet50, xception

print(train_dataset_rand.shape, train_labels_rand.shape, test_dataset.shape) # 查看一下数据大小

def write_feature_vector(MODEL, pre_process):
    inputs = Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    if pre_process:
        inputs = Lambda(pre_process)(inputs)
        
    model = MODEL(input_tensor=inputs, weights='imagenet', include_top=False, pooling='avg')
    train_vector = model.predict(train_dataset_rand)
    test_vector = model.predict(test_dataset)
    
    with h5py.File("./transfer_learning_data/vector_{}.h5".format(MODEL.__name__)) as h:
        h.create_dataset("train", data=train_vector)
        h.create_dataset("test", data=test_vector)
        h.create_dataset("train_label", data=train_labels_rand)
    print(MODEL.__name__ + " ok.")
    
write_feature_vector(inception_v3.InceptionV3, pre_process=inception_v3.preprocess_input)
write_feature_vector(xception.Xception, pre_process=xception.preprocess_input)
write_feature_vector(resnet50.ResNet50, pre_process=resnet50.preprocess_input)

from sklearn.model_selection import train_test_split

x_train2, x_valid, y_train2, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=2017)

# 输出层
def output(x_tensor, num_outputs):
    height = x_tensor.shape[1].value
    weights = tf.Variable(tf.truncated_normal([height, num_outputs], stddev=0.05))
    bias = tf.Variable(tf.zeros(num_outputs))

    return tf.add(tf.matmul(x_tensor, weights), bias)

# 直接dropout，然后一个输出层
x = tf.placeholder(tf.float32, shape=[None, 6144], name='x')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

x = tf.nn.dropout(x, keep_prob)
logits = tf.nn.sigmoid(output(x, 1))
logits = tf.identity(logits, name='logits')


# loss and optimizer
loss = tf.losses.log_loss(y, logits)
optimizer = tf.train.AdamOptimizer().minimize(loss)



epochs = 5
batch_size = 512
keep_probability = 0.5
train_loss = []
valid_loss = []
   
def print_status(session, feature_batch, label_batch, loss, accuracy, end='\n', saved_list=None):
    cur_loss = session.run(loss, feed_dict={x:feature_batch, y:label_batch, keep_prob:1})
    cur_acc = session.run(accuracy, feed_dict={x:feature_batch, y:label_batch, keep_prob:1})
    if saved_list is not None:
        saved_list.append(cur_loss)
    print('loss: {:>6,.4f}  acc: {:>6,.4f} '.format(cur_loss, cur_acc), end=end)
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(epochs):
        steps = math.ceil(y_train2.shape[0] / batch_size)
        for step in range(steps):
            offset = step * batch_size
            batch_features = x_train2[offset:(offset + batch_size), :]
            batch_labels = y_train2[offset:(offset + batch_size)]

            sess.run(optimizer, feed_dict={x:batch_features, y:batch_labels, keep_prob:keep_probability})
            if step % 5 == 0: # 每5步打印一次loss和acc
                print('Epoch {:>2}, Batch {:>2}:  '.format(epoch + 1, step), end='')
                print_status(sess, x_train2, y_train2, loss, accuracy, end='', saved_list=train_loss)
                print_status(sess, x_valid, y_valid, loss, accuracy, saved_list=valid_loss)
    
    # predict
    y_test = sess.run(logits, feed_dict={x:x_test, keep_prob:1})

# 截断降低logloss
label = np.clip(y_test, 0.005, 0.995)

df = pd.DataFrame(label, columns=['label'])
id = sorted([str(i+1) for i in range(12492)])
df['id'] = id

df.to_csv('myfile.csv', columns=['id', 'label'], index=None)
```
