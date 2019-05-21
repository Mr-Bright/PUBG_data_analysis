import tensorflow as tf
import pandas as pd
import numpy as np
#数据预处理，去掉字符型无用数据
match = pd.read_csv('match.csv')
prec = np.array(match['winPlacePerc']).tolist()
match.drop(['Id','groupId','matchId','matchType','winPlacePerc'],axis=1,inplace=True)
factor = np.array(match)
print('training data loaded')

test = pd.read_csv('test_V2.csv').fillna(0)
test.drop(['Id','groupId','matchId','matchType'],axis=1,inplace=True)
test = np.array(test)
print('test data loaded')

t = pd.read_csv('sample_submission_V2.csv')

#使用l2正则化处理损失
def get_weight(shape, lambd):
    var = tf.Variable(tf.random_normal(shape), dtype= tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    return var

#定义输入层和标签
x = tf.placeholder(tf.float32, shape=(None, 24))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

#定义选用数据大小和网络结构
batch_size = 400
layer_dimension = [24,64,32,16,8,4,1]
n_layers = len(layer_dimension)

#定义指数学习率下降
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, global_step, 3000,0.96,staircase=True)

#初始化输入数据和标签数据
dataset_size = 4411699
X = factor
Y = [[m] for m in prec]

#初始化各隐藏层节点
cur_layer = x
in_dimension = layer_dimension[0]

for i in range(1, n_layers-1):
    out_dimension = layer_dimension[i]
    print(layer_dimension[i])
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    cur_layer = tf.nn.leaky_relu(tf.matmul(cur_layer, weight)+bias)
    in_dimension = layer_dimension[i]

#初始化输出层
out_dimension = layer_dimension[n_layers-1]
print(layer_dimension[n_layers-1])
weight = get_weight([in_dimension, out_dimension], 0.001)
bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
output_layer = tf.nn.leaky_relu(tf.matmul(cur_layer, weight)+bias)

#计算损失，均方差作为损失函数
mse_loss = tf.reduce_mean(tf.square(y_ - output_layer))
tf.add_to_collection('losses',mse_loss)

loss = tf.add_n(tf.get_collection('losses'))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)


with tf.Session() as sess:
    #初始化全部节点
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #设定训练次数
    steps = 1000000
    #开始训练
    for i in range(steps):
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size, dataset_size)

        sess.run(train_step, feed_dict={x:X[start:end], y_: Y[start:end]})
        #每2000代计算一次总损失
        if i % 2000==0:
            total_cross_entropy = sess.run(loss, feed_dict={x:X, y_: Y})
            print('After %d training step(s), total loss on all data is %g' %(i,total_cross_entropy))

    #计算测试集泛化结果
    output = sess.run(output_layer, feed_dict={x: test,y_:Y[0:1934174]})
    print('testset:')
    #保存测试集运行结果
    output = np.array(output)
    print(output)
    win = []
    for i in output:
        for j in i:
            win.append(j)

    t['winPlacePerc'] = win
    t.loc[t['winPlacePerc'] > 1, 'winPlacePerc'] = 1
    t.loc[t['winPlacePerc'] < 0, 'winPlacePerc'] = 0
    t.to_csv('submission2.csv',index=False)


