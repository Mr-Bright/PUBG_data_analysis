import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

#数据预处理，去掉字符型无用数据
match = pd.read_csv('train_V2.csv')
match = match.fillna(0)
prec = np.array(match['winPlacePerc']).tolist()
match.drop(['Id','groupId','matchId','winPlacePerc'],axis=1,inplace=True)
matchtype = np.array(match['matchType'])
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(matchtype)
print(integer_encoded)
match['matchType'] = integer_encoded
factor = np.array(match)
print('training data loaded')
del match

test = pd.read_csv('test_V2.csv').fillna(0)
test.drop(['Id','groupId','matchId'],axis=1,inplace=True)
test_matchtype = np.array(test['matchType'])
#label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(test_matchtype)
print(integer_encoded)
test['matchType'] = integer_encoded
test = np.array(test)
print('test data loaded')

t = pd.read_csv('sample_submission_V2.csv')

label_encoder = LabelEncoder()
result = label_encoder.fit_transform(prec)
print(result)

#使用l2正则化处理损失
def get_weight(shape, lambd):
    var = tf.Variable(tf.random_normal(shape), dtype= tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    return var

#定义输入层和标签
x = tf.placeholder(tf.float32, shape=(None, 25))
y_ = tf.placeholder(tf.int64, shape=(None,))

#定义选用数据大小和网络结构
batch_size = 400
layer_dimension = [25,64,32,16,8,4,3000]
n_layers = len(layer_dimension)

#定义指数学习率下降
global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.01, global_step, 3000,0.96,staircase=True)

#初始化输入数据和标签数据
dataset_size = 4446966
testdataset_size = 1934174
X = factor
Y = result
print(Y)
del factor
del result

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

prec_result = tf.argmax(output_layer,1)

#计算损失，均方差作为损失函数
mse_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_,logits=output_layer))
tf.add_to_collection('losses',mse_loss)

loss = tf.add_n(tf.get_collection('losses'))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)


with tf.Session() as sess:
    #初始化全部节点
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    #设定训练次数
    steps = 300000
    #开始训练
    for i in range(steps):
        start = (i*batch_size)%dataset_size
        end = min(start+batch_size, dataset_size)
        #print(X[start:end])
        sess.run(train_step, feed_dict={x:X[start:end], y_: Y[start:end]})
        #每2000代计算一次总损失
        if i % 2000==0:
            total_cross_entropy = sess.run(loss, feed_dict={x:X[start:end], y_: Y[start:end]})
            print('After %d training step(s), total loss on all data is %g' %(i,total_cross_entropy))

    del X
    #计算测试集泛化结果
    test_start = 0
    test_batchsize = 40000
    test_end = test_start+test_batchsize
    total_test_loss = 0
    count = 0
    final = []
    while test_end< testdataset_size:
        test_loss, out = sess.run([loss, prec_result], feed_dict={x: test[test_start:test_end], y_: Y[0:test_batchsize]})
        #print('testing processing ok')
        #total_test_loss = test_loss*test_batchsize+total_test_loss
        #print(out)
        #temp = label_encoder.inverse_transform(out).tolist()
        final.extend(out)
        #pd.DataFrame(temp).to_csv('result/result.csv',mode='a',index=False,header=False)
        #print(str(test_start) + '~' + str(test_end) + ' loss: ' + str(test_loss))
        #print('--------------------------------------------------------------------')
        test_start = test_start+test_batchsize
        test_end = test_end+test_batchsize
        count = count+1

    test_loss, out = sess.run([loss, prec_result], feed_dict={x: test[test_start:testdataset_size], y_: Y[test_start:testdataset_size]})
    #total_test_loss = test_loss * (testdataset_size-test_start) + total_test_loss
    #pd.DataFrame(label_encoder.inverse_transform(out).tolist()).to_csv('result/result.csv', mode='a', index=False, header=False)
    final.extend(out)
    final = label_encoder.inverse_transform(final).tolist()
    t['winPlacePerc'] = final
    t.to_csv('submission2.csv', index=False)





