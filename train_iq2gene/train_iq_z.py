import numpy as np
import tensorflow as tf
import random
import os
import exceptions

BATCH_SIZE = 64
NUM_EPOCHS = 30



in_dim=36
fc1_dim=300
fc2_dim=500
fc3_dim=200
fc4_dim=100
out_dim=20

TRAIN_RATE=0.02
LEARNING_RATE_BASE = 0.001
LEARNING_DECAY = 0.98
LEARNING_RATE_STEP = 100

global_steps = tf.Variable(0,name='global_steps',trainable=False)
learning_rate = tf.train.exponential_decay(
	LEARNING_RATE_BASE,
	global_steps,
	LEARNING_RATE_STEP,
	LEARNING_DECAY,
	staircase=False)


def read_tfrecords(filename):
	filename_quene=tf.train.string_input_producer([filename])
	reader=tf.TFRecordReader()
	_,serialized_example=reader.read(filename_quene)
	features=tf.parse_single_example(serialized_example,features={
								      'iq_data' : tf.FixedLenFeature([36], tf.float32),
								      'z_data' : tf.FixedLenFeature([20], tf.float32)
								      })
	train_data=tf.cast(features['iq_data'],tf.float32)
	label_data=tf.cast(features['z_data'],tf.float32)
	#label_data=label_data/tf.constant(200.0)
	return train_data , label_data



def iq2z(input):
	with tf.variable_scope('fc1') as scope:
		weight1=variable_on_cpu('weight',[in_dim,fc1_dim],np.sqrt(2./in_dim))
		bias1=variable_on_cpu('bias',[fc1_dim],0)
		z1=tf.nn.relu(tf.matmul(input,weight1) + bias1)
	with tf.variable_scope('fc2') as scope:
		weight2=variable_on_cpu('weight',[fc1_dim,fc2_dim],np.sqrt(2./fc1_dim))
		bias2=variable_on_cpu('bias',[fc2_dim],0)
		z2=tf.nn.relu(tf.matmul(z1,weight2) + bias2)
	with tf.variable_scope('fc3') as scope:
		weight3=variable_on_cpu('weight',[fc2_dim,fc3_dim],np.sqrt(2./fc2_dim))
		bias3=variable_on_cpu('bias',[fc3_dim],0)
		z3=tf.nn.relu(tf.matmul(z2,weight3) + bias3)
	with tf.variable_scope('fc4') as scope:
		weight4=variable_on_cpu('weight',[fc3_dim,fc4_dim],np.sqrt(2./fc3_dim))
		bias4=variable_on_cpu('bias',[fc4_dim],0)
		z4=tf.nn.relu(tf.matmul(z3,weight4) + bias4)
	with tf.variable_scope('fc5') as scope:
		weight5=variable_on_cpu('weight',[fc4_dim,out_dim],np.sqrt(2./fc4_dim))
		bias5=variable_on_cpu('bias',[out_dim],0)
		z5=tf.matmul(z4,weight5) + bias5
	return z5	

def variable_on_cpu(name,shape,stddev):
	with tf.device('/cpu:0'):
		var=tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
	return var


with tf.device('/cpu:0'):
	
	train_data,train_label=read_tfrecords('../iq2gene_traindata/train_iq2gene.tfrecords')
	train_data,train_label=tf.train.shuffle_batch([train_data,train_label],batch_size=BATCH_SIZE,capacity=6400,min_after_dequeue=3200)
	#train_data,train_label=tf.train.batch([train_data,train_label],batch_size=BATCH_SIZE,capacity=6400,min_after_dequeue=3200)

train_logits=iq2z(train_data)
#train_out=tf.nn.sigmoid(train_logits)
#train_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=train_label,logits=train_logits))
logdata=tf.square(train_label-train_logits)
train_loss=tf.reduce_mean(logdata)
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(train_loss,global_step=global_steps)


saver=tf.train.Saver()
sess=tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

threads=tf.train.start_queue_runners(sess=sess)

sess.run(tf.global_variables_initializer())


step=0
for e in range(NUM_EPOCHS):
	for ii in range(59000//BATCH_SIZE):
		step=step+1
		loss_,_,learning_rate_=sess.run([train_loss,optimizer,learning_rate])
		print 'iteration:{}/{},{}batchs,Training loss: {:.4f},learning_rate: {:.5f}'.format(e+1,NUM_EPOCHS,step,loss_,learning_rate_)

		
		if step%10==0:
			logfile=open('log.txt','a')
			logfile.write('iteration:{}/{},{}batchs,Training loss: {:.4f},learning_rate: {:.5f}\n'.format(e+1,NUM_EPOCHS,step,loss_,learning_rate_))
			logfile.close()

		if step%3000==0:
			saver_path=saver.save(sess,'model/model.ckpt',global_step=step)
		

sess.close()











