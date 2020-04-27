import tensorflow as tf
import numpy as np
import voxel2pdb
import os
from scitbx.array_family import flex
from sastbx.zernike_model import pdb2zernike
from sastbx.zernike_model import model_interface


BATCH_SIZE = 1
SEED=56297

z_dim=3000
z_dim2=20

in_dim=36
fc1_dim=300
fc2_dim=500
fc3_dim=200
fc4_dim=100
out_dim=20

def variable_on_cpu(name,shape,stddev,trainable=False):
	with tf.device('/cpu:0'):
		var=tf.get_variable(name,shape,initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),trainable=trainable)
	return var
def variable_on_cpu_bias(name,shape,trainable=False):
	with tf.device('/cpu:0'):
		var=tf.get_variable(name,shape,initializer=tf.constant_initializer(0.1))
	return var


def iq2z(input):
	with tf.variable_scope('fc1_') as scope:
		weight1=variable_on_cpu('weight',[in_dim,fc1_dim],np.sqrt(2./in_dim))
		bias1=variable_on_cpu('bias',[fc1_dim],0)
		z1=tf.nn.relu(tf.matmul(input,weight1) + bias1)
	with tf.variable_scope('fc2_') as scope:
		weight2=variable_on_cpu('weight',[fc1_dim,fc2_dim],np.sqrt(2./fc1_dim))
		bias2=variable_on_cpu('bias',[fc2_dim],0)
		z2=tf.nn.relu(tf.matmul(z1,weight2) + bias2)
	with tf.variable_scope('fc3_') as scope:
		weight3=variable_on_cpu('weight',[fc2_dim,fc3_dim],np.sqrt(2./fc2_dim))
		bias3=variable_on_cpu('bias',[fc3_dim],0)
		z3=tf.nn.relu(tf.matmul(z2,weight3) + bias3)
	with tf.variable_scope('fc4_') as scope:
		weight4=variable_on_cpu('weight',[fc3_dim,fc4_dim],np.sqrt(2./fc3_dim))
		bias4=variable_on_cpu('bias',[fc4_dim],0)
		z4=tf.nn.relu(tf.matmul(z3,weight4) + bias4)
	with tf.variable_scope('fc5_') as scope:
		weight5=variable_on_cpu('weight',[fc4_dim,out_dim],np.sqrt(2./fc4_dim))
		bias5=variable_on_cpu('bias',[out_dim],0)
		z5=tf.matmul(z4,weight5) + bias5
	var_dict1 = {'fc1/weight':weight1,'fc1/bias':bias1,
			'fc2/weight':weight2,'fc2/bias':bias2,
			'fc3/weight':weight3,'fc3/bias':bias3,
			'fc4/weight':weight4,'fc4/bias':bias4,
			'fc5/weight':weight5,'fc5/bias':bias5,
			}
	return z5,var_dict1



def decode(z,batchsize):
	with tf.variable_scope('fc2') as scope:
		weight1_1=variable_on_cpu('weight',[z_dim2,8*8*8*32],np.sqrt(2./z_dim))
		bias1_1=variable_on_cpu('bias',[8*8*8*32],0)
		h=tf.nn.relu(tf.matmul(z,weight1_1) + bias1_1)
	h=tf.reshape(h,[-1,8,8,8,32])
	with tf.variable_scope('deconv1') as scope:
		weight2_1=variable_on_cpu('weight',[5,5,5,64,32],np.sqrt(2./(5*5*5*32)))
		bias2_1=variable_on_cpu('bias',[64],0)
		deconv=tf.nn.conv3d_transpose(h,weight2_1,[batchsize,16,16,16,64],[1,2,2,2,1],padding='SAME')
		deconv1=tf.nn.relu(deconv+bias2_1)
	with tf.variable_scope('deconv2') as scope:
		weight3_1=variable_on_cpu('weight',[5,5,5,128,64],np.sqrt(2./(5*5*5*64)))
		bias3_1=variable_on_cpu('bias',[128],0)
		deconv=tf.nn.conv3d_transpose(deconv1,weight3_1,[batchsize,32,32,32,128],[1,2,2,2,1],padding='SAME')
		deconv2=tf.nn.relu(deconv+bias3_1) 
	with tf.variable_scope('conv4') as scope:
		weight4_1=variable_on_cpu('weight',[3,3,3,128,1],np.sqrt(2./(3*3*3*128)))
		bias4_1=variable_on_cpu('bias',[1],0)
		conv=tf.nn.conv3d(deconv2,weight4_1,strides=[1,1,1,1,1],padding='SAME')
		logits=conv+bias4_1
	var_dict2 = {
			'fc2/weight':weight1_1,'fc2/bias':bias1_1,
			'deconv1/weight':weight2_1,'deconv1/bias':bias2_1,
			'deconv2/weight':weight3_1,'deconv2/bias':bias3_1,
			'conv4/weight':weight4_1,'conv4/bias':bias4_1}

	return logits,var_dict2


def generate_session(gpu_num):
	net_in=[]
	net_out=[]
	decode_in=[]
	decode_out=[]

	for ii in range(gpu_num):
		with tf.device('/gpu:%d'%ii):
			iq2z_iq=tf.placeholder(shape=[BATCH_SIZE,in_dim],dtype=tf.float32)
			iq2z_out,var_dict1=iq2z(iq2z_iq)

			de_in=tf.placeholder(shape=[BATCH_SIZE,z_dim2],dtype=tf.float32)
			de_out,var_dict2=decode(de_in,BATCH_SIZE)
			de_out=tf.nn.sigmoid(de_out)

			tf.get_variable_scope().reuse_variables()
			net_in.append(iq2z_iq)
			net_out.append(iq2z_out)
			decode_in.append(de_in)
			decode_out.append(de_out)
	return net_in,net_out,decode_in,decode_out,var_dict1,var_dict2


def get_iq(iqfilepath):
	iqdata=np.loadtxt(iqfilepath,delimiter=' ',dtype=float)
	#iqdata=(iqdata[:-1,1]-iqdata[1:,1]).reshape(1,-1)
	iqdata=iqdata[:,1].reshape(-1)
	prdata=np.zeros(100)
	prdata[:len(iqdata)]=iqdata
	return prdata.reshape(1,-1)


def get_result(filename):	
	try:
		#iqdata=get_iq(filename)
		iqdata=np.loadtxt(filename,delimiter=' ',dtype=float)
		iqdata=iqdata[:,1].reshape(-1)
		if len(iqdata)!=in_dim:
			print "iq size error"
			return
		iqdata=iqdata.reshape(1,-1)
		subz=sess.run(net_outt[0],feed_dict={net_int[0]:iqdata})
		wholez=subz.reshape(1,-1)
		voxel=sess.run(de_outt[0],feed_dict={de_int[0]:wholez})
		#voxel_a=np.zeros(shape=(32,32,32))
		voxel_a=voxel[0,:31,:31,:31,0].reshape(31,31,31)
		
		activepoint=np.greater(voxel_a,0.2)
		activepoint=activepoint.astype(int)
		rmax=70
		voxel2pdb.write_pdb_32(activepoint,'out.pdb',rmax)
		ccp4data=flex.double(activepoint.astype(float))
		pdb2zernike.ccp4_map_type(ccp4data, 15, rmax/0.9,file_name='out.ccp4')	
	except: 
		print "%s error!!!!!!!!!!!!!!!!!!!!!!"%filename

if __name__=='__main__':
	filename="../pisa_data_iq/1pgt.iq"
	saved_model_path1="../train_iq2gene/model"
	saved_model_path2="../train_autoencoder/model"
	net_int,net_outt,de_int,de_outt,var_dict1,var_dict2=generate_session(1)
	var_dict_one=dict(var_dict1.items())
	var_dict_two=dict(var_dict2.items())
	
	saver1=tf.train.Saver(var_list=var_dict_one)
	saver2=tf.train.Saver(var_list=var_dict_two)
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		model_path1=tf.train.latest_checkpoint(saved_model_path1)
		model_path2=tf.train.latest_checkpoint(saved_model_path2)
		saver1.restore(sess, model_path1)
		saver2.restore(sess, model_path2)
		get_result(filename)
		
	
	

