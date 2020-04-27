import tensorflow as tf 
import numpy as np
import auto_encoder_z
import os
import pdb2voxel
import voxel2pdb
import multiprocessing
import map2iq
from sastbx.zernike_model import model_interface
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D




"""
resize pdb file to same_Rmax 
"""
def generate_samermax_pdb(pdbname):
	#pisa
	try:
		pdbname=pdbname.strip()
		#cavitymodel=model_interface.build_model(pdbfile,'pdb',20,None)
		voxel=pdb2voxel.run(['pdbfile=rotate_pisa_pdb/%s'%pdbname])
		voxel2pdb.write_pdb(voxel,'samermax_pisa_pdb/%s'%pdbname,70)
	except:
		print '%s error!!!!!!!!!!!!!!!!!!!!!!!!!!!!'%pdbname
		fw=open('wrong_samples.txt','a')
		fw.write('%s\n'%pdbname)
		fw.close()

def run_samermax_pdb():
	#pisa
	f=open('pdbname.txt')
	pdbnames=f.readlines()
	pdbnames.sort()

	pool=multiprocessing.Pool(processes=20)
	pool.map(generate_samermax_pdb,pdbnames)
	pool.close()
	pool.join()


"""
generate autoencoder traindata
"""
def generate_aotoencoder_traindata():
	file=open('pdbname.txt')
	errorlog=open('wrong_pdb.txt','a')
	writer1=tf.python_io.TFRecordWriter('autoencoder_traindata/train.tfrecords')
	writer2=tf.python_io.TFRecordWriter('autoencoder_traindata/test.tfrecords')
	lines=file.readlines()
	count=0
	for line in lines:
		count+=1
		print count,line.strip()
		arg=['pdbfile=samermax_pisa_pdb/'+line.strip()]
		try:
			cube=np.zeros((32,32,32))
			cube[:31,:31,:31]=pdb2voxel.run(arg)
			cube=cube.astype(int)
			cube_list=cube.flatten()	
			example = tf.train.Example(features=tf.train.Features(feature={
		                "data": tf.train.Feature(int64_list=tf.train.Int64List(value=cube_list)),       #require a list of int 
		            }))
			if count%5==0:
				writer2.write(example.SerializeToString())
			else:
				writer1.write(example.SerializeToString())
		except:
			errorlog.write(line)
	writer1.close()
	writer2.close()
	errorlog.close()
	file.close()



"""
calculate pdb saxs profile
"""
def generate_iq(filename):
	#pisa
	iq_file='saxs_sample.txt'
	iq_data=np.loadtxt(iq_file,delimiter=' ',dtype=float)
	filename=filename.strip()
	try:
		t_voxel=pdb2voxel.run(['pdbfile=samermax_pisa_pdb/%s'%filename])
		#cavitymodel=model_interface.build_model(inpdbfile,'pdb',20,None)
		#rmax=cavitymodel.rmax*0.9
		rmax=70
		iq_t_curve,t_exp_data=map2iq.run_get_voxel_iq(t_voxel,iq_file,rmax)
		iq_t_curve=np.array(iq_t_curve)
		iq_t_curve=iq_t_curve/iq_t_curve[0]
		cacl_iq=np.concatenate([iq_data[:,0].reshape(-1,1),iq_t_curve.reshape(-1,1)],axis=1)
		np.savetxt('pisa_data_iq/%s.iq'%filename.split(".")[0],cacl_iq,fmt='%f')
	except: 
		print "%serror!!!!!!!!!!!!!!!!"%filename

def run_iq():
	#pisa
	f=open('pdbname.txt')
	pdbnames=f.readlines()
	pdbnames.sort()
	pool=multiprocessing.Pool(processes=20)
	pool.map(generate_iq,pdbnames)
	pool.close()
	pool.join()	





"""
generate gene from autoencoder
"""
def generate_gene():
	f=open('pdbname.txt')
	pdbnames=f.readlines()
	f.close()
	pdbnames=np.array(pdbnames)
	pdbnames=np.sort(pdbnames)
	saved_model_path='train_autoencoder/model'

	in_tensor_find,z_tensor_find,out_tensor_find=auto_encoder_z.generate_session(1)
	saver=tf.train.Saver()
	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		model_path=tf.train.latest_checkpoint(saved_model_path)
		saver.restore(sess, model_path)
		for pdbname in pdbnames:
			try:
				pdbname=pdbname.strip()
				inpdbfile='samermax_pisa_pdb/'+pdbname
				outpdbfile='pisa_data_gene/'+pdbname.split('.')[0]+'.txt'
				voxel=pdb2voxel.run(['pdbfile=%s'%inpdbfile])
				in_=np.zeros(shape=(1,32,32,32,1))
				in_[0,:31,:31,:31,0]=voxel

				z_=sess.run(z_tensor_find[0],feed_dict={in_tensor_find[0]:in_})
				z_=z_.reshape((-1,1))
				np.savetxt('%s'%outpdbfile,z_,fmt='%.3f')
			except: 
				print '%s error!!!!!!!!!!!!!!!!!!!!!!!!!!!!'%pdbname


"""
generate iq2gene train dataset
"""
def generate_iq2gene_traindata():
	writer1=tf.python_io.TFRecordWriter('iq2gene_traindata/train_iq2gene.tfrecords')
	writer2=tf.python_io.TFRecordWriter('iq2gene_traindata/test_iq2gene.tfrecords')
	f=open('pdbname.txt')
	pdbnames=f.readlines()
	lognum=0
	for line in pdbnames:
		try:
			pdbname=line.strip().split('.')[0]
			iq_data=np.loadtxt('pisa_data_iq/%s.iq'%pdbname,delimiter=' ',dtype=float)
			iq_data=iq_data[:,1].reshape(-1)
			z_data=np.loadtxt('pisa_data_gene/%s.txt'%pdbname,delimiter=' ',dtype=float)
			z_data=z_data.reshape(-1)
			
			example = tf.train.Example(features=tf.train.Features(feature={
				"iq_data": tf.train.Feature(float_list=tf.train.FloatList(value=iq_data)),
				"z_data":tf.train.Feature(float_list=tf.train.FloatList(value=z_data))
				}))
			
			writer1.write(example.SerializeToString())
			lognum=lognum+1
			if lognum%100==0:
				print 'lognum: %f\n'%lognum
		except:
			print "%s error!!!!!!!!!!!!!!!"%line
	writer1.close()
	writer2.close()
	f.close()


			

if __name__=='__main__':

	#run_samermax_pdb()
	#generate_aotoencoder_traindata()


	#run_iq()
	#generate_gene()
	generate_iq2gene_traindata()

	
	

	
	
	




    
