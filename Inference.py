import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import argparse
from train import InfoGan,sample
import sys
from imp import reload

reload(sys)
parser_=argparse.ArgumentParser(description='Inference')
parser_.add_argument('--model_file',dest='model',default=os.path.join(os.getcwd(),'InfoGan.ckpt'),help='model path')
param=parser_.parse_args()

def inference(z=None,c=None,d=None,size=100,reuse=True,name='test'):
    if z is None:
        z,_,_=sample(size)
    if c is None:
        _,c,_=sample(size)
    if d is None:
        _,_,d=sample(size)
    m_i=InfoGan()
    g_img=m_i.Generator(reuse=reuse)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,param.model)
        fake_image=sess.run(g_img,feed_dict={m_i.Z:z,m_i.C:c,m_i.D:d,m_i.is_training_g:False})
        canvas=np.empty((28*10,28*10))
        for xi in range(10):
            for yi in range(10):
                canvas[(xi)*28:(xi+1)*28,yi*28:(yi+1)*28]=fake_image[xi*10+yi].reshape(28,28)
        plt.figure(figsize=(8,10))
        plt.imshow(canvas,origin='upper',cmap='gray')
        plt.title('latent space to generate images '+name)
        plt.tight_layout()  
        plt.show()      
		
d=np.zeros((100,10))
for i in range(10):
    d[i*10:(i+1)*10,i]=1
ct_=np.linspace(-2,2,10)
ct_1=np.zeros((10,10))
ct_1[:]=ct_
ct_1=ct_1.reshape(100,1)
ct_2=np.zeros((100,1))
ct1=np.hstack((ct_1,ct_2))
ct2=np.hstack((ct_2,ct_1))
ct3=np.zeros((100,2))
z=np.zeros((100,62))

if __name__=='__main__':
    inference(d=d,z=z,c=ct3,reuse=False,name='varying catergorical');
    inference(d=d,z=z,c=ct1,reuse=True,name='varying continous1');
    inference(d=d,z=z,c=ct2,reuse=True,name='varying continous2')