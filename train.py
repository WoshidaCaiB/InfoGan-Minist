import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import argparse

parser=argparse.ArgumentParser(description='InfoGan')
parser.add_argument('--train_file',action='store',dest='path',default=os.path.join(os.getcwd(),'train.csv'),help='train file path')
parser.add_argument('--model_file',action='store',dest='model',default=os.path.join(os.getcwd(),'InfoGan.ckpt'),help='model store path')
parser.add_argument('--epoch number',action='store',type=int,dest='epoch',default=100,help='epoch number')
param=parser.parse_args()
train=pd.read_csv(param.path)
train_data=train.iloc[:,1:].values
train_label=train.iloc[:,0].values
train_data=train_data/127.5-1.
train_data=np.reshape(train_data,(-1,28,28,1))

'''##### Brief Intro ####
c: Noisy code - dim: 62
c1: discrete latent code [k=10,p=0.1]
c2, c3: continuous latent code Uniform (-1,1)

Adam as optimizer
Batch Normalization for most of layers
Leaky Relu (leak rate 0.1) for discriminator 
Relu for generator

Regards to discrete latent codes,softmax is applied to predict correct category
Regards to continuous latent codes, posterior probability through diagonal Gaussian distribution is used. Generator output mean and log_stddev.
log_stddev is parameterized through an exponential transform to ensure postivity

learning rate: 2e-4 for D, G and Q

# 
D/Q:
input 28*28 image
4*4 conv 64 IRELU stride=2
4*4 conv 128 IRELU stride=2 BN
FC 1024 IRELU BN
FC for D
FC 128-BN-IRELU_FC output for Q

G:
Input 74 latent code
FC 1024 RELU BN
FC 7*7*128 RELU BN
4*4 upconv 64 RELU stride 2 BN
4*4 upconv 1 channel
'''

class InfoGan:
    def __init__(self):
        '''
        Args:
           X: true image
           Z: 64 dim noisy code
           C: 2 dim continuous code
           D: 10 dim categorical latent code
           is_training_d, is_training_g: tell BN layer whether it is at training stage
        '''
        self.X=tf.placeholder(dtype=tf.float32,shape=[None,28,28,1])
        self.Z=tf.placeholder(dtype=tf.float32,shape=[None,62])
        self.C=tf.placeholder(dtype=tf.float32,shape=[None,2])
        self.D=tf.placeholder(dtype=tf.float32,shape=[None,10])
        self.is_training_d=tf.placeholder(tf.bool,[])
        self.is_training_g=tf.placeholder(tf.bool,[])
        
    def Discriminator(self,x,reuse):
        with tf.variable_scope('discriminator',reuse=reuse):
            out1=tf.layers.conv2d(x,64,4,2,padding='SAME',kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),name='c_1')
            out1=self.lrelu(out1)
            out2=tf.layers.conv2d(out1,128,4,2,padding='SAME',kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),use_bias=False,name='c_2')
            out2=tf.layers.batch_normalization(out2,training=self.is_training_d,name='n_2')
            out2=self.lrelu(out2)
            out3=tf.reshape(out2,(-1,7*7*128))
            out3=tf.layers.dense(out3,1024,kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),use_bias=False,name='f_3')
            out3=tf.layers.batch_normalization(out3,training=self.is_training_d,name='n_3')
            out3=self.lrelu(out3)
            with tf.variable_scope('Discriminator_output'):
                D_out=tf.layers.dense(out3,1,kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),name='D_out')
                D_out=tf.nn.sigmoid(D_out)
            with tf.variable_scope('latent_code'):
                out4=tf.layers.dense(out3,128,kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),use_bias=False,name='f_4')
                out4=tf.layers.batch_normalization(out4,training=self.is_training_d,name='n_4')
                out4=self.lrelu(out4)
                mean=tf.layers.dense(out4,2,kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),name='c_mean')
                log_std=tf.layers.dense(out4,2,kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),name='c_stddev')
                discrete=tf.layers.dense(out4,10,kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),name='discrete')
            return D_out,discrete,mean,log_std
                
    def Generator(self,reuse):
        with tf.variable_scope('generator',reuse=reuse):
            latent=tf.concat([self.Z,self.D,self.C],1)
            out1=tf.layers.dense(latent,1024,kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),use_bias=False,name='f_1')
            out1=tf.layers.batch_normalization(out1,training=self.is_training_g,name='b_1')
            out1=tf.nn.relu(out1)
            out2=tf.layers.dense(out1,7*7*128,kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),use_bias=False,name='f_2')
            out2=tf.reshape(out2,[-1,7,7,128])
            out2=tf.layers.batch_normalization(out2,training=self.is_training_g,name='b_2')
            out2=tf.nn.relu(out2)
            out3=tf.layers.conv2d_transpose(out2,64,4,2,padding='SAME',kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),use_bias=False,name='dc_3')
            out3=tf.layers.batch_normalization(out3,training=self.is_training_g,name='b_3')
            out3=tf.nn.relu(out3)
            out4=tf.layers.conv2d_transpose(out3,1,4,2,padding='SAME',kernel_initializer=tf.random_normal_initializer(0,stddev=0.02),name='dc_4')
            final=tf.nn.tanh(out4)
            return final
    
    def cost(self):
        d_true,_,_,_=self.Discriminator(self.X,False)
        self.fake_image=self.Generator(False)
        d_gen,discrete,mean,log_std=self.Discriminator(self.fake_image,reuse=True)
        D_loss=-tf.reduce_mean(tf.log(d_true))-tf.reduce_mean(tf.log(1.-d_gen))
        G_loss=tf.reduce_mean(-tf.log(d_gen))
        self.g_c_loss=tf.reduce_mean(tf.reduce_sum(log_std+0.5*tf.square((self.C-mean)/tf.exp(log_std)),reduction_indices=1))
        self.g_d_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.D,logits=discrete))
        self.mutual_info=self.g_d_loss+self.g_c_loss
        self.g_loss=G_loss
        self.d_loss=D_loss
        
    def lrelu(self,x,leak=0.1):
        return tf.maximum(x,leak*x)

def sample(size):
    z=np.random.uniform(-1,1,size=(size,62))
    c=np.random.uniform(-1,1,size=(size,2))
    d=np.zeros((size,10))
    idx=np.random.randint(0,10,size=size)
    d[np.arange(size),idx]=1
    return z,c,d
	
def shuffle(x):
    indice=np.random.permutation(len(x))
    return x[indice]

#train model:
#D, G and Q will be trained iteratively	

def train(data,batch_size):
    model=InfoGan()
    model.cost()
    var_list=tf.trainable_variables()
    D_var=[var for var in var_list if 'discriminator' in var.name and 'latent_code' not in var.name]
    G_var=[var for var in var_list if 'generator' in var.name]
    m1_var=[var for var in var_list if 'discriminator' in var.name and 'Discriminator_output' not in var.name]
    m_var=m1_var+G_var
    op_d=tf.train.AdamOptimizer(2e-4)
    op_g=tf.train.AdamOptimizer(2e-4)
    op_m=tf.train.AdamOptimizer(2e-4)
    saver=tf.train.Saver(tf.global_variables())
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        op_D=op_d.minimize(model.d_loss,var_list=D_var)
        op_G=op_g.minimize(model.g_loss,var_list=G_var)
        op_M=op_m.minimize(model.mutual_info,var_list=m_var)
    step=len(data)//batch_size
    print('##### {} steps in each epoch #####\n'.format(step))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            x=shuffle(data)
            for j in range(step):
                z,c,d=sample(batch_size)
                _,curr_dloss=sess.run([op_D,model.d_loss],feed_dict={model.X:x[j*batch_size:(j+1)*batch_size],model.Z:z,model.C:c,model.D:d,model.is_training_d:True,model.is_training_g:True})
                z,c,d=sample(batch_size)
                _,curr_gloss,curr_image=sess.run([op_G,model.g_loss,model.fake_image],feed_dict={model.X:x[j*batch_size:(j+1)*batch_size],model.Z:z,model.C:c,model.D:d,model.is_training_d:True,model.is_training_g:True})
                z,c,d=sample(batch_size)
                _,curr_mloss,curr_closs,curr_dloss=sess.run([op_M,model.mutual_info,model.g_c_loss,model.g_d_loss],feed_dict={model.X:x[j*batch_size:(j+1)*batch_size],model.Z:z,model.C:c,model.D:d,model.is_training_d:True,model.is_training_g:True})
               
                if j%100==0:
                    print('### Epoch {}, Step {} ###'.format(i,j))
                    print('Current D_loss: {}, Current G_loss: {}, current_mutual: {}, current c loss: {}, curr d loss: {}\n'.format(curr_dloss,curr_gloss,curr_mloss,curr_closs,curr_dloss))
                    #print('Current D_loss: {}, Current G_loss: {}, current_mutual: {}\n'.format(curr_dloss,curr_gloss,curr_mloss))
      
        saver.save(sess,param.model)  

if __name__=='__main__':		
    train(train_data,100)
