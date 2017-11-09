# InfoGan-Mnist

Tensorflow Implementation of InfoGan

InfoGan paper:https://arxiv.org/abs/1606.03657

The Model follows exactly the struture proposed in the paper

Learning rate is 2e-4 for D, Q and G
 

## Run Model

To run model, tensorflow >=1.0 and python 3.6 is required

Model training: python train.py.  To check parameters, see python train.py -h

Model inference: python inference.py 


## Results:

1. Use fixed z and c and varying categorical code to generate diffferent digits

![img](https://github.com/WoshidaCaiB/InfoGan-Mnist/blob/master/images/results1.png)

2. Use fixed z and d and varying continous code to generate images with varying rotation angles and width. c1

![img](https://github.com/WoshidaCaiB/InfoGan-Mnist/blob/master/images/results2.png)

![img](https://github.com/WoshidaCaiB/InfoGan-Mnist/blob/master/images/results3.png)

While varying the categorical codes, the model cannot generate 8 well... The latent information to generate digit 8 is hidden in continuous code rather than in categorical code


## Results for model with only categorical latent codes

I removed the continous code and retrain the Model

Result:

1. Varying catergorical codes and fix z 

![img](https://github.com/WoshidaCaiB/InfoGan-Mnist/blob/master/images/output_12_3.png)

2. Varying categorical codes and z codes

![img](https://github.com/WoshidaCaiB/InfoGan-Mnist/blob/master/images/output_12_1.png)

Model can generate different digits well
