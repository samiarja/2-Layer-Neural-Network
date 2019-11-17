import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.tests.test_multioutput import n_samples

#Variables
n_hidden= 10
n_in= 10

#Output
n_out= 10

#Sample Data
n_sample= 300

#HyperParameters
learning_rate= 0.01
momentum= 0.9

#Non deterministic seeding
np.random.seed(0)
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
    return 1- np.tanh(x)**2

#Input Data, Transpose, Layer 1, Layer 2, Biases
def train(x, t, V, W, bv, bw):
    #Forward Propagation -- Matrix Multiply + biases
    A= np.dot(x, V) + bv
    Z=np.tanh(A)
    B=np.dot(Z, W) + bw
    Y= sigmoid(B)

#Backward Propagation
    Ew= Y - t
    Ev= tanh_prime(A) * np.dot(W, Ew)

#Predict our Loss
    dW= np.outer(Z, Ew)
    dV= np.outer(x, Ev)

    #Cross Entropy
    loss= -np.mean(t*np.log(Y) + (1-t)*np.log(1-Y))
    return loss, (dV,dW, Ev, Ew)

def predict(x, V, W, bv, bw):
    A= np.dot(x, V) + bv
    B= np.dot(np.tanh(A), W) + bw
    return (sigmoid(B)> 0.5).astype(int)
#Create Layers
V= np.random.normal(scale= 0.1, size=(n_in, n_hidden))
W=np.random.normal(scale=0.1, size=(n_hidden, n_out))

bv= np.zeros(n_hidden)
bw=np.zeros(n_out)

params=[V, W, bv, bw]

#Generate our Data
X= np.random.binomial(1, 0.5, (n_samples, n_in))
T=X^1

#Training Timeeeeeeeeeeeee
for epoch in range(10):
    err= []
    upd= [0]*len(params)

    t0= time.clock()

#For each data point, update our weights
    for i in range (X.shape[0]):
        loss,grad= train(X[i], T[i], *params)

        #Update Loss
        for j in range(len(params)):
            params[j]-= upd[j]
            for j in range(len(params)):
                upd[j] = learning_rate * grad[j] + momentum * upd[j]
                err.append(loss)
                print ('Epoch: %d, Loss: %0.8f, Time: %.4fs' % (
                    epoch, np.mean(err), time.clock()-t0))
              
                
    

#Prediction
x= np.random.binomial(1, 0.5, n_in)
print('XOR prediction')
print(x)
print(predict, (x, *params))
