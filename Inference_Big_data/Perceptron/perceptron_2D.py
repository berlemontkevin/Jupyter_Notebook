# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 23:20:20 2017

@author: Kevin
"""

from pylab import rand,plot,show,norm
import numpy as np
#def generateData(n):
# """ 
#  generates a 2D linearly separable dataset with n samples. 
#  The third element of the sample is the label
# """
# xb = (rand(n)*2-1)/2-0.5
# yb = (rand(n)*2-1)/2+0.5
# xr = (rand(n)*2-1)/2+0.5
# yr = (rand(n)*2-1)/2-0.5
# inputs = []
# for i in range(len(xb)):
#  inputs.append([xb[i],yb[i],1])
#  inputs.append([xr[i],yr[i],-1])
# return inputs
 
def generateData(n,w_teacher=None):
  if w_teacher ==  None :
    inputs=[]
    w_teacher2  = np.random.uniform(low=-0.5, high=0.5, size=2)
    w_teacher2 = w_teacher2/np.linalg.norm(w_teacher2)
    x = np.random.randn(n, 1)
    y = np.random.randn(n, 1)
    #x = np.concatenate((np.ones((n,1)),x), axis=1)
    for i in range(n):
	inputs.append([x[i],y[i],np.sign(x[i]*w_teacher2[0]+y[i]*w_teacher2[1])])

  else :
    inputs=[]
    w_teacher2 = w_teacher
    x = np.random.randn(n, 1)
    y = np.random.randn(n, 1)
    #x = np.concatenate((np.ones((n,1)),x), axis=1)
    for i in range(n):
	inputs.append([x[i],y[i],np.sign(x[i]*w_teacher2[0]+y[i]*w_teacher2[1])])

  return inputs,w_teacher2


class Perceptron:
 def __init__(self):
  """ perceptron initialization """
  self.w = rand(2)*2-1 # weights
  self.learningRate = 1

 def response(self,x):
  """ perceptron output, only in 2-D """
  y = x[0]*self.w[0]+x[1]*self.w[1] # dot product between w and x
  if y >= 0:
   return 1
  else:
   return -1

 def updateWeights(self,x,iterError):
  """
   updates the weights status, w at time t+1 is
       w(t+1) = w(t) + learningRate*(d-r)*x
   where d is desired output and r the perceptron response
   iterError is (d-r)
  """
  self.w[0] += self.learningRate*iterError*x[0]
  self.w[1] += self.learningRate*iterError*x[1]
  
 def train(self,data):
  """ 
   trains all the vector in data.
   Every vector in data must have three elements,
   the third element (x[2]) must be the label (desired output)
  """
  learned = False
  iteration = 0
  while not learned:
   globalError = 0.0
   for x in data: # for each sample
    r = self.response(x)    
    if x[2] != r: # if we have a wrong response
     iterError = x[2] - r # desired response - actual response
     self.updateWeights(x,iterError)
     globalError += abs(iterError)
   iteration += 1
   if globalError == 0.0 or iteration >= 100: # stop criteria
    print ('iterations',iteration)
    learned = True # stop learning
    
    
    
trainset,w_teacher = generateData(10) # train set generation
perceptron = Perceptron()   # perceptron instance
perceptron.train(trainset)  # training
#testset,w_teacher = generateData(100,w_teacher)  # test set generation

# Perceptron test
for x in trainset:
 r = perceptron.response(x)
 if r != x[2]: # if the response is not correct
  print ('error')
 if r == 1:
  plot(x[0],x[1],'ob')  
 else:
  plot(x[0],x[1],'or')

print w_teacher/norm(w_teacher)
print perceptron.w/norm(perceptron.w)
# plot of the separation line.
# The separation line is orthogonal to w
n = norm(perceptron.w)
ww = perceptron.w/n
#n2=norm(w_teacher)
ww1 = [ww[1],-ww[0]]
ww2 = [-ww[1],ww[0]]
#w_teacher1 = [w_teacher[1]/n2,-w_teacher[0]/n2]
#w_teacher2 = [-w_teacher[1]/n2,w_teacher[0]/n2]
plot([ww1[0], ww2[0]],[ww1[1], ww2[1]],'--k')
#plot([w_teacher1[0], w_teacher2[0]],[w_teacher1[1], w_teacher2[1]],'--g')

x1=np.arange(-3,3,2)
plot(x1,-w_teacher[0]/w_teacher[1]*x1 )

show()
 
