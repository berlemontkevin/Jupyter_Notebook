# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 18:28:27 2016

@author: arthur
"""

#****************************************************
#***************Library******************************
#**************************************************** 
import random
import numpy as np
import math
import matplotlib.pyplot as plt

#****************************************************
#**************************************************** 


#****************************************************
#***************Fonction*****************************
#**************************************************** 

def sample(n_iter, cste):#donne un echantillon suivant la loi voulue
    sample = []
#    if cste == 0: return []
    norma = (math.exp((-1)/cste)-math.exp((-20)/cste))
    for i in range(n_iter):
        xi = random.uniform(0,1)
        y = (-1)*cste*math.log(math.exp((-1)/cste)-norma*xi)
        sample.append(y)
    return sample
    


def merde(x): #calcul l'expression de merde que l'on trouve pour ml
    tot = x + (math.exp(-1/x)-20*math.exp(-20/x))/(math.exp(-1/x)-(math.exp(-20/x)))
    return tot
    
def min_dicho(x_mean, x_max, x_min, n):#trouve un valeur de lambda ml à 0.001
    d = (x_max +x_min)/2
    diff = x_mean-merde(d)
    if n == 15:
        return d
    if diff>0: 
        return min_dicho(x_mean, x_max, d, n+1 )
    if diff<0: 
        return min_dicho(x_mean, d, x_min, n+1 )
  

def estimator_ml(data_sample):#donne le ml pour un exchantillon suivant une loi exp decroissante
    n = len(data_sample)
    x_mean = 0
    for i in range(n):
        x_mean += data_sample[i]
    x_mean = x_mean/n
    dicho = min_dicho(x_mean, 25, 0, 0)
    return dicho  
    
    
def sq_error(lambda_ml, cste):
    x = (lambda_ml - cste)**2
    return x
    
def integer(f, lamb):#fonction pour avoir l'integrale d'une fonction de 1 à 20
    sum_f = 0
    delta = 0.1
    n_step = 190
    for i in range(n_step):
        sum_f = sum_f + delta*f(1+(i*delta), lamb)
    return sum_f

def fonction_fisher(y, x):#donne la fonction que l'on doit integrer pour avoir l'information de fisher
    cste_c = math.exp(-1/x)-math.exp(-20/x)  
    cste_a = x + (math.exp(-1/x)-20*math.exp(-20/x))/(math.exp(-1/x)-math.exp(-20/x))
    dens = (math.exp(-y/x)/x**5)    
    tot = (dens/cste_c)*(y-cste_a)**2
    return tot
    
def formule_posterior(y):#fonction que l'on va chercher à minimiser
    info_fisher= integer(fonction_fisher, y)
    derive_fisher = (integer(fonction_fisher, y)-integer(fonction_fisher, y-0.03))/0.03
    tot = y + (math.exp(-1/y)-20*math.exp(-20/y))/(math.exp(-1/y)-(math.exp(-20/y)))-(0.5*derive_fisher/info_fisher)
    return tot
    
    
def min_dicho_bayes(x_mean, x_max, x_min, n):#trouve un valeur de lambda ml à 0.001
    d = (x_max +x_min)/2
    diff = x_mean-merde(d)
    if n == 15:
        return d
    if diff>0: 
        return min_dicho(x_mean, x_max, d, n+1 )
    if diff<0: 
        return min_dicho(x_mean, d, x_min, n+1 )
    
def estimator_bayes(data_sample):
    n = len(data_sample)
    x_mean = 0
    for i in range(n):
        x_mean += data_sample[i]
    x_mean = x_mean/n
    dicho = min_dicho_bayes(x_mean, 25, 0, 0)
    return dicho  
    
#****************************************************
#**************************************************** 


#****************************************************
#***************Programme****************************
#**************************************************** 

n = 10000
n_iter = 500
lamb = 1
ml = 0
sq = 0
data_sample = []
data_se = []
mse = 0
fisher = 0

data_mse = []
data_fisher = []
data_lamb = []


while lamb<17:
    mse = 0
    data_se = []
    data_lamb.append(lamb)
    #on construit une liste de se
    for i in range(n_iter):
        data_sample = sample(n, lamb)
 #       ml = estimator_ml(data_sample)
        ml = estimator_bayes(data_sample)
        sq = sq_error(ml, lamb)
        data_se.append(sq)

    #on fait la moyenne que l'on note mse
    for i in range(n_iter):
        mse += data_se[i]    
    mse = mse/n_iter
    data_mse.append(mse)  
    #on calcule l'information de fisher
    fisher = integer( fonction_fisher, lamb)
    inverse_fisher = 1/(n*fisher)
    data_fisher.append(inverse_fisher)
    lamb += 1
    
#on trace mse et fisher en fonction de lambda
plt.plot(data_lamb,data_mse)
plt.plot(data_lamb, data_fisher)
   
'''
print('---------------------------------------------')
print('Estimator Lambda_ml = ', ml)
print('Squared error SE = ', sq)
print('---------------------------------------------')
'''
