#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import msprime
from IPython.display import display, SVG
import numpy as np
import matplotlib.pyplot as plt
import tskit
from random import choices
import statistics as stat
import math
from scipy import stats


# In[ ]:


#General parameters being used 
mut_rate = 0.000001
rec_rate = 0.00001
sequence_length= 10000
nsamples = 1000
ini_size_ooa = 1000
ini_size_afr = 10000
mig_rate = 0
bneck_time = 1000


# In[ ]:


#Set up demography model with a population split
def demography_model(ini_size,growth_rate,sample_size,rec_rate,seq_length, mut_rate, mig_rate):
    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size=10000)
    demography.add_population(name="AFR",initial_size =10000)
    demography.add_population(name="OOA",initial_size =ini_size,growth_rate=growth_rate)
    demography.set_migration_rate('AFR','OOA',mig_rate)
    demography.add_population_split(time=bneck_time, derived=["AFR", "OOA"], ancestral="Ancestral")
    ts = msprime.sim_ancestry(samples={"AFR":sample_size, "OOA": sample_size},recombination_rate=rec_rate,
                              sequence_length= seq_length,demography=demography)
    #add mutations onto the model
    mts = msprime.mutate(ts, rate=mut_rate)
    #obtain the mutation matrix and split it corresponding to the two split populations
    X = mts.genotype_matrix().transpose()
    X_afr = X[nsamples*2:,:]
    X_ooa = X[:nsamples*2,:]
    return X_afr, X_ooa, mts


# In[ ]:


#map corresponding to Linkage Disequilibrium
def show_corr_mat(X):
    plt.matshow(abs(np.corrcoef(X)), cmap = 'Reds')
    plt.show()


# In[ ]:


#finding the frequencies of each row of the mutation matrix
def find_freq(X): 
    N = len(X)
    f = np.zeros(len(X[0]))
    for i in range(len(X[0])):
        f[i] = (np.sum(X[:,i])/N)
        
    return f


# In[ ]:


def generate_func_B(freq, func_number): #returns B and list of functional SNPs
    B = []
    func_B = np.sort(np.random.choice([i for i in range(len(freq))],
                              size = func_number, replace = False))
    for i in range(len(freq)):
        f1mf=freq[i]*(1-freq[i])
        if(f1mf==0):
            sd = 0
        else:
            sd = np.power(f1mf,-0.5)
        B.append(np.random.normal(loc = 0.0, scale = sd**2))
    B = np.array(B)
    return B, func_B


# In[ ]:


#generate the phenotype y=XB+epsilon 
def y_vector(beta,X,h):
    N = len(X)
    y = np.zeros(N)
    for i in range(N-1):
        y[i] = np.dot(beta,X[i,:])
    var_Y0 = np.var(y)
    sigmasq = ((var_Y0*(1-h))/(h))
    epsilon = (np.random.normal(size=N,loc = 0.0, scale = np.sqrt(sigmasq)))
    y = y + epsilon
    return y, epsilon, sigmasq


# In[ ]:


#estimate the heritability in one population from another
def gaining_h(beta1,X1,h1,X2):
    y1, epsilon, sigmasq = y_vector(beta1,X1,h1)
    N = len(X2)
    y2 = np.zeros(N)
    for i in range(N):
        y2[i]= np.dot(beta1,X2[i,:])
    varY2 = np.var(y2)
    h2 = varY2/(varY2+sigmasq)
    return h2


# In[ ]:


#heritability experiment when fixing the bottleneck population size
def heritability_n1(N,h1,g_rate,n0_fix):
#generate original population with mutation matrix and frequencies
    h = []
    x = np.logspace(np.log10(0.1*N), np.log10(10*N), num=30, endpoint=True)
    SE = np.zeros((len(x)))
    n = 0
    for i in x:
        h_test=[]
        for j in range(50):
#generate new population with 0 growth rate
            if g_rate == 0:
                X_afr,X_ooa,mts = demography_model(i,0,nsamples,rec_rate,sequence_length,mut_rate,mig_rate)
#generate new population with growth rate
            elif g_rate == 1:
                growth_rate = -math.log(n0_fix/i)/1000
                X_afr,X_ooa,mts = demography_model(i,growth_rate,nsamples,rec_rate,sequence_length,mut_rate,mig_rate)
#frequencies, betas and reduced matrix for new population
            freq_ooa = find_freq(X_ooa)
            freq_afr = find_freq(X_afr)
            B_ooa,func_B_ooa = generate_func_B(freq_ooa,int(len(freq_ooa)))
#betas and reduced matrix for original population
            B_afr,func_B_afr = generate_func_B(freq_afr,int(len(freq_afr)))
            h_test.append(gaining_h(abs(B_afr),X_afr,h1,X_ooa))
        SE[n] = stats.sem(h_test)
        n+= 1
        h.append(np.mean(h_test))
    return h,x,SE


# In[ ]:


#heritability experiment for when fixing the size of the population after t generations
def heritability_n0(N,h1,g_rate,n1_fix):
#generate original population with mutation matrix and frequencies
    h = []
    x = np.logspace(np.log10(0.1*N), np.log10(10*N), num=30, endpoint=True)
    SE = np.zeros((len(x)))
    n=0
    for i in x:
        h_test=[]
        for j in range(50):
#generate new population with 0 growth rate
            if g_rate == 0:
                X_afr,X_ooa,mts = demography_model(i,0,nsamples,rec_rate,sequence_length,mut_rate,mig_rate)
#generate new population with growth rate
            elif g_rate == 1:
                growth_rate = -math.log(i/n1_fix)/1000
                X_afr,X_ooa,mts = demography_model(n1_fix,growth_rate,nsamples,rec_rate,sequence_length,mut_rate,mig_rate)
#frequencies, betas and reduced matrix for new population            
            freq_ooa = find_freq(X_ooa)
            freq_afr = find_freq(X_afr)
            B_ooa,func_B_ooa = generate_func_B(freq_ooa,int(len(freq_ooa)))
#betas and reduced matrix for original population
            B_afr,func_B_afr = generate_func_B(freq_afr,int(len(freq_afr)))
            h_test.append(gaining_h(abs(B_afr),X_afr,h1,X_ooa))
        SE[n] = stats.sem(h_test)
        n += 1
        h.append(np.mean(h_test))
    return h,x,SE


# In[ ]:


#return the effective population size by calculating the harmonic mean
def EPS(start_value, x, N0_true,fix_value):
    length = int(len(x))
    harm_mean = np.zeros(length)
    x_index = 0
    #for a fixed bottleneck population size
    if N0_true == 1:
        for i in x:
            growth_rate = -math.log(i/fix_value)/1000
            values = start_value * np.exp(growth_rate * np.arange(1000))
            harm_mean[x_index] = stat.harmonic_mean(values)
            x_index += 1
    #for a fixed population size after t generations
    elif N0_true == 0:
        for i in x:
            growth_rate = -math.log(fix_value/i)/1000
            values = start_value * np.exp(growth_rate * np.arange(1000))
            harm_mean[x_index] = stat.harmonic_mean(values)
            x_index += 1
    return harm_mean      


# In[ ]:


#find the coefficient of determination for a point
def r_squared(harm_mean, h):
    slope,intercept,r_value,p_value,std_err = stats.linregress(harm_mean,h)
    print(r_value**2)
    return r_value**2
    

