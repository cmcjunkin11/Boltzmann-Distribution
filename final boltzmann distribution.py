# -*- coding: utf-8 -*-
"""

Analyzing the Boltzmann distribution

@Chris McJunkin


"""



from numpy import (
    linspace,array,zeros,log,exp,sin,cos,sqrt,pi,e, 
    ones, arange, zeros, real, imag, sign, shape, dot, size,
    mean, asarray
    )

import numpy as np

from numpy.random import rand,randint

from matplotlib.pyplot import (
    plot,xlabel,ylabel,legend,show, figure, subplot, title, tight_layout, stem, pcolormesh,
    get_cmap, semilogy
    )   

import matplotlib.pyplot as plt

# Model the state as a list in python 
N = 20**2
q = 10
solid = [q]*N
L = 10**5


def exchange(cell, N, L):

# Exchange energy between a specified number of random pairs. Loop over number of exchanges, L:
    for i in range(L):
        take = randint(0,N-1) # random index 
        give = randint(0,N-1) # another random index
        while cell[take] == 0:
            take = randint(0,N-1)
        cell[take] = cell[take] - 1
        cell[give] = cell[give] + 1        
        
    return cell

# Use random.randint()
# https://www.w3schools.com/python/module_random.asp
# Get two random numbers<N
# Subtract 1 from donor and add one to receiver. If donor has q=0, pick another cell.

# For sample() count the number of cells with q = [0, 1, 2, ...,

def sample(cell, N):  # sample energy distribution

    # compute 
    # 1 qmax
    qmax = 0
    for i in range(len(cell)):
        if cell[i] > qmax:
            qmax = cell[i]
        
    # 2 number of cells with q = [1, 2, 3, ... qmax] use solid.count(q)
    counts = [0]*(qmax + 1)
    for q in range(qmax + 1):
        counts[q] = cell.count(q)
        
    # 3 probability = number/float(N)
    prob = []
    for q in range(qmax + 1):
        prob.append(counts[q]/N)

    
    return qmax, counts, prob


exchange(solid,N,L)
m, c, p = sample(solid,N)


"""
---------------------------------
regression model
"""
x = arange(N)
equil = ones(N)*q
plog = []

for i in range(m+1):
    if p[i] > 0:
        plog.append(log(p[i]))
        
xprob = arange(len(plog))


from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(xprob,plog)


"""
-------------theory----------------
avg q values, solve for kT
kt =  1/ln(1/avgQ + 1)

C "can be selected to shift the theoretical curve to match your data" ----- ????????

"note n = 0 --> E_n = 0", P_0 = C, so if we set C equal to numerical value for P0, the numerical


P_t = C exp(-q * ln(1/avgQ + 1))

"""
qavg = q
kT = 1/(log(1/qavg + 1))
C = p[0]
p_t = zeros(m+1)
for i in range(m+1):
    p_t[i] = C * exp(-i/kT)
    


"""
error bars

"""

# https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.07-Error-Bars/
# https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html


err = []
for i in range(m+1):
    err.append(sqrt(p[i]/N))

err_p = asarray(err)

print("Slope: " + str(slope))
print("-1/kT: " + str(-1/kT))
print("Ratio: " + str(-slope*kT)) 
print("Error: " + str(1-(-1*slope*kT)))

"""
Data

"""

title("Probabilities of Energy Distribution (Data)\nCell Number = {}, Avg Energy = {}, Exchanges = {}".format(N, q, L))
plt.errorbar(arange(len(p)), p, yerr = err_p, fmt = ".")
plot(p_t)
legend(("Boltzmann Theory Curve", "Data Probabilities"))
xlabel("Energy Units")
ylabel("Probability")


"""
Linear Regression

"""
figure()
title("Linear Regression of Probabilities on Log Scale\nCell Number = {}, Avg Energy = {}, Exchanges = {}".format(N, q, L))
plot(plog,".")
plot(xprob,slope*xprob + intercept)
legend(("Log of Probabilities","Linear Curve Fit of Log Data"))
xlabel("Energy Units")
ylabel("Log Probability")

"""
Log Graphs
"""
figure()
title("Probability Data and Theory on Log Scale\nCell Number = {}, Avg Energy = {}, Exchanges = {}".format(N, q, L))
plt.errorbar(xprob, plog, yerr = std_err, fmt = ".")
plot(log(p_t))
legend(("Log of Boltzmann Curve","Log of Probabilities"))
xlabel("Energy Units")
ylabel("Log Probability")

"""
Solid Counts
"""
figure()
title("Distribution of Energy per Cell in Data\nCell Number = {}, Avg Energy = {}, Exchanges = {}".format(N, q, L))
plot(x,solid)
xlabel("Cell Number")
ylabel("Energy Units")


"""
Comparison of Linear Regression Log and Theory Log
"""
figure()
title("Linear Curve and Theory Curve in Absolute on Log Scale\nCell Number = {}, Avg Energy = {}, Exchanges = {}".format(N, q, L))
plot(xprob,abs(slope*xprob + intercept))
plot(abs(log(p_t)))
legend(("Absolute Log of Linear Curve","Absolute Log of Boltzmann Curve"))
xlabel("Energy Units")
ylabel("Abs Log Probability")
