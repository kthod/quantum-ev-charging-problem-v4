from qiskit import *
from qiskit.circuit import Parameter
import numpy as np
from scipy.optimize import minimize
from typing import Literal
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
import time
from VQE_Class import*
from CompressedVQE_Class import*
from CompressedVQE_RPA_Class import*
from ClusterVQE_Class import*
from ClusterCompressedVQE_Class import*
import matplotlib.colors as mcolors

epsilon = np.array(12*[26880,26880,26880,7680,23040,23040,7680,7680])/1000
de = 1*np.array(12*[3,3,3,1,3,3,1,1])


Q = []



def get_qubomat3(eps,delta,Horizon,V,DT):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    d = np.ones(2*Horizon)
    #print(eps)
# Then, change the elements after the first k elements to 1000
    d[2*delta:] = 0
    #print(d)
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    p = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    matrix = np.zeros((len(d),len(d)))
    nvar = len(matrix)
    
    for i in range(len(d)):
        for j in range(i,len(d)):
            if i == j:
                matrix[i][i] = 256*(V**2)*(DT**2) * ((p[i]**2)*(d[i]**2)) - 16*2*eps*V*DT*d[i]*p[i]# + 256*p[i]**2
            else:
                matrix[i][j] = 256*2*(V**2)*(DT**2) *p[i]*p[j]*d[i]*d[j]
    return matrix


def get_qubomat4(evi,evj,deltai,deltaj,Horizon):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    di = np.ones(2*Horizon)
    dj = np.ones(2*Horizon)
# Then, change the elements after the first k elements to 1000
    di[2*deltai:] = 0
    dj[2*deltaj:] = 0
   
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    pi = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    pj = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    matrix = np.zeros((len(di),len(di)))
    nvar = len(matrix)
    if evi==evj:
        for i in range(0,len(di),2):
 
            matrix[i:(i+2),i:(i+2)] =  256*np.array([[(pi[i]**2)*di[i]**2,2*(pi[i]*pi[i+1])*(di[i]*di[i+1])],[0,(pi[i+1]**2)*di[i+1]**2]]) 
    else:
        for i in range(0,len(di),2):
            matrix[i:(i+2),i:(i+2)] =2*256*np.array([[(pi[i]**2)*di[i]*dj[i],(pi[i]*pi[i+1])*(di[i]*dj[i+1])],[(pi[i]*pi[i+1])*(dj[i]*di[i+1]),(pi[i+1]**2)*di[i+1]*dj[i+1]]])           
    return matrix


def get_qubomat_energy_limit1(evi,evj,deltai,deltaj,Horizon,V,C):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    di = np.ones(2*Horizon)
    dj = np.ones(2*Horizon)
# Then, change the elements after the first k elements to 1000
    di[2*deltai:] = 0
    dj[2*deltaj:] = 0
   
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    pi = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    pj = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    matrix = np.zeros((len(di),len(di)))
    nvar = len(matrix)
    if evi==evj:
        for i in range(0,len(di),2):
 
            matrix[i:(i+2),i:(i+2)] =  np.array([[256*V**2*(pi[i]**2)*di[i]**2 - 2*C*V*16*pi[i]*di[i] ,256*V**2*2*(pi[i]*pi[i+1])*(di[i]*di[i+1])],[0, 256*V**2*(pi[i+1]**2)*di[i+1]**2 - 2*C*V*16*pi[i+1]*di[i+1]]]) 
    else:
        for i in range(0,len(di),2):
            matrix[i:(i+2),i:(i+2)] =2*256*V**2*np.array([[(pi[i]**2)*di[i]*dj[i],(pi[i]*pi[i+1])*(di[i]*dj[i+1])],[(pi[i]*pi[i+1])*(dj[i]*di[i+1]),(pi[i+1]**2)*di[i+1]*dj[i+1]]])           
    return matrix

def get_qubomat_energy_limit2(V,C,Horizon):
    # Adjacency is essentially a matrix which tells you which nodes are connected.

   
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    pi = np.array([2 ** ((i+3) % 3) for i in range(4*Horizon)])
    pj = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    matrix = np.zeros((4*Horizon,4*Horizon))
    nvar = len(matrix)
 
    for i in range(0, 4*Horizon,4):
        matrix[i:(i+4),i:(i+4)] =  np.array([[256*V**2*(pi[i]**2)- 2*C*V*16*pi[i] ,256*V**2*2*(pi[i]*pi[i+1]),256*V**2*2*(pi[i]*pi[i+2]),256*V**2*2*(pi[i]*pi[i+3])],
                                            [0, 256*V**2*(pi[i+1]**2) - 2*C*V*16*pi[i+1], 256*V**2*2*(pi[i+1]*pi[i+2]), 256*V**2*2*(pi[i+1]*pi[i+3])], 
                                            [0, 0, 256*V**2*(pi[i+2]**2) - 2*C*V*16*pi[i+2], 256*V**2*2*(pi[i+2]*pi[i+3])],
                                            [0, 0, 0, 256*V**2*(pi[i+1]**2) - 2*C*V*16*pi[i+1]]  ]) 
    
    return matrix

def get_qubomat_energy_limit3(delta, V,Horizon):
    # Adjacency is essentially a matrix which tells you which nodes are connected.
    d = np.ones(2*Horizon)
    
# Then, change the elements after the first k elements to 1000
    d[2*delta:] = 0
    
   
    T = np.array([(Horizon - i//2)/Horizon for i in range(2*Horizon)])
    pi = np.array([2 ** ((i+1) % 2) for i in range(2*Horizon)])
    pj = np.array([2 ** ((i+3) % 4) for i in range(4*Horizon)])
    matrix = np.zeros((len(d),4*Horizon))
    nvar = len(matrix)
   

 
    for i in range(0, Horizon):
        matrix[i*2:(i+1)*2,i*4:(i+1)*4] = 2*256*V**2*np.array([[pi[i]*d[i]*pj[i], pi[i]*d[i]*pj[i+1], pi[i]*d[i]*pj[i+2], pi[i]*d[i]*pj[i+3]],
                                                                [pi[i+1]*d[i+1]*pj[i], pi[i+1]*d[i+1]*pj[i+1], pi[i+1]*d[i+1]*pj[i+2], pi[i+1]*d[i+1]*pj[i+3]]])            
   
    return matrix
#Q = get_qubomat(a,S)

# Q = np.array([[1,1,0,0],
#      [0,1,0,0],
#      [0,0,1,1],
#      [0,0,0,1]]
# )
# Q1 = Q[0:5,0:5]
# Q2 = Q[5:10,5:10]
# Q3 = Q[0:5,5:10]
# print(Q)


evs = 2
V=0.240
DT = 1
Horizon = 4
C = 6*3840
C=30
#def get_qubomat4():
Q  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))
Q1  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))
Q2  = np.zeros((evs*Horizon*2 + 4*Horizon,evs*Horizon*2 + 4*Horizon))

Q  = np.zeros((evs*Horizon*2,evs*Horizon*2 ))
Q1  = np.zeros((evs*Horizon*2,evs*Horizon*2))
#Q2  = np.zeros((evs*Horizon*2*Horizon,evs*Horizon*2))
#Q = get_qubomat3(11520,3,Horizon,V,DT)
for ev in range(evs):
    Q[ev*Horizon*2:(ev+1)*Horizon*2,ev*Horizon*2:(ev+1)*Horizon*2] = get_qubomat3(epsilon[ev],de[ev],Horizon,V,DT)

for evi in range(evs):
    for evj in range(evi,evs):
        Q1[evi*Horizon*2:(evi+1)*Horizon*2,evj*Horizon*2:(evj+1)*Horizon*2] = get_qubomat4(evi,evj,de[evi],de[evj],Horizon)

        #print(Q1[evi*Horizon*2:(evi+1)*Horizon*2,evj*Horizon*2:(evj+1)*Horizon*2])
for evi in range(evs):
    for evj in range(evi,evs):
        Q2[evi*Horizon*2:(evi+1)*Horizon*2,evj*Horizon*2:(evj+1)*Horizon*2] = get_qubomat_energy_limit1(evi,evj,de[evi],de[evj],Horizon, V, C)

Q2[evs*Horizon*2: evs*Horizon*2 + Horizon*4, evs*Horizon*2 : evs*Horizon*2 + Horizon*4] = get_qubomat_energy_limit2(V, C, Horizon)
print(Q2[evs*Horizon*2: evs*Horizon*2 + Horizon*4, evs*Horizon*2 : evs*Horizon*2 + Horizon*4])
#print(Q1)
for ev in range(evs):
    Q2[ev*Horizon*2:(ev+1)*Horizon*2, evs*Horizon*2 : evs*Horizon*2 + Horizon*4] = get_qubomat_energy_limit3(de[ev], V,Horizon)
    print(Q2[ev*Horizon*2:(ev+1)*Horizon*2, evs*Horizon*2 : evs*Horizon*2 + Horizon*4])
Q =  1000*Q + Q1#+0.5*Q2
# inst = VQE(Q,2)
# inst.optimize()
# inst.show_solution()
# for i in range(evs*(evs-1)/2):
#     Q.append() 

def ret_schedule(solution):
    sched = np.zeros((evs,Horizon))
    for i in range(evs):
        sol = solution[i]
        #bitstring = np.array([int(x) for x in sol])
        for j in range(Horizon):
            if j<de[i]:
                sched[i,j] = 16*int(sol[2*j:2*j+2],2)
            else:
                sched[i,j] = 0

    print(sched)
    return sched

def reshape_solution(sol):
    solution = []
    for i in range(evs):
        solution.append(sol[i*2*Horizon:(i+1)*2*Horizon])
        
    return solution
  
#print(Q)


# inst = CompressedVQE(Q,2,na=2*Horizon)
# inst.optimize(number_of_experiments = 5,maxiter=1200)
# solution = inst.show_solution()

# solution = reshape_solution(solution)
# ret_schedule(solution)

    #####################################
layers = [2,4,6,8,10]
F = [(0.001,0.01),(0.01,0.01),(0.01,0.1)]
markers = ['*','^','o']
min_cost = np.zeros((len(layers),5))
full_cost = np.zeros((len(layers),5))
for j in range(len(markers)+1):
    for i in range(len(layers)):
        inst_min = CompressedVQE(Q,layers=layers[i],na=2)
        label_min = r'Compressed VQE'
        marker_min = 's'
        # label_full = r'VQE'
        if (j>0):
            inst_min.add_noise(t1 = 50e3, t2 = 70e3, prob_1_qubit = F[j-1][0] ,prob_2_qubit = F[j-1][1], p1_0 = 0.1, p0_1 = 0.1)
            label_min = f'Compressed VQE $F_1 = {100-10*F[j-1][0]}\%, F_2 = {100-10*F[j-1][1]}\%$'
            marker_min =  markers[j-1]

        # inst_full = VQE(Q,layers=layers[i])
        # if (j==1):
        #     inst_full.add_noise(t1 = 50e3, t2 = 70e3, prob_1_qubit = 0.001 ,prob_2_qubit = 0.01, p1_0 = 0.1, p0_1 = 0.1)
        #     label_full = r'VQE $F_1 = 99.99%, F_2 = 99,9%$'
        for k in range(5):

            inst_min.optimize(n_measurements = 10000,number_of_experiments = 1,maxiter=500)
            solution = inst_min.show_solution()

            min_cost[i,k] = inst_min.optimal_cost
            solution = reshape_solution(solution)
            ret_schedule(solution)

            # inst = CompressedVQE_RPA(Q,2,na=2)
            # inst.optimize(number_of_experiments = 5,maxiter=200)
            # solution = inst.show_solution()

            # solution = reshape_solution(solution)
            # ret_schedule(solution)


            # inst = split_compressedVQE(Q,2,na=2,group=evs//16)
            # inst.optimize(number_of_experiments = 5,maxiter=800)
            # solution = inst.show_solution()
            # solution = reshape_solution(solution)
            # ret_schedule(solution)

            # inst = splitVQE(Q,2,evs)
            # inst.optimize(number_of_experiments = 5,maxiter=800)
            # solution = inst.show_solution()

            # solution = reshape_solution(solution)
            # ret_schedule(solution)


            # inst_full.optimize(n_measurements = 1000, number_of_experiments = 1, maxiter = 600)
            # solution = inst_full.show_solution()
            # full_cost[i,k] = inst_full.optimal_cost
            # solution = reshape_solution(solution)
            # ret_schedule(solution)

            # inst_min.plot_evolution()
            # inst_full.plot_evolution()

        

    upper_bound_min = [np.max(min_cost[i, :]) for i in range(len(layers))]
    lower_bound_min = [np.min(min_cost[i, :]) for i in range(len(layers))]
    mean_min = [np.mean(min_cost[i, :]) for i in range(len(layers))]

    # upper_bound_full = [np.max(full_cost[i, :]) for i in range(len(layers))]
    # lower_bound_full = [np.min(full_cost[i, :]) for i in range(len(layers))]
    # mean_full = [np.mean(full_cost[i, :]) for i in range(len(layers))]

    plt.fill_between(layers, upper_bound_min, lower_bound_min, alpha=0.4)
    plt.plot(layers, mean_min, marker=marker_min, linestyle='--', label=label_min)
    # plt.fill_between(range(3), upper_bound_full, lower_bound_full, alpha=0.4)
    # plt.plot(range(3), mean_full, marker='^', linestyle='--', label=label_full)

plt.xlabel("Layers L")
plt.ylabel("Cost Function")
plt.title("Noise Simulation")
plt.legend()
plt.show()


solutions =10
# scatter = inst_full.get_solution_distribution(solutions = 10, shots = 1000)
#scatter = inst.get_solution_distribution()
scatter = inst_min.get_solution_distribution(solutions = 10, shots = 10000)

norm = mcolors.Normalize(vmin=0, vmax=1)
plt.colorbar(mappable=scatter, norm=norm, label='Fraction of solutions')
plt.yticks(range(1,8))
plt.xlabel('Cost Function')
plt.ylabel('Optimization run')
plt.title('Distribution of solutions for 4 EV 4 timesteps')
plt.grid(True)
plt.legend()
plt.show()

