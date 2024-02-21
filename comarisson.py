from quantum_MPC import *
from classical_algorithms import *
import numpy as np
from matplotlib import pyplot as plt

def simulation (e,d, a, energy_limit,alg):
    departure = a +d
    Daily_schedule = np.zeros((total_number_of_EVs, 6//DT))
    epsilon = e.copy()
    de = d.copy()
    for t in range(6//DT):
        temp_e = []
        temp_d = []
        active_EVs = []
        for ev in range(total_number_of_EVs):
            if a[ev] <= t < departure[ev]:
                active_EVs.append(ev)
                temp_e.append(epsilon[ev])
                temp_d.append(de[ev])

        alg.set_attributes(temp_e,temp_d,energy_limit)
        alg.optimize()
        optimized_schedules = alg.get_sched()
        for i in range(len(active_EVs)):
            epsilon[active_EVs[i]]-=240*DT*optimized_schedules[i,0]/1000
            de[active_EVs[i]]-=1
            Daily_schedule[active_EVs[i],t] = optimized_schedules[i,0]
    
    return Daily_schedule

total_number_of_EVs = 10
DT = 1
T=4
energy_limit=60
Daily_schedule_fcfs = np.zeros((total_number_of_EVs, 8//DT))
Daily_schedule_edf = np.zeros((total_number_of_EVs, 8//DT))
Daily_schedule_llf = np.zeros((total_number_of_EVs, 8//DT))
Daily_schedule_qmpc = np.zeros((total_number_of_EVs, 8//DT))

epsilon = np.array([26880,26880,26880,7680,23040,23040,7680,7680,7680,7680])/1000
de = 1*np.array([3,3,3,1,3,3,1,1,1,1])
# epsilon = np.array([26880,26880,26880,7680])/1000
# de = 1*np.array([3,3,3,1])

print(epsilon/1000)
print(energy_limit)
print((epsilon/1000)*1000)

# optimized_schedules_qmpc = quantum_mpc(epsilon , de, energy_limit, T, DT)
# optimized_schedules_fcfs = optimize_charging_schedule(epsilon , de, energy_limit, T, DT)
# optimized_schedules_llf = optimize_charging_schedule_llf(epsilon , de, energy_limit, T, DT)
# optimized_schedules_edf = optimize_charging_schedule_edf(epsilon , de, energy_limit, T, DT)


# print(optimized_schedules_fcfs)
# print(optimized_schedules_llf)
# print(optimized_schedules_edf)
# print(optimized_schedules_qmpc)
print("----------------------------")




a = 1*np.array([0,0,0,0,1,2,3,4,5,5])
departure = a +de
energy_limits = [5,15,30,50,60,100]
demand_met_fcfs = []
demand_met_llf = []
demand_met_edf = []
demand_met_qmpc = []
j = 0

name = ["QMPC","FCFS","EDF","LLF"]
algorithms = [Quantum_MPC(epsilon,de,energy_limit,T,DT),FCFS(epsilon,de,energy_limit,T,DT),EDF(epsilon,de,energy_limit,T,DT),LLF(epsilon,de,energy_limit,T,DT)]

for i in range(len(algorithms)):  
    demand_met = []
    for energy_limit in energy_limits:
        print("==========================================")
        print(energy_limit)
        print("==========================================")
        

        Daily_schedule = simulation(epsilon,de,a,energy_limit,algorithms[i])

        epsilon = np.array([26880,26880,26880,7680,23040,23040,7680,7680,7680,7680])/1000
        de = 1*np.array([3,3,3,1,3,3,1,1,1,1])

        supply = np.sum(240*DT*Daily_schedule/1000,axis = 1)

        diff = supply - epsilon
        diff[diff>0] = 0

        supply = epsilon + diff

        demand_met.append(sum(supply)/sum(epsilon)*100)

    plt.plot(energy_limits,demand_met,label=name[i])
        # print(Daily_schedule_fcfs)
        # print(Daily_schedule_edf)
        # print(Daily_schedule_llf)






# print(Daily_schedule_fcfs)
# print(Daily_schedule_edf)
# print(Daily_schedule_llf)
# print(demand_met_fcfs)
# print(demand_met_llf)
# print(demand_met_edf)
# plt.plot(energy_limits,demand_met_fcfs,label="fcfs")
# plt.plot(energy_limits,demand_met_llf,label="llf")
# plt.plot(energy_limits,demand_met_edf,label="edf")


plt.legend()
plt.show()



    #print(optimized_schedules_edf)