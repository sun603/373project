import numpy as np
import pickle

d1 = np.loadtxt("../occupancy_data/datatest.txt",
    dtype={ 'names' : ("order","date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"),
            'formats':("S10"    ,"S25"   ,"f"          ,"f"       ,"f"    ,"f"  ,"f"            ,"i")},
    delimiter=',',skiprows=1,)

d2 = np.loadtxt("../occupancy_data/datatest2.txt",
    dtype={ 'names' : ("order","date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"),
            'formats':("S10"    ,"S25"   ,"f"          ,"f"       ,"f"    ,"f"  ,"f"            ,"i")},
    delimiter=',',skiprows=1,)

d3 = np.loadtxt("../occupancy_data/datatraining.txt",
    dtype={ 'names' : ("order","date","Temperature","Humidity","Light","CO2","HumidityRatio","Occupancy"),
            'formats':("S10"    ,"S25"   ,"f"          ,"f"       ,"f"    ,"f"  ,"f"            ,"i")},
    delimiter=',',skiprows=1,)

dall = np.concatenate([d1, d2, d3])

np.random.shuffle(dall)

pos = int(len(dall) * 0.3)
pos += (len(dall) - pos) %5
testset = dall[:pos]
training = dall[pos:]

# k-folder value k
k = 5
# kf = [ [] for i in range(k)]
kf = np.split(training,5)

with open('testset.p','wb') as fl:
    pickle.dump(testset,fl)

with open('kf.p','wb') as fl:
    pickle.dump(testset,fl)
