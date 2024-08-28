
"""
Created on Tue Aug 19 10:42:02 2024

@author: Miguel Potes
"""
import numpy as np
import numpy.matlib 
import matplotlib.pyplot as plt
import glob
import pandas as pd

basepath= "D:/TRABALHO/ICT/2022/fisica_porto/calculo_zub2/"
procura=glob.glob("irgason_fluxos_todos.txt")
files=procura[0]
print(basepath+files)

mat = np.loadtxt(files,dtype='str',delimiter=',',skiprows=2)
hour=np.zeros(len(mat[:,0]),dtype=int)
for i in range(len(mat[:,0])):
    aux=mat[i,0]
    x = aux.split(" ", 1)
    y = x[1].split(":", 2)
    hour[i]=int(y[0])

mat_array=np.zeros([1776,25],dtype=float)
for j in range(len(mat[:,0])):
    mat_array[j] = [float(string) for string in mat[j,1:26]]
   
#print(mat_array[140,5])

fig = plt.figure(figsize =(20, 10))

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 22

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#plotting the points
plt.plot(mat_array[:,5])
# naming the x axis
plt.xlabel('2019')
# naming the y axis
plt.ylabel('evap [mm]')
# giving a title to my graph
#plt.title('Lets go man!')
# function to show the plot
#plt.show()
#save plot
plt.savefig(basepath+'evap.png')

#close plot
plt.clf()

fig = plt.figure(figsize =(20, 10))
#plotting the points
plt.hist(mat_array[:,5],color='skyblue', edgecolor='black')
# naming the x axis
plt.xlabel('evap [mm]')
# naming the y axis
plt.ylabel('frequency')
# giving a title to my graph
#plt.title('Lets go man!')
# function to show the plot
#plt.show()
#save plot
plt.savefig(basepath+'evap_hist.png')

#close plot
plt.clf()

nbins=np.arange(24)

mat30=np.zeros([74,24],dtype=float)
mat_hour=np.zeros([37,24],dtype=float)

for k in range(len(nbins)):
    aux2 = np.where(hour==k)
    #print(aux2)
    mat30[:,k] = mat_array[aux2,5]
    mat30[np.where(np.isnan(mat30[:,k])),k]=np.inf 
    count=0
    
    for kk in range(0,len(aux2[0])-1,2):
        aux3 = mat30[kk,k]+mat30[kk+1,k]
        mat_hour[count,k] = aux3
        count=count+1
        
data = [mat30[:,0], mat30[:,1], mat30[:,2],\
        mat30[:,3],mat30[:,4],mat30[:,5],\
        mat30[:,6],mat30[:,7],mat30[:,8],\
        mat30[:,9],mat30[:,10],mat30[:,11],\
        mat30[:,12],mat30[:,13],mat30[:,14],\
        mat30[:,15],mat30[:,16],mat30[:,17],\
        mat30[:,18],mat30[:,19],mat30[:,20],\
        mat30[:,21],mat30[:,22],mat30[:,23]]

hourly_data = [mat_hour[:,0], mat_hour[:,1], mat_hour[:,2],\
        mat_hour[:,3],mat_hour[:,4],mat_hour[:,5],\
        mat_hour[:,6],mat_hour[:,7],mat_hour[:,8],\
        mat_hour[:,9],mat_hour[:,10],mat_hour[:,11],\
        mat_hour[:,12],mat_hour[:,13],mat_hour[:,14],\
        mat_hour[:,15],mat_hour[:,16],mat_hour[:,17],\
        mat_hour[:,18],mat_hour[:,19],mat_hour[:,20],\
        mat_hour[:,21],mat_hour[:,22],mat_hour[:,23]]

fig = plt.figure(figsize =(20, 10))

plt.boxplot(data,positions=nbins)

plt.savefig(basepath+'evap_boxplot_30min.png')

#close plot
plt.clf()

fig = plt.figure(figsize =(20, 10))

fig = plt.boxplot(hourly_data,positions=nbins,patch_artist=True,\
      flierprops={'marker': 'o', 'markersize': 10, 'markerfacecolor': 'mediumblue'})

top_points = fig["fliers"][0].get_data()[1]
bottom_points = fig["fliers"][2].get_data()[1]
outliers = [flier.get_ydata() for flier in fig["fliers"]]
#boxes = [box.get_ydata() for box in bp["boxes"]]
medians = [median.get_ydata() for median in fig["medians"]]
whiskers = [whiskers.get_ydata() for whiskers in fig["whiskers"]]

#print("Outliers: ", outliers)
# print("Boxes: ", boxes)
#print("Medians: ", medians)
#print("Whiskers: ", whiskers)

#naming the x axis
plt.xlabel('Time of the day',labelpad=8)
# naming the y axis
plt.ylabel('Evaporation rate [mm $h^{-1}$]',labelpad=18)

plt.savefig(basepath+'evap_boxplot_hourly.png')

#close plot
plt.clf()


