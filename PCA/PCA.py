import os
import csv
import numpy as np
import math
dir_path = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(dir_path+'/train_data')
csv_files = ['train_data/'+f for f in files if f[-4:]=='.csv' ]
data = np.array
col_count = 21 #0-2,3-24,25
row_count = 24027-3
l = 10 #num principal components
data = np.zeros((row_count,col_count))
y = np.zeros(l)
w = np.random.normal(size=[l,col_count],scale=0.001)

ln = 0.001 #learning rateju
i = 0
for file_name in csv_files:
	
	with open(file_name, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		next(reader)
		for row in reader:
			data_list = row[3:24]
			for j in range(len(data_list)):
				data[i][j] = data_list[j]
			i+=1
def update_weights(weights,x,y,ln):
	num_dest = weights.shape[0] #cols
	num_src = weights.shape[1]	#rows
	for j in range(num_dest):
		for i in range(num_src):

			d_w = delta_w(x,y,weights,i,j,ln)
			print(d_w)
			weights[j,i] += d_w

def delta_w(x,y,w,source_i,dest_j,ln): #x is a 1x21 vector and y is a 1xl vector
	sum_thing = w[dest_j][source_i]*y[dest_j]
	for k in range(dest_j):
		sum_thing += w[k][source_i]*y[k]
	r = ln*((x[source_i]*y[dest_j]) - (y[dest_j] * sum_thing))
	return r

def zero_centre(matrix): #col is a feature
	num_col = matrix.shape[1]
	num_row = matrix.shape[0]
	for col_i in range(num_col):
		col_mean = np.mean(matrix[:,col_i])
		for row_i in range(num_row):
			val = matrix[row_i,col_i]
			matrix[row_i,col_i] = (val - col_mean)/col_mean
	return matrix

data = zero_centre(np.array(data))

rand_is = np.random.choice(len(data),len(data),replace=False)
#FEATURES: SPEED	TRACK_POSITION	ANGLE_TO_TRACK_AXIS	TRACK_EDGE_0_TO_17
g = 0

y1=[]
y2=[]
y3=[]
y4=[]
y5=[]
y6=[]
y7=[]

#for rand_i in range(520):
for rand_i in rand_is:
	g+=1
	x = np.array(data[rand_i]) #random sample [21x1] vector w = lx21 matrix y = 1xl matrix
	y = np.matmul(w,x)
	update_weights(w,x,y,ln)
	y1.append(y[0])
	y2.append(y[1])
	y3.append(y[2])
	y4.append(y[3])
	y5.append(y[4])
	y6.append(y[5])
	y7.append(y[6])
	if math.isnan(y[0]):
		print(g)
		break

import matplotlib.pyplot as plt

def plot(y):
	plt.scatter(np.arange(len(y)),y)
plot(y1)
plot(y2)
plot(y3)
plot(y4)
plot(y5)
plot(y6)
plot(y7)
plt.show()
print(x)
print(y)
print(w)

#PROBLEMS:
#weights go to infinity.
