import os
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(dir_path+'/train_data')
csv_files = ['train_data/'+f for f in files if f[-4:]=='.csv' ]

NUM_FEATURES = 18 #0-2,3-24,25

l = 10#num principal components
y = np.zeros(l)
w = np.random.normal(size=[l,NUM_FEATURES],scale=0.001)
ln = 0.001 #learning reju
num_epochs = 10

def read_data(files_list):
	print(files_list)
	num_datapoints = 0
	for file_name in files_list:
		with open(file_name, newline='') as f:
			r = csv.reader(f)
			num_datapoints += len(list(r)) -1 #ignore titles rownum_datapoints
	print(num_datapoints)
	print(NUM_FEATURES)
	data  = np.zeros((num_datapoints,NUM_FEATURES))
	i = 0
	for file_name in files_list:
		with open(file_name, newline='') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='|')
			next(reader)
			for row in reader:
				data_list = row[6:24]
				for j in range(len(data_list)):
					data[i][j] = data_list[j]
				i+=1
	return data

def get_total_variance(data):
	d = np.transpose(data)
	cov_matrix = np.cov(d)
	num_rows = cov_matrix.shape[0]
	print(cov_matrix.shape)
	result = 0
	for i in range(num_rows):
		for j in range(num_rows):
			if i == j:
				result+= cov_matrix[i,j]
	return result

def update_weights(weights,x,y,ln):
	num_dest = weights.shape[0] #cols
	num_src = weights.shape[1]	#rows
	for j in range(num_dest):
		for i in range(num_src):

			d_w = delta_w(x,y,weights,i,j,ln)
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
		col_std = np.std(matrix[:,col_i])
		col_mean = np.mean(matrix[:,col_i])
		for row_i in range(num_row):
			val = matrix[row_i,col_i]
			matrix[row_i,col_i] = (val-col_mean)/col_std
	return matrix

def plot_binary(pc_data):
	x = []
	y = []
	for row in range(pc_data.shape[0]):
		x.append(pc_data[row,0])
		y.append(pc_data[row,1])
	plt.scatter(x,y,color='r')
	plt.show()
#data = np.array(data)

#FEATURES: SPEED	TRACK_POSITION	ANGLE_TO_TRACK_AXIS	TRACK_EDGE_0_TO_17

def plot_variance_explained(data,pc_data):
	N = get_total_variance(data) #sums variance of each of its features
	num_pc = pc_data.shape[1]
	y=[]
	for principal_component in range(num_pc):
		var_pc = np.var(pc_data[:,principal_component])
		var_explained = var_pc/N
		y.append(var_explained)
	plt.scatter(np.arange(num_pc),y)
	plt.show()

#for rand_i in range(520):
def train(data,w,ln):
	g=0
	for epoch in range(num_epochs):
		rand_is = np.random.choice(len(data),len(data),replace=False)
		for rand_i in rand_is:
			g+=1
			x = np.array(data[rand_i]) #random sample [21x1] vector w = lx21 matrix y = 1xl matrix
			y = np.matmul(w,x)
			update_weights(w,x,y,ln)

			if math.isnan(y[0]):
				print(g)
				break
	return w


def PCA(data,w,l): #converts dataset to principal components
	num_data_points = data.shape[0]
	print(num_data_points,l)
	result = np.zeros((num_data_points,l))
	for point in range(num_data_points):
		x = np.array(data[point])
		result[point] = np.matmul(w,x)
	return result

train_list = csv_files[0:2]

test_list = []
test_list.append(csv_files[-1])
train_data = zero_centre(np.array(read_data(train_list)))
test_data = zero_centre(np.array(read_data(test_list)))

train(train_data,w,ln)
pc_train = PCA(train_data,w,l)
pc_test = PCA(test_data,w,l)
print('total variance in training data:',get_total_variance(train_data))
print('total variance in the principal components',get_total_variance(pc_train))
print('variance in PC1',np.var(pc_train[:,0]))
print('variance in PC2',np.var(pc_train[:,1]))
plot_variance_explained(test_data,pc_test)
plot_binary(pc_train)
plot_binary(pc_test)



