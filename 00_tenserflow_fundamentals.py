import tensorflow as tf
import numpy as np
def create_scalar_const():
    scalar=tf.constant([[10, 22],[10,22]])
    print(scalar.ndim)
def create_var_const():
    change_able_tensor=tf.Variable([10,6])
    print(change_able_tensor[1])
    change_able_tensor[1].assign(4)
    print(change_able_tensor[1])
def random_tesndors():
    random_1=tf.random.Generator.from_seed(-3)
    random_1=random_1.normal(shape=(3,2))
    print(random_1)
    random_2=tf.random.Generator.from_seed(-3)
    random_2=random_2.normal(shape=(3,2))
    print(random_2)
    print(random_1==random_2)
def shuffle_data():
    not_shuffled=tf.constant([[20,21],
                              [44,33],
                              [44,32]])
    tf.random.set_seed(51)
    tf.random.shuffle(not_shuffled)
    print(tf.random.shuffle(not_shuffled,seed=51))
def num_py_with_tenserflow():
    a=tf.ones([10,7])
    print(a)
    b=tf.zeros([20,1])
    print(b)
    #turn numpy arrays in tensors
    #Their dofference is that tensors cam be run on a GPU
    numpy_a=np.arange(1,25, dtype=np.int32)
    print(numpy_a)
    A=tf.constant(numpy_a,shape=(3,2,4))
    print(A)
#Getting information about tensors
#Shape tf.shape()
#Rank tf.ndim()
#Axis or dimesion tensor[0],tensor[:,1]
#Size tf.size()
#create a 4 dimensions tensor
def properties_of_tensors():
    rank_4=tf.zeros(shape=(2,3,4,5))
    print(rank_4)
    print("the shape is",rank_4.shape)
    print("Number of dimensions",rank_4.ndim)
    print("Total size of elements:",tf.size(rank_4).numpy())# If we dont want to see the thong with tf we conver it on a numpy array
    print("The datatype is:",rank_4.dtype)
    print("elements for the last axis",rank_4.shape[-1])
#Indexing tensors
#Tensors can be indexes just like python list
def reshape_and_playing_with_tensors():
    some_list=[1,2,3,4,5]
    print(some_list[:2])
    tens=tf.zeros(shape=(2,3,4,5))
    print(tens[:2,:2,:2,:2])
    print("Separator")
    print(tens[:2,:3,:,:4])
    rank_2_tensor=tf.constant([[10,7],
                               [4,1],
                               [6,4]])
    print(rank_2_tensor.shape)
    print(rank_2_tensor.ndim)
    print(rank_2_tensor[:,:-1])
    rank_3_tensor=rank_2_tensor[...,tf.newaxis]#... means every axis existemt
    print(rank_3_tensor)
    print(tf.expand_dims(rank_2_tensor,axis=0))
#You can use basic operations on tensors
def operations_with_tensors():
    tensor=tf.constant([[20,7],
                        [8,9]])
    print(tensor+10)
    #So we didn't modified the tensor
    print(tensor)
    print(tensor*10)
    print(tensor-10)
    print(tensor/10)
    #We can also use the tensorflow function
    print(tf.multiply(tensor,10))
#Matrix multiplication
#In ml matrix multiplication is one of the most comon operation
#The "Dot Product" is where we multiply matching members, then sum up:(1, 2, 3) • (7, 9, 11) = 1×7 + 2×9 + 3×11  = 58
#We match the 1st members (1 and 7), multiply them, likewise for the 2nd members (2 and 9) and the 3rd members (3 and 11), and finally sum them up.
def matrix_multiplication_part_1():
    a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
    print(a)
    b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
    print(b)
    print(tf.matmul(a,b))
    c=tf.constant([1,2,3,4,5,6],shape=(3,2))
    d=tf.constant([7,8,9,10,11,12],shape=(2 ,3))
    #As u can see we cannot multiply the same type of matrix shape
    #there are our tensor need to fullfill
    #The inner dimensions must match
    #The resulting matrix has the shape of the inner dimension
    print(c)
    print(d)
    #tf.reshape(c,shape=(3,2))
    print(tf.tensordot(c,d,axes=1))
    #print(c)
def change_dt_of_tensors():
    B=tf.constant([1.4,2.3])
    print(B.dtype)
    B=tf.cast(B,dtype=tf.float16)
    print(B.dtype)
#Agreatting tensors
#Agreatting means condensing them from multiple values down to a smaller value
#GEtting the absolute values
def agreation():
    D=tf.constant([-7,10])
    print(tf.abs(D))
    #Forms of agreattion
    #GEt the min
    #GEt the max
    #GEt the mean of a tensor
    #GEt the sum
    print("the minumum value of a tensor is")
    print(tf.reduce_min(D))
    print("The maximum value of a tensor")
    print(tf.reduce_max(D))
    print("THe sum is:")
    print(tf.reduce_sum(D))
    print("The mean of the D tensor is")#The mean inseamna sa faci media
    print(tf.reduce_mean(D))
def get_the_ARG_OF_A_TENSOR():
    tf.random.set_seed(42)
    F=tf.random.uniform(shape=[50])
    print(F)
    print(tf.argmax(F))#Index of max
    print(F[tf.argmax(F)])#The maximum value
    print(tf.argmin(F))#Index of min
    print(F[tf.argmin(F)])#The minimum value
#Squeezing a tensor(finding all single dimensions
def Squeezing_a_tensor():
    tf.random.set_seed(42)
    G=tf.constant(tf.random.uniform(shape=[50]),shape=(1,1,1,1,50))
    print(G)
    G_squeezed=tf.squeeze(G)
    #Removes dimensions of size 1 from the shape of a tensor.
    print(G_squeezed)
#One hot encoding
#Create a list of indices
def hot_encoding():
    some_list=[0,1,2,3]
    print(tf.one_hot(some_list,depth=4))
    print("Separate")
    print(tf.one_hot(some_list,depth=4,on_value="a",off_value="Ib"))
#Squaring,log square root
def advanced_functions():
    H=tf.range(1,10)
    print(H)
    print("Square it ")
    print(tf.square(H))
    print("Square root it")
    k=tf.constant([4,16])
    print(tf.sqrt(tf.cast(k,dtype=tf.float16)))
    #Now lets use the log
    print("we will use log")
    print(tf.math.log(tf.cast(k,dtype=tf.float16)))
#Tensors and numpy
#Create a tensor firectly from a numpy array
def tensors_and_numpy():
    J=tf.constant(np.array([10,3,2]))
    print(J)
    print(np.array(J))
    print(type(np.array(J)))
    J.numpy()
    print(type(J.numpy()))
    print(J.numpy()[0])
    numpy_A=tf.constant(np.array([3.,7.,10.]))
    tensor_J=tf.constant([3.,7.,10.])
    print("The datatype of the tensor")
    print(tensor_J.dtype)
    print("The datatype of array from numpy")
    print(numpy_A.dtype)
print(tf.config.list_physical_devices())#You can see that wer do not have on GPU
