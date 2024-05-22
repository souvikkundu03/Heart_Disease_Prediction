import numpy as np
import tensorflow as tf
import time

array_size = 50000
array_np1 = np.random.random((array_size, array_size))
array_np2 = np.random.random((array_size, array_size))
array_tf1 = tf.random.normal((array_size, array_size))
array_tf2 = tf.random.normal((array_size, array_size))

start_time = time.time()
result_np = np.dot(array_np1, array_np2)
numpy_time = time.time() - start_time


start_time = time.time()
result_tf = tf.matmul(array_tf1, array_tf2)
tensorflow_time = time.time() - start_time

print(f"NumPy array multiplication time: {numpy_time:.5f} seconds")
print(f"TensorFlow array multiplication time: {tensorflow_time:.5f} seconds")