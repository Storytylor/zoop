import tensorflow as tf

a = tf.constant([[1,2],[3,4]])
b = tf.constant([[5,6],[7,8]])

print("Addition:\n", tf.add(a,b))
print("Subtraction:\n", tf.subtract(a,b))
print("Multiplication:\n", tf.multiply(a,b))
print("Matrix Multiplication:\n", tf.matmul(a,b))
print("Reshape:\n", tf.reshape(a,[4]))

x = tf.constant(10)
y = tf.constant(20)
print("Eager Execution:", x+y)

@tf.function
def calculate(x,y):
    return x*y+10

print("Computation Graph:", calculate(5,3).numpy())