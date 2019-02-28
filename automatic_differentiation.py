# -*- coding: utf8 -*-
# 自动微分说明
import tensorflow as tf

print(tf.__version__)
# enable eager execution
tf.enable_eager_execution()

# 计算梯度实例 变量的
x = tf.ones((2, 2))
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
print()
print("x: \n{}\n".format(x))
print("z: \n{}\n".format(z))

dz_dx = t.gradient(z, x)
print("gradient: \n{}\n".format(dz_dx))
for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0


x = tf.constant(3.0)
with tf.GradientTape() as t:
    t.watch(x)
    y = x * x
    z = y * y

print("x: \n{}\n".format(x))
print("z: \n{}\n".format(z))
dz_dx = t.gradient(z, x)
print("gradient: \n{}\n".format(dz_dx))

def f(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
    return output

def grad(x, y):
    with tf.GradientTape() as t:
        t.watch(x)
        out = f(x, y)
    return t.gradient(out, x)

x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0
assert grad(x, 5).numpy() == 12.0
assert grad(x, 4).numpy() == 4.0

# 二阶微分测试
x = tf.Variable(1.0)

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    dy_dx = t2.gradient(y, x)
d2y_d2x = t.gradient(dy_dx, x)

print("x: \n{}\n".format(x))
print("y: \n{}\n".format(y))
print("dy_dx: \n{}\n".format(dy_dx))
print("d2y_d2x: \n{}\n".format(d2y_d2x))

assert dy_dx == 3
assert d2y_d2x == 6
