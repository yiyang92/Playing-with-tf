import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# number of inputs
N_INPUTS = 500
# number of training steps
N_TR     = 7000
# learning rate
learning_rate = 0.001

# our function
xt = np.array(np.linspace(0.0,1.0,N_INPUTS)).astype(np.float32)
yt = np.sin(2*np.pi*xt) + np.random.normal(0.0,0.3,N_INPUTS)

# fig, ax = plt.subplots(1,1)
plt.scatter(xt, yt)
# fig.show()
# plt.draw()

# create placeholders for computational graph
X = tf.placeholder(tf.float32, [1, N_INPUTS])
Y = tf.placeholder(tf.float32, [N_INPUTS])

# create model
# power of polinomial function
POW = 5
# create base weights and biases
w = tf.Variable(tf.random_normal([POW, 1]), tf.float32)
b = tf.Variable(tf.random_normal([1]))
# hypothesis function
Yhat = tf.Variable(0,dtype=tf.float32)
for p in range(1, POW):
    Yhat = tf.add(Yhat, tf.mul(tf.pow(X, p), w[p, 0]))
# add bias
Yhat = tf.add(Yhat, b)
# cost function
yerror = tf.sub(Yhat, Y)
loss = tf.nn.l2_loss(yerror)
# implement gradient descend

update_weights = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# launching graph
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(1, N_TR):
        sess.run(update_weights, feed_dict={X:np.reshape(xt,(1, N_INPUTS)), Y:np.transpose(yt)})
        print("Error is: {} Weight is:{} bias is:{}\n".format(loss.eval(feed_dict={X:np.reshape(xt,(1, N_INPUTS)), Y:np.transpose(yt)}),w.eval(), b.eval()))
    print(Yhat.eval(feed_dict={X:np.reshape(xt, (1, N_INPUTS))}))
    print(np.reshape(xt, (1, N_INPUTS)))
    plt.plot(np.reshape(xt, (1, N_INPUTS))[0],
             Yhat.eval(feed_dict={X: np.reshape(xt, (1, N_INPUTS))})[0], 'g')
    plt.show()


