import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
training_epochs = 1000
display_step = 50

# Training Data
train_X = np.asarray([3.3,4.4,5.5,6.79,6.93,4.178, 8.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.5,1.704,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,3.94,1.3])
n_samples = train_X.shape[0]

#tf graph
X = tf.placeholder("float")
Y = tf.placeholder("float")

#model weights
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

#linear model
pred = tf.add(tf.multiply(X, W), b)

#mean squared error (cost function)
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

#gradient descent (optimizer)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#init the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            #it only runs the computation subgraph necessary for every operation you run 
            sess.run(optimizer, feed_dict={X: x, Y: y})

            if (epoch+1) % display_step == 0:
                ev = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(ev), "W=", sess.run(W), "b=", sess.run(b))
    print("Optimization done!")
    
    #display graphic
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

