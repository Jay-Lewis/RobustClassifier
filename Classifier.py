import tensorflow as tf
import utils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

########################################################################################3
####################            Simple Classifier            ###########################3
########################################################################################3

class Classifier_simple(object):

    def __init__(self,x_size,y_size,model_directory):
        self.x_size = x_size
        self.y_size = y_size
        self.model_directory = model_directory

        self.x = tf.placeholder(tf.float32, shape=[None, x_size])
        self.y = tf.placeholder(tf.int32, shape=[None, y_size])
        self.logits, self.class_vars = self.build_classifier(x=self.x, reuse=False)

        self.classifier_saver = tf.train.Saver(var_list=self.class_vars, max_to_keep=1)
        self.set_load_func()

        self.X_adv = []
        self.y_adv = []

    def build_classifier(self, x, reuse=False):
        with tf.variable_scope("Classifier", reuse=reuse):
            dense1 = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu)
            dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.relu)
            dense3 = tf.layers.dense(inputs=dense2, units=30, activation=tf.nn.relu)
            dense4 = tf.layers.dense(inputs=dense3, units=20, activation=tf.nn.relu)
            dense_out = tf.layers.dense(inputs=dense4, units=10, activation=tf.nn.relu)
            outputs = tf.layers.dense(inputs=dense_out, units=2)
            classifier_vars = [var for var in tf.trainable_variables() if 'Classifier' in var.name]
        return outputs, classifier_vars

    def set_load_func(self):
        self.load_func = utils.moon_load

    def set_dataset(self,batch_size,num_samples,noise,random_state):
        self.train_epoch, self.data, self.test_epoch = utils.load_dataset(batch_size, self.load_func,False, num_samples, noise, random_state)
        self.X_train, self.y_train, self.X_test, self.y_test = self.data
        self.save_training_pts()

    def save_training_pts(self):
        self.original_X_train = self.X_train
        self.original_y_train = self.y_train

    def set_adv_dataset(self, batch_size, X_adv, y_adv):

        # Save adversarial points
        if(len(self.X_adv) == 0 or len(self.y_adv) == 0):
            self.X_adv = X_adv
            self.y_adv = y_adv
        else:
            self.X_adv = np.vstack([self.X_adv, X_adv])
            self.y_adv = np.hstack([self.y_adv, y_adv])

        # Append adversarial points to training data
        self.X_train = np.vstack([self.X_train, X_adv])
        self.y_train = np.hstack([self.y_train, y_adv])
        self.data = (self.X_train, self.y_train, self.X_test, self.y_test)

        # Create Generator functions (for training)
        self.train_epoch = utils.adv_load_dataset(batch_size, self.data)

    def train_model(self, sess, x, y, train_epoch, train_op, num_epochs):
        train_gen = utils.batch_gen(train_epoch, True, y.shape[1], num_epochs)
        for x_train, y_train in train_gen:
            sess.run(train_op, feed_dict={x: x_train, y: y_train})

    def save_model(self, sess, saver, checkpoint_dir):
        saver.save(sess, checkpoint_dir + 'trained_model')
        saver.export_meta_graph(checkpoint_dir + 'trained_model_graph' + '.meta')

    def get_train_op(self, logits, y, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        loss_op = tf.losses.softmax_cross_entropy(y, logits=logits)
        return optimizer.minimize(loss_op)

    def get_accuracy_op(self, logits, y):
        correct_pred = tf.equal(tf.argmax(logits, 1),
                                tf.argmax(y, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def train(self, sess, num_epochs, learning_rate, retrain = False):

        train_op = self.get_train_op(self.logits, self.y, learning_rate)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        if(not retrain):
            init = tf.global_variables_initializer()
            sess.run(init)

        print('training model')
        self.train_model(sess, self.x, self.y, self.train_epoch, train_op, num_epochs)
        print('saving model')
        self.save_model(sess,self.classifier_saver, self.model_directory)

    def eval_model(self, sess):
        accuracy_op = self.get_accuracy_op(self.logits, self.y)
        batch_gen = utils.batch_gen(self.test_epoch, True, self.y.shape[1], num_iter=1)
        iteration = 0
        normal_avr = 0
        for points, labels in batch_gen:
            iteration += 1
            avr = sess.run(accuracy_op, feed_dict={self.x: points, self.y: labels})
            normal_avr += avr
        normal_accuracy = normal_avr / iteration
        print("Normal Accuracy:", normal_accuracy)

        return normal_accuracy

    def decision_boundary(self,sess, ax=None, plot_flag=False,contourf_flag=True,supress_flag = False):
        # Check for figure
        if(ax == None):
            ax = plt.axes()

        # Color Maps (red to blue)
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        # Plot the decision boundary. For that, we will assign a prediction to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].

        h = .2  # step size in the mesh
        x_min, x_max = self.X_train[:, 0].min() - .5, self.X_train[:, 0].max() + .5
        y_min, y_max = self.X_train[:, 1].min() - .5, self.X_train[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        points = np.transpose([xx.ravel(), yy.ravel()])

        Z = sess.run(self.logits, feed_dict={self.x: points})
        Z = utils.softmax(Z)
        Z = np.asarray([pred[1] for pred in Z])

        # Put the result into a contour plot
        shape = np.shape(xx)
        Z = Z.reshape(shape)
        num_contours = 1
        if(contourf_flag):
            contour = ax.contourf(xx, yy, Z, num_contours, cmap=cm, alpha=.8)
        else:
            contour = ax.contour(xx, yy, Z, num_contours, cmap=cm, alpha=.8)

        if(plot_flag):
            # Plot training points
            ax.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap=cm_bright,
                       edgecolors='k')
            # and testing points
            ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap=cm_bright, alpha=0.6,
                       edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())

            plt.show()

        if(supress_flag):
            plt.cla()

        paths = contour.collections[0].get_paths()[0]
        decision_boundary = paths.vertices

        return decision_boundary