from Classifier import *
from attacks import *
import utils
from optimal_boundary import optimal_boundary

import numpy as np
import matplotlib.pyplot as plt



# -----------------------------------------------------------------
# Set up Classifier + training parameters
# -----------------------------------------

x_size = 2
y_size = 2
# model_directory = "./model_Classifier/2d/"
model_directory = "/home/justin/Github/RobustClassifier/model_Classifier/2d/"

my_classifier = Classifier_simple(x_size,y_size,model_directory)

# -----------------------------------------------------------------
# Initial Train + Test + Boundary
# -----------------------------------------

# Train Classifier
sess = tf.InteractiveSession()
learning_rate = 0.001
num_epochs = 500
batch_size = 100
num_samples = 500
noise = 0.05
random_state = 0

my_classifier.set_dataset(batch_size,num_samples,noise,random_state)
my_classifier.train(sess, num_epochs, learning_rate)

# Test Classifier
my_classifier.eval_model(sess)

# Classifier Decision Boundary
ax = plt.axes()
decision_boundary, _ = my_classifier.decision_boundary(sess, ax, plot_flag=False, contourf_flag=False, supress_flag= False)


# -----------------------------------------------------------------
# Find Adversarial Examples
# -----------------------------------------

# PGD attack
epsilon = 0.1
pgd_iter = 40
a = 0.01
random_start = True

loss = tf.losses.softmax_cross_entropy(my_classifier.y, my_classifier.logits)
attack = LinfPGDAttack(loss, my_classifier.x,epsilon,pgd_iter,a,random_start)
y_labels = utils.make_one_hot(my_classifier.y_train)


X_adv = attack.perturb(my_classifier.X_train, y_labels, my_classifier.x, my_classifier.y, sess)
y_Adv = my_classifier.y  #NOTE: would need to change this and X_adv to create loop (adv. examp. shouldn't be used to find new adv. examp.)


# -----------------------------------------------------------------
# Plot Adversarial Examples + Boundary
# -----------------------------------------

cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# # Plot normal points
# ax = plt.axes()
# ax.scatter(my_classifier.X_train[:,0], my_classifier.X_train[:,1], c=my_classifier.y_train, cmap=cm_bright,
#            edgecolors='k')

# Plot adversarial points
cm_bright = ListedColormap(['#ffff00', '#00ff00'])
ax.scatter(X_adv[:,0], X_adv[:,1], c=my_classifier.y_train, cmap=cm_bright,
           edgecolors='k')

plt.tight_layout()
plt.show()


# -----------------------------------------------------------------
# Adversarial Training + new Boundary
# -----------------------------------------

my_classifier.set_adv_dataset(batch_size,random_state,X_adv, y_Adv)
my_classifier.train(sess, num_epochs, learning_rate)

# Test Classifier
my_classifier.eval_model(sess)

# Classifier Decision Boundary
decision_boundary, _ = my_classifier.decision_boundary(sess, ax, plot_flag=True, contourf_flag=False, supress_flag= False)



