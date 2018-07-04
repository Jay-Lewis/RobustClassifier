from Classifier import *
from attacks import *
import utils
from optimal_boundary import optimal_boundary

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler



# -----------------------------------------------------------------
# Set up Classifier + training parameters
# -----------------------------------------

x_size = 2
y_size = 2
model_directory = "./model_Classifier/2d/"
# model_directory = "/home/justin/Github/RobustClassifier/model_Classifier/2d/"

my_classifier = Classifier_simple(x_size,y_size,model_directory)

# -----------------------------------------------------------------
# Initial Train + Test + Boundary
# -----------------------------------------

# Train Classifier
sess = tf.InteractiveSession()
learning_rate = 0.001
num_epochs = 500
num_samples = 300
batch_size = min(100, int(num_samples/4.0))
noise = 0.05
random_state = 0

my_classifier.set_dataset(batch_size, num_samples, noise, random_state)
my_classifier.train(sess, num_epochs, learning_rate)

# Test Classifier
my_classifier.eval_model(sess)

# Classifier Decision Boundary
initial_decision_boundary = my_classifier.decision_boundary(sess=sess, plot_flag=False, contourf_flag=False, supress_flag= True)

# -----------------------------------------------------------------
# Get Optimal Boundary
# -----------------------------------------

# Get Optimal Boundary + pts
optimal_decision_boundary, _ = optimal_boundary(plot_flag=False, power=2, metric='minkowski')

# -----------------------------------------------------------------
# Adversarial Training Loop
# -----------------------------------------
num_iter = 8
boundaries = []

for i in range(0, num_iter):
    print('---------------------------')
    print('PGD Iteration: ' + str(i+1))
    print('---------------------------')
    # -----------------------------------------------------------------
    # Find Adversarial Examples
    # -----------------------------------------

    # PGD attack
    epsilon = 0.1
    pgd_iter = 12
    a = 0.2
    random_start = True

    loss = tf.losses.softmax_cross_entropy(my_classifier.y, my_classifier.logits)
    attack = LinfPGDAttack(loss, my_classifier.x, epsilon, pgd_iter,a,random_start)
    y_labels = utils.make_one_hot(my_classifier.original_y_train)

    X_adv = attack.perturb(my_classifier.original_X_train, y_labels, my_classifier.x, my_classifier.y, sess)
    y_Adv = my_classifier.original_y_train


    # -----------------------------------------------------------------
    # Adversarial Training
    # -----------------------------------------
    my_classifier.set_adv_dataset(batch_size, X_adv, y_Adv)
    my_classifier.train(sess, num_epochs, learning_rate)

    # Test Classifier
    my_classifier.eval_model(sess)

    # Classifier Decision Boundary
    decision_boundary = my_classifier.decision_boundary(sess, plot_flag=False,
                                                        contourf_flag=False, supress_flag= True)
    boundaries.append(decision_boundary)

    if(i+1 == num_iter):

        # # -----------------------------------------------------------------
        # # Plot Adversarial Examples + Boundaries
        # # -----------------------------------------
        ax = plt.axes()
        cm = ListedColormap(['#FF0000', '#0000FF'])

        # Plot normal points
        ax.scatter(my_classifier.original_X_train[:, 0], my_classifier.original_X_train[:, 1],
                   c=my_classifier.original_y_train, cmap=cm, edgecolors='k')

        # Plot adversarial points
        cm = ListedColormap(['#ffff00', '#00ff00'])
        ax.scatter(my_classifier.X_adv[:, 0], my_classifier.X_adv[:, 1], c=my_classifier.y_adv, cmap=cm,
                   edgecolors='k')

        # Plot initial boundary
        x, y = initial_decision_boundary.T
        ax.plot(x, y, c='k')

        # Plot progression of Adv. Tr. boundaries
        num_colors = num_iter
        endpoints = ['#4c4c4c', '#cecece']
        cm = LinearSegmentedColormap.from_list('my_cmap', endpoints, N=num_colors)
        colors = [cm(1. * i / num_colors) for i in range(num_colors)]
        ax.set_prop_cycle(cycler('color', colors))

        for j, decision_boundary in enumerate(boundaries):
            x, y = decision_boundary.T
            ax.plot(x, y)

        # Plot optimal boundary
        x, y = optimal_decision_boundary.T
        ax.plot(x, y, c='m')

        plt.tight_layout()
        plt.show()

#----------------------------------------------------------------
# End loop
# ----------------------------------------------------------------



