from Classifier import *
from optimal_boundary import optimal_boundary

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -----------------------------------------------------------------
# Set up Classifier + training parameters
# -----------------------------------------

x_size = 2
y_size = 2
model_directory = "./model_Classifier/2d/"

my_classifier = Classifier_simple(x_size,y_size,model_directory)
sess = tf.InteractiveSession()

# -----------------------------------------------------------------
# Train + Test + Boundary Loop (loops through noise values)
# -----------------------------------------
boundaries = []
num_figures = 1
num_boundaries = 1

# Get Optimal Boundary + pts
optimal_boundary_pts, _ = optimal_boundary(plot_flag=False, power=2, metric='minkowski')

if(num_figures > 1):
    factor1, factor2 = utils.subplot_values(num_figures)
    fig, axs = plt.subplots(factor1, factor2)
    axs = axs.ravel()
else:
    axs = [plt.axes()]


for i, noise in enumerate(np.linspace(0.05,0.2,num_figures)):
    noise = 0.2
    axs[i].set_aspect(1.0)
    for random_state in range(0, num_boundaries):

        # Train Classifier
        learning_rate = 0.001
        num_epochs = 300
        batch_size = 100
        num_samples = 1000

        my_classifier.set_dataset(batch_size,num_samples,noise,random_state)
        my_classifier.train(sess, num_epochs, learning_rate)

        # Test Classifier
        my_classifier.eval_model(sess)

        # Classifier Decision Boundary
        decision_boundary = my_classifier.decision_boundary(sess, axs[i], plot_flag=False, contourf_flag=False, supress_flag=False)
        x, y = decision_boundary.T
        # axs[i].plot(x,y)
        axs[i].set_title("noise = " + str(round(noise,2)))
        boundaries.append(decision_boundary)

        # Plot Optimal Boundary
        x, y = optimal_boundary_pts.T
        axs[i].plot(x, y)

        # Plot Training Points
        # colorpoints = [(0.0, '#ff0000'),(0.1, '800080'),(1.0, '#0000ff')]
        colorpoints = [(0,    '#ffff00'),
                       (0.005, '#002266'),
                       (0.995, '#002266'),
                       (1,    '#002266')]
        num_colors = 256
        cmap = LinearSegmentedColormap.from_list('my_cmap', colorpoints, N=num_colors)

        logits = sess.run(my_classifier.logits, feed_dict={my_classifier.x: my_classifier.X_train})
        preds = np.asarray([pred[1] for pred in utils.softmax(logits)])
        axs[i].scatter(my_classifier.X_train[:, 0], my_classifier.X_train[:, 1],
                       c=preds, cmap=cmap, edgecolors='k')
        # axs[i].scatter(my_classifier.X_train[:, 0], my_classifier.X_train[:, 1],
        #                c=my_classifier.y_train, cmap=cmap, edgecolors='k')

plt.tight_layout()
plt.show()
#----------------------------------------------------------------
# End loop
# ----------------------------------------------------------------




