from Classifier import *
from optimal_boundary import optimal_boundary

import numpy as np
import matplotlib.pyplot as plt


def main():

    # -----------------------------------------------------------------
    # Set up Classifier + training parameters
    # -----------------------------------------

    x_size = 2
    y_size = 2
    model_directory = "./model_Classifier/2d/"

    my_classifier = Classifier_simple(x_size,y_size,model_directory)

    # -----------------------------------------------------------------
    # Train + Test + Boundary Loop (loops through noise values)
    # -----------------------------------------
    boundaries = []
    num_figures = 2
    num_boundaries = 1

    # Get Optimal Boundary + pts
    optimal_boundary_pts, _ = optimal_boundary(plot_flag=False)

    factor1, factor2 = utils.subplot_values(num_figures)
    fig, axs = plt.subplots(factor1, factor2)
    axs = axs.ravel()


    for i, noise in enumerate(np.linspace(0.05,0.2,num_figures)):
        axs[i].set_aspect(1.0)
        for random_state in range(0, num_boundaries):

            # Train Classifier
            sess = tf.InteractiveSession()
            learning_rate = 0.001
            num_epochs = 500
            batch_size = 100
            num_samples = 500

            my_classifier.set_dataset(batch_size,num_samples,noise,random_state)
            my_classifier.train(sess, num_epochs, learning_rate)

            # Test Classifier
            my_classifier.eval_model(sess)

            # Classifier Decision Boundary
            decision_boundary = my_classifier.decision_boundary(sess, axs[i], plot_flag=False, contourf_flag=False, supress_flag=False)

            x,y = decision_boundary.T
            # axs[i].plot(x,y)
            axs[i].set_title("noise = " + str(round(noise,2)))
            boundaries.append(decision_boundary)

            # Plot Optimal Boundary
            x,y = optimal_boundary_pts.T
            axs[i].plot(x,y)

    plt.tight_layout()
    plt.show()
    #----------------------------------------------------------------
    # End loop
    # ----------------------------------------------------------------



if __name__ == '__main__': main()
