from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

def acc_score(targs, classes):
    """
    return an accuracy score of how well predictions did
    :param targs: predictive targets
    :param classes: predicted targets
    :returns:                                                                                                         
      - the mean squared error score (higher than 0.5 should be a good indication)
    """
    lw = np.ones(len(targs))
    for idx, m in enumerate(np.bincount(targs)):
        lw[targs == idx] *= (m/float(targs.shape[0]))
    return accuracy_score(targs, classes, sample_weight=lw)


def score(targs, classes):
    """
    return an accuracy score of how well predictions did
    :param targs: predictive targets
    :param classes: predicted targets
    :returns:                                                                                                         
      - the mean squared error score (higher than 0.5 should be a good indication)
    """
    lw = np.ones(len(targs))
    for idx, m in enumerate(np.bincount(targs)):
        lw[targs == idx] *= (m/float(targs.shape[0]))
    return mean_squared_error(targs, classes, sample_weight=lw)
