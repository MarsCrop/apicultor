import numpy as np
import logging
import warnings
from ..gradients.subproblem import *

warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def p_rule(y_predicted, y, theta, x, proba, thresh = 1e-4):
    """
    Unlike group fairness, satisfying p-rule should be enough
    to know if the model accomplishes statistical parity.
    If boundary error is independent of the product between dataset and regression,
    then statistical parity is satisfied and all data must be treated equally 
    """
    try:
        significance = 1/(y.size*np.sum((y-y_predicted) * np.multiply(theta, x.T)))
    except Exception as e:
        try:
            significance = 1/(y.size*np.sum((y-y_predicted) * np.multiply(theta.T, x)))
        except Exception as e:
            x_count_rows = [len(np.array(arr)) for arr in x]
            theta_count_rows = [len(np.array(arr)) for arr in theta]        
            theta = satisfy_pad_requirements(theta_count_rows, theta_count_rows, theta, is_vector=True)
            x = satisfy_pad_requirements(x_count_rows, x_count_rows, x, is_vector=True)
            weighted_samples = np.multiply(theta, x)
            try:
                significance = 1/(y.size*np.sum((y-y_predicted) * weighted_samples))
            except Exception as e:
                try:
                    significance = 1/(y.size*np.sum(np.multiply((y-y_predicted).T, weighted_samples)))
                except Exception as e:  # Log every 10 seconds
                    significance = 1/(y.size*np.sum((y-y_predicted).dot(weighted_samples.T)))

    thresh = 2
    #print('Significance is', significance, 'WITH THRESHOLD OF', thresh)
    if -(thresh) < significance < thresh:
    #if -.005 < significance < .005:
        # tradeoff is inversely proportional to probability of being assigned more than 1 label
        #print('Probability of being assigned more than 1 class:', min(-np.sum(np.log(proba), axis=1)))
        return min(-np.sum(np.log(proba), axis=1))
    else:
        return False


def unprotection_score(old_loss, fx, y):
    """
    A measure of conditional procedure accuracy equality (disparate mistreatment) between binary instances
    """
    try:
        new_loss = np.mean((y-np.sign(fx))**2)
    except Exception as e:
        new_loss = np.mean((np.square(y-np.sign(fx))))
    unprotected = ((1.0 + 1) * old_loss) - new_loss
    return unprotected

# both measure fairness for regression problems


def ind_fairness(fxi, fxj, y):
    """A measure of conditional parity, which is the local probability of being assigned to either targets
    of a binary problem
    """
    return 1/(len(fxi)**2) * (np.sum(y.size*((fxi-fxj)**2)))


def group_fairness(fairness):
    """Given a conditional parity between all binary problems, return global probability of being assigned to all targets
    """
    return fairness**2

