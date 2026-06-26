import numpy as np
import logging
import warnings
from ..gradients.subproblem import *

warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def p_rule(y_predicted, y, theta, x, proba, thresh=1e-4):
    try:
        # Convertir theta a array si es necesario
        if theta is None:
            return 1.0
        
        if hasattr(theta, 'values'):
            theta = np.array(list(theta.values()))
        elif isinstance(theta, (defaultdict, dict)):
            theta = np.array(list(theta.values()))
        elif not isinstance(theta, np.ndarray):
            theta = np.array(theta)
        
        # Asegurar que x tenga la forma correcta
        if hasattr(x, 'shape') and x.ndim > 1:
            x_flat = x.flatten()
        else:
            x_flat = np.array(x).flatten()
        
        # Aplanar y y y_predicted
        y_flat = y.flatten() if hasattr(y, 'flatten') else np.array(y).flatten()
        y_pred_flat = y_predicted.flatten() if hasattr(y_predicted, 'flatten') else np.array(y_predicted).flatten()
        
        # Calcular diferencia
        min_len = min(len(y_flat), len(y_pred_flat))
        diff = y_flat[:min_len] - y_pred_flat[:min_len]
        y_size = len(y_flat)
        
        # Aplanar theta para multiplicación
        theta_flat = theta.flatten() if theta.ndim > 1 else theta
        
        # Alinear dimensiones
        max_len = max(len(diff), len(theta_flat))
        diff_padded = np.pad(diff, (0, max_len - len(diff)), 'constant')
        theta_padded = np.pad(theta_flat, (0, max_len - len(theta_flat)), 'constant')
        
        significance = 1 / (y_size * np.sum(diff_padded * theta_padded) + 1e-9)
        
    except Exception as e:
        logger.debug(f"p_rule calculation error: {e}")
        significance = 0.0
    
    thresh_val = 2
    if -thresh_val < significance < thresh_val:
        if hasattr(proba, 'size') and proba.size > 0:
            proba_clipped = np.clip(proba, 1e-10, 1 - 1e-10)
            if proba.ndim > 1:
                entropy = -np.sum(np.log(proba_clipped), axis=1)
            else:
                entropy = -np.sum(np.log(proba_clipped))
            return float(np.min(entropy))
        else:
            return 1.0
    else:
        return False

def unprotection_score(old_loss, fx, y):
    """
    A measure of conditional procedure accuracy equality (disparate mistreatment) between binary instances
    """
    # Asegurar que fx sea 2D si y es 2D
    if y.ndim > 1 and fx.ndim == 1:
        # Expandir fx a la misma forma que y
        fx = np.tile(fx.reshape(-1, 1), (1, y.shape[1]))
    
    # Asegurar que tengan la misma forma
    min_rows = min(y.shape[0], fx.shape[0])
    if y.ndim == 2 and fx.ndim == 2:
        min_cols = min(y.shape[1], fx.shape[1])
        y_cut = y[:min_rows, :min_cols]
        fx_cut = fx[:min_rows, :min_cols]
    else:
        y_cut = y.flatten()[:min_rows * (y.shape[1] if y.ndim > 1 else 1)]
        fx_cut = fx.flatten()[:min_rows * (fx.shape[1] if fx.ndim > 1 else 1)]
    
    try:
        new_loss = np.mean((y_cut - np.sign(fx_cut)) ** 2)
    except Exception as e:
        new_loss = np.mean(np.square(y_cut - np.sign(fx_cut)))
    
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

