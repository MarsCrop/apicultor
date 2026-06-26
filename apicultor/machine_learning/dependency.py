import numpy as np
from ..gradients.subproblem import *

def BTC(y, yhat):
    """
    Backward trust compatibility measures target dependency in points after updating their targets                                                           
    :param y: targets                                                                                         
    :param yhat: predictions        
    :returns:                                                                                                         
      - btc score: target-wise trust compatibility         
    """
    hat_btcs = []
    istarget_prev = 1e-4
    istarget_dep = 1e-4
    if y[0].size > 1:
        if attention == False:
            y = sigmoid(y)
            yhat = sigmoid(yhat)
            y = np.argmax(np.array(y),axis=1)
            yhat = np.argmax(np.array(yhat),axis=1)       	
        else:
            y = sigmoid(y)
            yhat = sigmoid(yhat)
            y = np.argmax(np.array(y),axis=0)
            yhat = np.argmax(np.array(yhat),axis=0)    
    for i in np.unique(y):
        for j in range(len(y)):
            if y[j] == i:
                if yhat[j] == i:
                    istarget_prev += 1
                    istarget_dep += 1
            elif yhat[j] == i:
                istarget_dep += 1
        hat_btcs.append(1-(istarget_prev / istarget_dep))
        istarget_prev = 1e-4
        istarget_dep = 1e-4
    return hat_btcs

def BEC(y, yhat, keep_index=None, return_array=None, attention=False):
    hat_becs = []
    istarget_prev = 1e-4
    istarget_con = 1e-4
    hat_cons = []
    
    # Si son 2D, aplanar o usar solo la primera columna para clasificación
    if len(y.shape) > 1 and y.shape[1] > 1:
        # Para regresión multidimensional, usar la media de cada fila como valor representativo
        y_flat = np.mean(y, axis=1) if attention else y.flatten()
        yhat_flat = np.mean(yhat, axis=1) if attention else yhat.flatten()
    else:
        y_flat = y.flatten() if y.ndim > 1 else y
        yhat_flat = yhat.flatten() if yhat.ndim > 1 else yhat
    
    # Convertir a valores discretos para clasificación
    y_binned = np.digitize(y_flat, bins=np.percentile(y_flat, [33, 66]))
    yhat_binned = np.digitize(yhat_flat, bins=np.percentile(yhat_flat, [33, 66]))
    
    for i in np.unique(y_binned):
        if keep_index is None:
            iscon = []
        elif keep_index is True:
            iscon = []
        else:
            iscon = [0 for _ in range(len(y_binned))]
            try:
                iscon[keep_index[i]] = 1
            except Exception:
                pass
        
        for j in range(len(y_binned)):
            if y_binned[j] != i:
                if yhat_binned[j] != i:
                    istarget_prev += 1
                    istarget_con += 1
                    if keep_index is None or keep_index is True:
                        iscon.append(1)
                    else:
                        iscon[j] = 1
            elif yhat_binned[j] != i:
                istarget_con += 1
                if keep_index is None or keep_index is True:
                    iscon.append(1)
                else:
                    iscon[j] = 1
            else:
                if keep_index is None or keep_index is True:
                    iscon.append(0)
                else:
                    iscon[j] = 0
        
        hat_becs.append(1 - (istarget_prev / (istarget_con + 1e-9)))
        iscon = np.array(iscon)
        hat_cons.append(np.where(iscon == 1)[0])
        istarget_prev = 1e-4
        istarget_con = 1e-4
    
    return hat_becs, hat_cons

