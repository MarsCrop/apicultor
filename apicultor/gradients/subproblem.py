#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from ..arch.thread import *
import asyncio
import numpy as np
from collections import defaultdict

s = lambda smax, b, B: (1/smax)+(smax-(1/smax))*((b-1)/(B-1))
sigmoid = lambda x: 1 / (1 + np.power(np.e, (-1 * x)))
g = lambda lab, a0, Q: 1 - lab * np.sum(a0 * Q)                 
Q_a = lambda a, Q: np.sum(a) - (np.sum(np.sum(a * Q, axis = 0), axis = 0)/2)

def context_vector(hidden_states, attention_weights):
    if hidden_states is None or attention_weights is None:
        return np.zeros(1)
    
    if not isinstance(hidden_states, np.ndarray):
        hidden_states = np.array(hidden_states)
    if not isinstance(attention_weights, np.ndarray):
        attention_weights = np.array(attention_weights)
    
    if hidden_states.size == 0 or attention_weights.size == 0:
        return np.zeros(max(1, hidden_states.shape[0] if hidden_states.size > 0 else attention_weights.shape[0]))
    
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    attention_weights = np.asarray(attention_weights, dtype=np.float64)
    
    if len(hidden_states.shape) == 0 or len(attention_weights.shape) == 0:
        return np.zeros(1)
    
    try:
        if attention_weights.shape[-1] != hidden_states.shape[0]:
            attention_weights = attention_weights.T
        
        result = np.sum(attention_weights * hidden_states, axis=1)
        return result if result.size > 0 else np.zeros(hidden_states.shape[0])
    except Exception as e:
        logger.debug(f"context_vector error: {e}")
        return np.zeros(hidden_states.shape[0] if hidden_states.size > 0 else 1)

def decode(context, x, y, s_max, s_scalar):
    """
    Decodifica usando atención donde:
    - Q = x (features)
    - K = y (targets/predictions)
    - V = y (targets/predictions) - la misma matriz que K
    - context se usa como información adicional para modificar los scores
    """
    context = np.nan_to_num(context)
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    s_max = np.nan_to_num(s_max)
    s_scalar = np.nan_to_num(s_scalar)
    
    # Calcular atención donde V es y (targets), no context
    # context se usa para modular los scores si es necesario
    attention_result = attention(x, y, y, s_max, s_scalar)
    
    # Si context tiene la forma correcta, podemos usarlo para modular el resultado
    if context.shape[0] == attention_result.shape[0] and context.shape[1] == attention_result.shape[1]:
        attention_result = attention_result + context * 0.1  # pequeña influencia del contexto
    elif context.shape[0] == 1 and context.shape[1] == attention_result.shape[1]:
        # Contexto como bias por feature
        attention_result = attention_result + context
    
    return attention_result

def substraction_fun(a,b):
    return a-b

def mul_fun(a,b):
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)    
    if type(a) == list:           
        a = a[0]
    try:           
        return np.multiply(a, b)
    except Exception:           
        try:
            return np.multiply(a.T, b)
        except Exception:
            try:
                return np.multiply(a, b.T)
            except Exception:
                try:
                    return np.multiply(a.reshape(-1, np.shape(a)[1]), b)
                except Exception:
                    try:
                        return np.multiply(a.reshape(-1, np.shape(a)[1]), b.T)
                    except Exception:
                        try:
                            return np.multiply(a.reshape(-1, np.shape(a)[1]), b.T[0][0])
                        except Exception:
                            return np.multiply(a.reshape(-1, np.shape(a)[1]), b.T[0][0].T)

def gather_layers_outputs(accepted_targets):
    try:
        tensor2tensor_dot = np.array([mul_fun(accepted_targets[i], accepted_targets[i+1].T) for i in range(len(accepted_targets)-1)])
        return np.sum(tensor2tensor_dot, axis=1)
    except Exception as e:
        logger.exception(e)
        return np.sum(tensor2tensor_dot, axis=0)

def forward(attention_function, x, hidden_states, s_max, s_size):
    attention_scores = np.tanh(attention_function)
    attention_weights_proba = sigmoid(attention_scores)
    if len(attention_weights_proba) == 1:           
        attention_weights_proba = attention_weights_proba[0]
    context = context_vector(hidden_states, attention_weights_proba).T
    return decode(context, x, hidden_states, s_max, s_size)[0], attention_scores, attention_weights_proba, context

def attention(Q, K, V, s_max, s_scalar):
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    if K.ndim == 1:
        K = K.reshape(1, -1)
    if V.ndim == 1:
        V = V.reshape(1, -1)
    
    if K.shape[0] == 1:
        num = Q @ K.T
    else:
        num = Q @ K.T
    
    if num.shape[0] != Q.shape[0]:
        num = Q.T @ K
        if num.shape[0] != Q.shape[0]:
            num = Q @ K
    
    denum = np.sqrt(K.shape[0])
    r = num / (denum + 1e-9)
    
    s_val = s(s_max, r, s_scalar)
    sigmoid_val = sigmoid(r)
    attention_weights = s_val * sigmoid_val
    
    result = attention_weights @ V
    
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    
    return result

def satisfy_pad_requirements(count_rows, count_reverted, arrays, is_vector = False):
    if len(set(count_rows)) != 1:
        pad_solve = True
    elif len(set(count_reverted)) != 1:
        pad_solve = True
    else:
        pad_solve = False
    if pad_solve is True:
        if is_vector == False:
            max_rows = max(count_rows)
            max_cols = max(count_reverted)
            expected_size = max(max_rows * len(arrays), max_cols * len(arrays))
            padded_arrays = np.array([np.pad(arr, ((0, max_rows - arr.shape[0]), (0, max_cols - arr.shape[1])), mode='constant', constant_values=0) for arr in arrays])
        else:
            max_rows = max(count_rows)
            padded_arrays = np.zeros((len(arrays), max_rows), dtype=arrays[0].dtype)
        if not (padded_arrays.shape[0] * padded_arrays.shape[1]) >= expected_size:
            return np.vstack(padded_arrays.reshape(padded_arrays.shape[0],padded_arrays.shape[2],padded_arrays.shape[1]))
        else:
            return np.vstack(padded_arrays)
    else:
        return np.vstack(arrays)

async def _process_batch(features, targets, context, total_len, do_forward, batch_idx):
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        
        # Asegurar dimensiones correctas
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
        
        # Asegurar que context tenga la forma correcta
        if context.ndim == 1:
            context = context.reshape(1, -1)
        
        attention_function = decode(context, features, targets, 1.5, total_len)
        
        if attention_function is None or attention_function.size == 0:
            return None
        
        if do_forward:
            attention_scores = np.tanh(attention_function)
            attention_weights_proba = sigmoid(attention_scores)
            
            if attention_weights_proba.ndim == 1:
                attention_weights_proba = attention_weights_proba.reshape(1, -1)
                
            print("PROCESS BATCH FEATURES:", np.shape(features)) 
            print("ATTENTION SCORES:", np.shape(attention_scores))    
            print("ATTENTION WEIGHTS PROBA:", np.shape(attention_weights_proba)) 
            
            # Calcular contexto correctamente - usar la forma que siempre funcione
            # forwarded_context debe tener forma (n_features, n_features) o (n_samples, n_features)
            if attention_weights_proba.shape[0] == features.shape[0]:
                # Mismo número de muestras
                forwarded_context = attention_weights_proba.T @ features
            elif attention_weights_proba.shape[1] == features.shape[0]:
                forwarded_context = attention_weights_proba @ features
            else:
                # Fallback: promediar
                if features.shape[0] > 0:
                    forwarded_context = np.mean(features, axis=0, keepdims=True)
                else:
                    forwarded_context = np.zeros((1, features.shape[1]))
            
            if forwarded_context.ndim == 1:
                forwarded_context = forwarded_context.reshape(1, -1)
            
            attention_y = decode(forwarded_context, features, targets, 1.5, total_len)
            
            return (attention_function, attention_y, attention_scores, attention_weights_proba, forwarded_context)
        else:
            return attention_function

async def continuous_decode_async(features, targets, context, batch_size=5000, do_forward=False):
    n_samples = len(features)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    batch_tasks = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        
        batch_features = features[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]

        print("BATCH FEATURES SHAPE:", batch_features.shape)
        print("BATCH TARGETS SHAPE:", batch_targets.shape)
        
        if len(batch_targets.shape) == 1:
            batch_targets = batch_targets.reshape(-1, 1)
        
        if len(context.shape) == 1:
            batch_context = context[start_idx:end_idx] if len(context) > start_idx else context
            if batch_context.ndim == 1:
                batch_context = batch_context.reshape(-1, 1)
        else:
            batch_context = context[start_idx:end_idx] if len(context) > start_idx else context
        
        batch_tasks.append(_process_batch(batch_features, batch_targets, batch_context, len(features), do_forward, batch_idx))
    
    batch_results = await asyncio.gather(*batch_tasks)
    
    continuous_attention_function = []
    continuous_attention_y = []
    continuous_attention_scores = []
    continuous_attention_function_weights_proba = []
    continuous_attention_context = []
    
    for result in batch_results:
        if result is None:
            continue
        if do_forward:
            func, y, scores, weights, ctx = result
            continuous_attention_function.append(func)
            continuous_attention_y.append(y)
            continuous_attention_scores.append(scores)
            continuous_attention_function_weights_proba.append(weights)
            continuous_attention_context.append(ctx)
        else:
            continuous_attention_function.append(result)
    
    if do_forward:
        if continuous_attention_function:
            continuous_attention_function = np.vstack(continuous_attention_function) if len(continuous_attention_function) > 1 else continuous_attention_function[0]
            continuous_attention_y = np.vstack(continuous_attention_y) if len(continuous_attention_y) > 1 else continuous_attention_y[0]
            continuous_attention_scores = np.vstack(continuous_attention_scores) if len(continuous_attention_scores) > 1 else continuous_attention_scores[0]
            continuous_attention_function_weights_proba = np.vstack(continuous_attention_function_weights_proba) if len(continuous_attention_function_weights_proba) > 1 else continuous_attention_function_weights_proba[0]
            continuous_attention_context = np.vstack(continuous_attention_context) if len(continuous_attention_context) > 1 else continuous_attention_context[0]
            
            return (continuous_attention_function, continuous_attention_y, 
                    continuous_attention_scores, continuous_attention_function_weights_proba, 
                    continuous_attention_context)
        return None
    else:
        return np.vstack(continuous_attention_function) if continuous_attention_function else None

def continuous_decode(features, targets, context, batch_size=5000, do_forward=False):
    try:
        loop = asyncio.get_running_loop()
        return asyncio.create_task(continuous_decode_async(features, targets, context, batch_size, do_forward))
    except RuntimeError:
        return asyncio.run(continuous_decode_async(features, targets, context, batch_size, do_forward))

def continuous_multi_lorax(features, targets, a, batch_size=5000, do_forward=False):
    # Convertir a_array a numpy array
    if isinstance(a, (defaultdict, dict)):
        a_array = np.array(list(a.values()))
    else:
        a_array = np.asarray(a, dtype=np.float64)
    
    # Aplanar targets si es multidimensional
    if targets.ndim > 1:
        targets_flat = targets.flatten()
    else:
        targets_flat = targets
    
    # Obtener dimensiones para logging
    print(f"DEBUG: a_array shape={a_array.shape}, targets_flat shape={targets_flat.shape}, targets original shape={targets.shape}")
    
    # Determinar la mejor forma de multiplicar basado en las dimensiones reales
    if a_array.ndim == 1:
        # a es un vector 1D
        a_len = len(a_array)
        t_len = len(targets_flat)
        
        if a_len == t_len:
            # Misma longitud - producto punto directo
            context = np.dot(a_array, targets_flat)
        elif a_len == targets.shape[1] if targets.ndim > 1 else a_len == targets.shape[0]:
            # a coincide con feature dimension
            # context = a @ targets.T (si a es fila)
            if targets.ndim > 1:
                context = a_array.reshape(1, -1) @ targets.T
            else:
                context = np.dot(a_array, targets_flat)
        else:
            # Fallback: usar correlación simple
            logger.warning(f"Vector dimension mismatch: a_len={a_len}, t_len={t_len}")
            context = np.mean(targets_flat) * np.ones(max(a_len, t_len))[:min(a_len, t_len)]
    
    elif a_array.ndim == 2:
        # a es 2D
        a_rows, a_cols = a_array.shape
        t_len = len(targets_flat)
        
        # Para features con forma (n_samples, n_features), a debería ser (n_features, n_features) o (n_features, 1)
        # El contexto se calcula como a @ targets.T o similar
        
        if targets.ndim == 1:
            # targets es 1D
            if a_cols == t_len:
                # a (n_features, n_features) con targets (n_features,)
                context = a_array @ targets_flat
            elif a_rows == t_len:
                context = a_array.T @ targets_flat
            else:
                # Fallback: reshape
                min_dim = min(a_array.size, t_len)
                a_reshaped = a_array.flatten()[:min_dim].reshape(1, -1)
                t_reshaped = targets_flat[:min_dim].reshape(-1, 1)
                context = a_reshaped @ t_reshaped
        else:
            # targets es 2D (n_samples, n_features)
            t_rows, t_cols = targets.shape
            
            if a_cols == t_cols:
                # a (n_features, n_features) con targets (n_samples, n_features)
                # context = targets @ a.T
                context = targets @ a_array.T
            elif a_rows == t_cols:
                context = a_array @ targets.T
            elif a_cols == t_rows:
                context = a_array.T @ targets
            elif a_rows == t_rows:
                context = a_array @ targets.T
            else:
                # Fallback: usar dot product con flatten
                a_flat = a_array.flatten()
                t_flat = targets.flatten()
                min_len = min(len(a_flat), len(t_flat))
                context = np.dot(a_flat[:min_len], t_flat[:min_len])
    else:
        # a es escalar o forma extraña
        logger.warning(f"Unexpected a_array shape: {a_array.shape}")
        context = np.mean(targets_flat) if targets_flat.size > 0 else 0.0
    
    # Asegurar que context tenga la forma correcta
    if np.isscalar(context):
        context = np.array([[context]])
    elif context.ndim == 0:
        context = np.array([[context]])
    elif context.ndim == 1:
        context = context.reshape(1, -1)
    
    # Si context tiene más de 2 dimensiones, aplanar
    if context.ndim > 2:
        context = context.reshape(context.shape[0], -1)
    
    print("FEATURES SHAPE", np.shape(features))    
    print("TARGETS SHAPE", np.shape(targets))    
    print("MULTI LORAX CONTEXT SHAPE", np.shape(context))
    context = context.flatten()    
    try:
        loop = asyncio.get_running_loop()
        return asyncio.create_task(continuous_decode_async(features, targets, context, batch_size, do_forward))
    except RuntimeError:
        return asyncio.run(continuous_decode_async(features, targets, context, batch_size, do_forward))

def parallel_continuous_multi_lorax(features_train, decoded_targets, cross_validation_weight, min_len=4000, batch_size=250, do_forward=True):
    # Incluir cross_validation_weight en el cálculo de actual_len
    actual_len = min(len(features_train), len(decoded_targets))
    if actual_len == 0:
        logger.warning("parallel_continuous_multi_lorax: datos vacíos")
        return None
    
    features_trim = np.float64(features_train[:actual_len])
    targets_trim = np.float64(decoded_targets[:actual_len])
    weights_trim = np.float64(cross_validation_weight[:actual_len])
    
    logger.debug(f"parallel_continuous_multi_lorax: features={features_trim.shape}, targets={targets_trim.shape}, weights={weights_trim.shape}")
    
    result = continuous_multi_lorax(features_trim, targets_trim, weights_trim, batch_size=batch_size, do_forward=do_forward)
    
    return result