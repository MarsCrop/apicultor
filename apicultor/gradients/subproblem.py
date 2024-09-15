#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from ..arch.thread import *
import numpy as np

s = lambda smax, b, B: (1/smax)+(smax-(1/smax))*((b-1)/(B-1))
sigmoid = lambda x: 1 / (1 + np.power(np.e, (-1 * x)))
g = lambda lab, a0, Q: 1 - lab * np.sum(a0 * Q)                 
Q_a = lambda a, Q: np.sum(a) - (np.sum(np.sum(a * Q, axis = 0), axis = 0)/2)

def context_vector(hidden_states, attention_weights):
    #print("HIDDEN STATES", np.shape(hidden_states))
    #print("HIDDEN STATES", np.shape(attention_weights))
    try:           
        return np.sum(asyncio.run(parallel([1], 1, (attention_weights, hidden_states), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))[0], axis=1)
    except Exception as e:           
        try:
            return np.sum(asyncio.run(parallel([1], 1, (attention_weights, hidden_states.T), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))[0], axis=1)
        except Exception as e:
            return np.sum(asyncio.run(parallel([1], 1, (attention_weights, hidden_states.T), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))[0], axis=0)

def decode(context, x, y, s_max, s_scalar):
    """
    This method performs decoding with
    an attention layer
    """
    return asyncio.run(parallel([1], 1, (x, y, context, s_max, s_scalar), func=attention, index = False, shared=False, fifo = False, lifc = False, continuous = False))

def substraction_fun(a,b):
    """
    This is a multiplication function that 
    wraps numpy multiply
    """
    return a-b


def mul_fun(a,b):
    """
    This is a multiplication function that 
    wraps numpy multiply
    """
    if type(a) == list:           
        a = a[0]
    try:           
        return np.multiply(a, b)
    except Exception as e:           
        try:
            return np.multiply(a.T, b)
        except Exception as e:
            try:
                return np.multiply(a, b.T)
            except Exception as e:
                try:
                    return np.multiply(a.reshape(-1, np.shape(a)[1]), b)
                except Exception as e:
                    try:
                        return np.multiply(a.reshape(-1, np.shape(a)[1]), b.T)
                    except Exception as e:
                        try:
                            return np.multiply(a.reshape(-1, np.shape(a)[1]), b.T[0][0])
                        except Exception as e:
                            return np.multiply(a.reshape(-1, np.shape(a)[1]), b.T[0][0].T)

def gather_layers_outputs(accepted_targets):
    """
    This method gathers all output layers to
    produce a single output
    """
    #print("ACCEPTED TARGETS", accepted_targets)
    try:
        tensor2tensor_dot = np.array([ asyncio.run(parallel([1], 1, (accepted_targets[i], accepted_targets[i+1].T), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False)) for i in range(len(accepted_targets)-1)])
        #print("TENSOR2TENSOR DOT", tensor2tensor_dot, np.shape(tensor2tensor_dot))
        return np.sum(tensor2tensor_dot, axis=1)
    except Exception as e:
        logger.exception(e)
        #print("CONTRACTING TO AXIS 0")
        return np.sum(tensor2tensor_dot, axis=0)

def forward(attention_function, x, hidden_states, s_max, s_size):
    """
    Thi method applies deep learning forwarding. If you're applying
    multidimensionality on hidden states, please contract hidden states
    to forward with the hidden states vector
    """
    attention_scores = np.tanh(attention_function)
    attention_weights_proba = sigmoid(attention_scores)
    if len(attention_weights_proba) == 1:           
        attention_weights_proba = attention_weights_proba[0]
    context = asyncio.run(parallel([1], 1, (hidden_states, attention_weights_proba), func=context_vector, index = False, shared=False, fifo = False, lifc = False, continuous = False))[0].T
    #print("FORWARDED CONTEXT", context)
    return asyncio.run(parallel([1], 1, (context, x, hidden_states, s_max, s_size), func=decode, index = False, shared=False, fifo = False, lifc = False, continuous = False))[0][0], attention_scores, attention_weights_proba, context

#Q = encoded data, K = y target, V = related weight val 
def attention(Q, K, V, s_max, s_scalar):
    """
    A function that approximates the input to a subset based
    on scoring the most relevant parts in it
    """
    #print("GETTING NUM WITH Q AND K", np.shape(Q), np.shape(K.T))
    #print("SIZE OF K", len(K))
    if len(K) == 1:           
        num = asyncio.run(parallel([1], 1, (Q, K.T), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))
        #print("GOT NUM 0", np.shape(num))
        #print("SIZE OF Q DIFFERS FROM SIZE OF NUM", len(num) != len(Q))
    else:            
        #print("GOT NUM 1")
        num = asyncio.run(parallel([1], 1, (Q.T, K), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))
    if len(num) != len(Q):            
        #print("GETTING NUM AGAIN")
        num = asyncio.run(parallel([1], 1, (Q, K), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))
        try:
            #print("GOT NUM 2 Q",Q,"KT",K.T)
            num = asyncio.run(parallel([1], 1, (Q, K.T), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))       
        except Exception as e:
            #print("GOT NUM 3")
            try:
                num = asyncio.run(parallel([1], 1, (Q.T, K), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))
            except Exception as e:
                num = asyncio.run(parallel([1], 1, (Q, K), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))
    try:            
        denum = np.sqrt(K.shape[0])
        r = num / denum 
    except Exception as e:               
        #print("GETTING NUM AGAIN", num)
        num = asyncio.run(parallel([1], 1, (Q, K), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))
        denum = np.sqrt(K.shape[0])   
        r = num / denum
    try:
        return asyncio.run(parallel([1], 1, (asyncio.run(parallel([1], 1, (s(s_max, r, s_scalar), sigmoid(r)), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False)), V), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))         
    except Exception as e:
        return asyncio.run(parallel([1], 1, (asyncio.run(parallel([1], 1, (s(s_max, r, s_scalar), sigmoid(r)), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False)), V.T), func = mul_fun, index = False, shared=False, fifo = False, lifc = False, continuous = False))  

def satisfy_pad_requirements(count_rows, count_reverted, arrays, is_vector = False):
    if len(set(count_rows)) != 1:
        pad_solve = True
    elif len(set(count_reverted)) != 1:
        pad_solve = True
    else:
        pad_solve = False
    if pad_solve is True:
        if is_vector == False:
            #print("Solving padding requirements")
            # Find the maximum number of columns
            max_rows = max(count_rows)
            max_cols = max(count_reverted)
            expected_size = max(max_rows * len(arrays), max_cols * len(arrays))
            # Pad each array to have the same number of columns^I             
            padded_arrays = np.array([np.pad(arr, ((0, max_rows - arr.shape[0]), (0, max_cols - arr.shape[1])), mode='constant', constant_values=0) for arr in arrays])
        else:
            #print("Solving padding requirements")
            # Find the maximum number of columns
            max_rows = max(count_rows)
            padded_arrays = np.zeros((len(arrays), max_rows), dtype=arrays[0].dtype)        
            #print("PADDED ARRAYS", np.shape(padded_arrays))
        # Stack the padded arrays
        if not (padded_arrays.shape[0] * padded_arrays.shape[1]) >= expected_size:
            return np.vstack(padded_arrays.reshape(padded_arrays.shape[0],padded_arrays.shape[2],padded_arrays.shape[1]))
        else:
            return np.vstack(padded_arrays)
    else:
        return np.vstack(arrays)

#Q = encoded data, K = y target, V = related weight val 
def continuous_decode(features, targets, context, batch_size=5000, do_forward = False):
    """
    This methods decodes and forwards layers. Keep in mind that
    outputs should be contracted before decoding because operations
	 are batch performed.
    """
    batches = int(np.ceil(len(features) / batch_size))
    bin = 0
    bend = batch_size
    continuous_attention_function = []
    continuous_attention_y = []
    continuous_attention_scores = []
    continuous_attention_function_weights_proba = []
    continuous_attention_context = []
    for batch in range(batches):
        #apply contraction by transposition
        if len(context) == 1:
            context = context.T
        #print("BIN", bin)  
        #print("BEND", bend)           
        print("CONTINUOUS DECODE TARGETS", np.shape(np.mat(targets[bin:bend]).T))
        if len(np.mat(targets[bin:bend]).T) != batch_size:
            attended_targets = targets
        else:
            attended_targets = np.mat(targets[bin:bend]).T
        attention_function = asyncio.run(parallel([1], 1, (context[bin:bend], features[bin:bend], attended_targets, 1.5, len(features)), func=decode, index = False, shared=False, fifo = False, lifc = False, continuous = batch))[0][0]
        #handle attention from batch accordingly  
        if len(attention_function[0]) != 1:
            attention_function = attention_function[0]
        elif len(attention_function) == 1:
            attention_function = attention_function[0][0].T            
        #print("ATTENTION FUNCTION SHAPE", np.shape(attention_function))    
        if attention_function.shape[1] != np.mat(targets[bin:bend]).T.shape[1]:
            attention_function = attention_function.T[:len(targets[bin:bend])].T
        attending_features = features[bin:bend]    
        if attention_function.shape[1] != attending_features.shape[0]:
            attending_features = attending_features[:attention_function.shape[1]].T
        if attention_function.shape[1] == 0:
            break
        #print("ATTENTION FUNCTION SHAPE", attention_function, np.shape(attention_function)) 
        #print("FEATURES SHAPE", attending_features, np.shape(attending_features)) 
        continuous_attention_function.append(attention_function)                
        if do_forward == True:                
            attention_y, attention_scores, attention_function_weights_proba, forwarded_context = asyncio.run(parallel([1],1, (attention_function, attending_features, np.mat(targets[bin:bend]).T, 1.5, len(features)), func=forward, index = False, shared=False, fifo = False, lifc = False, continuous = batch))[0]
            if len(context) == 1:
                context = context.T           
            elif len(context) == 0:
                context = context.T     
            bin += batch_size
            bend += batch_size
            continuous_attention_y.append(attention_y[0])
            continuous_attention_scores.append(attention_scores)
            continuous_attention_function_weights_proba.append(attention_function_weights_proba)
            continuous_attention_context.append(forwarded_context.T)
    if do_forward == True:
        continuous_attention_function_padding_count_rows = [len(np.array(arr)) for arr in continuous_attention_function]
        continuous_attention_y_padding_count_rows = [len(np.array(arr)) for arr in continuous_attention_y]
        continuous_attention_scores_padding_count_rows = [len(np.array(arr)) for arr in continuous_attention_scores]
        continuous_attention_function_weights_proba_padding_count_rows = [len(np.array(arr)) for arr in continuous_attention_function_weights_proba]
        continuous_attention_context_padding_count_rows = [len(np.array(arr)) for arr in continuous_attention_context]
        continuous_attention_function_padding_count_reverted = [np.array(arr).shape[1] for arr in continuous_attention_function]
        continuous_attention_y_padding_count_reverted = [np.array(arr).shape[1] for arr in continuous_attention_y]
        continuous_attention_scores_padding_count_reverted = [np.array(arr).shape[1] for arr in continuous_attention_scores]
        continuous_attention_function_weights_proba_padding_count_reverted = [np.array(arr).shape[1] for arr in continuous_attention_function_weights_proba]
        continuous_attention_context_padding_count_reverted = [np.array(arr).shape[1] for arr in continuous_attention_context]
        #print("ATTENTION FUNCTION COUNT", continuous_attention_function_padding_count_rows)
        #print("ATTENTION Y COUNT", continuous_attention_y_padding_count_rows)
        #print("ATTENTION SCORES COUNT", continuous_attention_scores_padding_count_rows)
        #print("ATTENTION WEIGHTS COUNT", continuous_attention_function_weights_proba_padding_count_rows)
        #print("ATTENTION CONTEXT COUNT", continuous_attention_context_padding_count_rows)
        #print("ATTENTION FUNCTION REVERTED COUNT", continuous_attention_function_padding_count_reverted)
        #print("ATTENTION Y REVERTED COUNT", continuous_attention_y_padding_count_reverted)
        #print("ATTENTION SCORES REVERTED COUNT", continuous_attention_scores_padding_count_reverted)
        #print("ATTENTION WEIGHTS REVERTED COUNT", continuous_attention_function_weights_proba_padding_count_reverted)
        #print("ATTENTION CONTEXT REVERTED COUNT", continuous_attention_context_padding_count_reverted)
        try:
            continuous_attention_function = satisfy_pad_requirements(continuous_attention_function_padding_count_rows, continuous_attention_function_padding_count_reverted, continuous_attention_function)
            continuous_attention_y = satisfy_pad_requirements(continuous_attention_y_padding_count_rows, continuous_attention_y_padding_count_reverted, continuous_attention_y)
            continuous_attention_scores = satisfy_pad_requirements(continuous_attention_scores_padding_count_rows, continuous_attention_scores_padding_count_reverted, continuous_attention_scores)
            continuous_attention_function_weights_proba = satisfy_pad_requirements(continuous_attention_function_weights_proba_padding_count_rows, continuous_attention_function_weights_proba_padding_count_reverted, continuous_attention_function_weights_proba)
            continuous_attention_context = satisfy_pad_requirements(continuous_attention_context_padding_count_rows, continuous_attention_context_padding_count_reverted, continuous_attention_context)
            print("RETURNING 5 ARGS")
            return np.nan_to_num(continuous_attention_function), np.nan_to_num(continuous_attention_y), np.nan_to_num(continuous_attention_scores), np.nan_to_num(continuous_attention_function_weights_proba), np.nan_to_num(continuous_attention_context)
        except Exception as e:
            return None
    else:
        return np.hstack(continuous_attention_function)

#Q = encoded data, K = y target, V = related weight val 
def continuous_multi_lorax(features, targets, a, batch_size=5000, do_forward= False):
    """
    A function that performs continuous attention
    with batches of a context vector given Q, K and V
    values with a fixed batch size.
    """
    context = np.nan_to_num(asyncio.run(parallel([1], 1, (targets, a), func = context_vector, index = False, shared=False, fifo = False, lifc = False, continuous = False)))
    print("FEATURES SHAPE", np.shape(features))    
    print("TARGETS SHAPE", np.shape(targets))    
    print("MULTI LORAX CONTEXT SHAPE", np.shape(context))        
    return asyncio.run(parallel([1], 1, (features, targets, context, batch_size, do_forward), func=continuous_decode, index = False, shared=False, fifo = False, lifc = False, continuous = False))

def parallel_continuous_multi_lorax(features_train, decoded_targets, cross_validation_weight, min_len=4000):
    return asyncio.run(parallel([1], 1, (np.float64(features_train), np.float64(decoded_targets)[:min_len], np.float64(cross_validation_weight)[:min_len], 250, True),func=continuous_multi_lorax, shared=False, index=False))[0][0]



