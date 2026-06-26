#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from .subproblem import *

s = lambda smax, b, B: 1/smax+(smax-(1/smax))*((b-1)/(B-1))

#taken from Pegasos algorithm by avaitla
def SGD(a, lab, Q, lr):
    for i in range(20):
        iterations = 1
        for tau in range(len(a)):
            if a[tau] > 0:
                wx = a @ Q[:,tau]
                a[tau] *= (1 - 1/iterations)
                if(lab[tau]*wx < 1):
                    a[tau] += lab[tau]/(lr * iterations)
                iterations += 1
    return a

def backpropagate_attention_sgd(a,x,y, s_max, s_scalar):
    """
    This method performs backpropagation with 
    Gradient Descent and an attention mechanism
    """
    #Gradient descent attention (backpropagation context vector 
    #decoding-encoding with a gradient)
    #print("RUNNING ATTENTION WITH", np.shape(np.mat(x)), np.shape(np.mat(y)), np.shape(a), s_max, s_scalar)
    #decoded_states = asyncio.run(parallel([1], 1, (np.mat(x),np.mat(y),a, s_max, s_scalar), func=attention, index = False, shared=False, fifo = False, lifc = False, continuous = False))
    decoded_states = attention(np.mat(x), np.mat(y), a, s_max, s_scalar)
    #print("DECODED STATES", decoded_states)
    gh = abs(1-np.min( decoded_states )) * np.gradient(x.T.dot(a),axis=0)
    return np.array(gh)

def forward_attention_sgd(a, gh, s_size, i): 
    """
    This method performs a forward operation with Stochastic Gradient Descent
    """
    #Hidden states (decode -estimate upcoming data- model hypothesis/output)    
    try:
        #print("FORWARDING", np.array(gh[i]))
        et = sigmoid(np.array(gh[i]))
        s_max = 1.5
        #s_max = 0.8
        s1 = s(s_max,i, s_size)
        #Backward use hyperbolic cosine to prevent catastrophic forgetting
        #and keep the important gradients of the losses
        ql =(s_max * np.abs(np.cosh(s1 * et)+1)) / (s1*np.abs(np.cosh(s1 * et)+1))
        try:
            a[i] -= ql
        except Exception as e:
            a[i] -= ql[i]
    except Exception as e:
        logger.exception(e)
    return a[i]
    
async def parallel_attention_sgd(steps, args, smax, batch_size = 2000):      
    """
    Perform parallel backpropagation with
    attention mechanism and continuous batching
    """
    #assert dataset is not leaked
    #or hijacked before backpropagating
    #print("ARGS0", type(args[0]))
    if type(args[0]) != np.ndarray:
        assert is_shareable_data(args[0])
    else:
        pass
    values = args[0]
    batches = int(np.ceil(len(values)/batch_size))
    gh_args = deepcopy(args)
    idxs = np.array([i for i in range(len(args[0]))])
    x = gh_args[1]
    y = gh_args[2] 
    gh_args.append(idxs)
    # Enqueue the total number of tasks per time
    for _ in range(steps):
        try:
            binit = 0
            bend = batch_size
            ghs = []
            continuous_batches = []
            #Share weights internally to update continously
            gh_args[0] = np.array(values)
            for b in range(batches):
                batch = deepcopy(values[binit:bend])
                gh_args[0] = batch
                gh_args[1] = x[binit:bend,binit:bend]
                gh_args[2] = y[binit:bend]
                #print("GH ARGS", gh_args)
                ghs = asyncio.run(parallel([1], steps, gh_args[:5], func=backpropagate_attention_sgd, index = False))
                if len(ghs) == 1:
                    #np.append("GH0", ghs[0], np.shape(ghs[0]))
                    vargs = (ghs[binit:bend])
                    #print("VARGS", np.shape(vargs))
                    if len(vargs) == 0:
                        #print("GHS", np.shape(np.array(ghs).T))
                        #print("BINIT", binit, 'BEND', bend)
                        vargs = (list(np.array(ghs).T[binit:bend].T))
                        #print("VARGS", np.shape(vargs))
                        #if len(vargs) == 1:
                        #    vargs = (list(np.array(ghs).T[binit:bend]))
                    else:
                        vargs = (list(np.array(ghs).T[binit:bend].T))
                else:
                    vargs = ghs[binit:bend]
                if np.shape(vargs)[1] == 0:
                    #np.append("GH0", ghs[0], np.shape(ghs[0]))
                    vargs = (np.array(ghs))                    
                #print("VARGS GIVEN", np.shape(vargs))
                #print("GHS", np.shape(ghs))
                #print("BATCH", np.shape(batch) )
                continuous_batches = np.array(asyncio.run(parallel(batch, 1, vargs, func=forward_attention_sgd, index=True, shared=True, continuous = b, recursion_size = [2,len(values)] )))
                #print("CONTINUOUS BATCHES", continuous_batches)
                try:
                    values[binit:bend] = continuous_batches[binit:bend]
                except Exception as e:
                    values[binit:bend] = continuous_batches
                binit += batch_size
                bend += batch_size
        except Exception as e:
            logger.exception(e)
            continue 
    #Decode all as (x, hidden as from x, context a from layers -kernels-)        
    return values
