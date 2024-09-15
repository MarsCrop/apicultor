#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
import time
from ..gradients.subproblem import *
from .fairness import *
from .dependency import *
from .explain import *
from random import sample
import logging
import warnings
import signal
import os
from pathos.pools import ParallelPool as Pool

warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_imputed_data(dataset_shape=(4000,4000), imputed_min_features = 0, imputed_max_features = 0, regression = True, min_val = None, max_val = None):
    """
    This method generates imputated data
    between a random range of values for
    train/test with imputated samples. 
    :param targs: predictive targets
    :param classes: predicted targets
    :returns:                                                                                                         
      - the mean squared error score (higher than 0.5 should be a good indication)
    """
    if regression == True:
        imputed_data = np.array([[np.random.uniform(imputed_min_features, imputed_max_features) for i in range(dataset_shape[1])] for i in range(dataset_shape[0])])
        imputed_targets = np.array([[np.random.uniform(min_val, max_val) for i in range(dataset_shape[1])] for i in range(dataset_shape[0])])
        print("TARGETS SIZE", len(imputed_targets))
        return imputed_data, imputed_targets
    else:
        imputed_data = np.vstack(np.array([[np.random.uniform(min_feature_val[j], max_feature_val[j]) for j in range(dataset_shape[1])] for i in range(dataset_shape[0])])).T
        imputed_targets = np.array([[np.random.uniform(min_val, max_val) for j in range(dataset_shape[1])] for i in range(dataset_shape[0])])
        return imputed_data, imputed_targets

def define_adversary_data(dataset, y, categories = None, unequal_treatment_factor = 0.2, hyp = 0, impute=False, regression=True, imputed_min_features = [], imputed_max_features = [], min_val = None, max_val = None):
    """
    This method defines an adversary data with
    high parity for model testing. It computes
    parity to help making sure values are correct.
    This can help reducing biases and stereotypes
    when productionizing the model or stochasticity 
    if doing continuous batching testing    
    :param targs: predictive targets
    :param classes: predicted targets
    :returns:                                                                                                         
      - the mean squared error score (higher than 0.5 should be a good indication)
    """
    if hyp == 'var':
        hyp = np.var(y)
    elif hyp == 'std':
        hyp = np.std(y)
    elif hyp == 'mean':
        hyp = np.mean(y)
    elif hyp == 'median':
        hyp = np.median(y)
    if regression == False:
        categories = np.unique(categories)
    else:
        cls0 = np.where(y > hyp)[0]
        cls1 = np.where(np.logical_not(y > hyp))[0]
        categories = [cls0, cls1]        
    adversary_datasets = []
    adversary_targets = []
    #Adversary 1: dataset that could be biased 
    #is based on less learnt information towards 
    #class 1 
    adversary_group = []   
    adversary_group_y = []    
    for i in range(len(categories)):
        if impute == False:
            adversary_set_size = len(np.unique(categories[i]))
        else:
            adversary_set_size = int(np.floor(len(dataset[np.unique(categories[i])]) * unequal_treatment_factor)) 	
        print("UNEQUAL TREATMENT FACTOR", unequal_treatment_factor)
        print("ADVERSARY SET SIZE", adversary_set_size)
        print("DATASET SHAPE TO IMPUTE", np.shape(dataset))
        imputed_dataset, imputed_targets = generate_imputed_data( (adversary_set_size,dataset.shape[1]), dataset.min(), dataset.max(), regression = regression, min_val = np.array(y[categories[i]] ).min(), max_val = np.array(y[categories[i]]).max())      	
        print("IMPUTED DATASET", imputed_dataset, np.shape(imputed_dataset))
        adversary_group.append(imputed_dataset)
        adversary_group_y.append(imputed_targets)   
    return np.vstack(adversary_group), np.vstack(adversary_group_y)                 

def scan_data_leakage(y, ytest, wrong_yrate=.2, error_threshold = 0):
    """
    This method detects data leakage by counting
    the total number of wrong predicted samples using
    a threshold   
    :param y: y
    :param y_test: y_test
    :param wrong_yrate: threshold by which is estimated
    there is data leakage
    :returns:                                                                                                         
      - the mean squared error score (higher than 0.5 should be a good indication)
    """
    error = np.array([y[i] -ytest[i] for i in range(len(y))])
    print("ERROR SHAPE", np.shape(error))
    wrongly_predicted_samples = np.where(error > error_threshold)[0]
    correctly_predicted_samples = y[np.unique(np.where(np.logical_not(error > error_threshold))[0])]
    print("WRONGLY PREDICTED SAMPLES", y[np.unique(wrongly_predicted_samples)] )
    wpr = len(wrongly_predicted_samples)/y.size
    print("Wrong prediction rate:", wpr*100, "%")    
    if wpr > wrong_yrate:
        print("Data leakage test: FAILED") 
    else:
        print("Data leakage test: OK")
    return wpr, correctly_predicted_samples, y[np.unique(wrongly_predicted_samples)], wrongly_predicted_samples, error

# it's going to work with MEM DSVM implementation
class DiscriminationError(Exception):
    def __init__(self):
        super().__init__("Instance violates demographic parity")
        
class DiscriminationError(Exception):
    def __init__(self,e):
        super().__init__(("Something happened!",e,"Continuing"))

def hard_kill_pool(pids, pt):
    #terminates a function
    for pid in pids:
        os.kill(pid, signal.SIGINT)
    #terminates a tree of processes
    pt.terminate()        
        
async def productionize(model,features_train,targets_train,features_test,targets_test,features,targets,C,reg,k0,k1,last,criteria,intersects,logical,track_conflict):
    try:
        training_time = time.time()
        await model.fit_model(features_train, targets_train, k0, k1,
                                C, reg, 1./features_train.shape[1], 0.8)
        training_time = np.abs(training_time - time.time())
        #timing['training_times'][i].append(training_time)
        #print("Training time is: ", training_time)
        clf_predictions_train = model.predictions(features_train, targets_train)
        train_score = score(targets_train, clf_predictions_train)
        # Train statistical parity
        protection_val_rule_train = p_rule(clf_predictions_train, targets_train, model.w, features_train, model.proba)
        if type(protection_val_rule_train) != bool and protection_val_rule_train >= .8:
            #print('Statistical parity at instance training is',protection_val_rule_train)
            parity_train = protection_val_rule_train
        else:
            return DiscriminationError
        pex, cex, vis = explain(model, features_train, targets_train, criteria, intersects, logical)
        # depends on linked classes
        print("Train parent explanation: ", pex)
        # a subgroup
        print("Train child explanation: ", cex)
        print("Train BTC: ", BTC(targets_train, clf_predictions_train))
        bec, cons = BEC(targets_train, clf_predictions_train)
        #print("Train BEC: ", bec)
        #print("Train Scoring Error is: ",train_score)
        clf_predictions_test = model.predictions(
                    features_test, targets_test)
        test_score = score(targets_test, clf_predictions_test)
        # Test statistical parity
        protection_val_rule_test = p_rule(
                    clf_predictions_test, targets_test, model.w, features_test, model.proba)
        if type(protection_val_rule_test) != bool and protection_val_rule_test >= .8:
            #print('Statistical parity at instance testing is',
            #              protection_val_rule_test)
            parity_test = protection_val_rule_test
        else:
            return DiscriminationError
        pex, cex, vis = explain(model, features_test, targets_test, criteria, intersects, logical)
        print("Test parent explanation: ", pex)
        print("Test child explanation: ", cex)
        #print("Test BTC: ", BTC(targets_test, clf_predictions_test))
        bec, cons = BEC(targets_test, clf_predictions_test)
        #print("Test BEC: ", bec)
        #print("Test Scoring Error is: ",test_score)
        clf_time = time.time()
        # apply weights after having found a fit for previous config
        if last is True:
            model.apply(model.best_layer[-1])
        await model.fit_model(features, targets, kernel_configs[i][0], kernel_configs[i][1],
                                C, reg, 1./features.shape[1], 0.8)
        model.written = False
        # undo apply for later
        clf_time = np.abs(clf_time - time.time())
        #print("Classification time is: ", clf_time)
        clf_predictions = model.predictions(features, targets)
        # Grid statistical parity
        protection_val_rule = p_rule(clf_predictions, targets, model.w, features, model.proba)
        if type(protection_val_rule) != bool and protection_val_rule >= .8:
            #print('Statistical parity at instance is',protection_val_rule)
            parity = protection_val_rule
        else:
            #print("Low parity in instance")
            return DiscriminationError
        mse = score(targets, clf_predictions)
        pex, cex, vis = explain(model, features, targets, criteria, intersects, logical)
        print("Parent search explanation: ", pex)
        print("Child search explanation: ", cex)
        if track_conflict == None:
            #print("BTC: ", BTC(targets, clf_predictions))
            bec, cons = BEC(targets, clf_predictions)
            #print("BEC: ", bec)
        else:
            btc = BTC(targets, clf_predictions)
            bec, cons = BEC(targets, clf_predictions, track_conflict, True)
            #print("BTC: ", btc)
            #print("BEC: ", bec)
        #print("Scoring error is: ",mse)
        return mse, train_mse, test_mse, model, cons, training_time, clf_time, train_parity, test_parity, parity
    except Exception as e:
        return ProductionizationError(logger.exception(e))

def define_weight(value, threshold, func):
    """
    Perform Cross-Validation using search to find out which is the best configuration for a layer. Ideally a full cross-validation process would need at least +4000 configurations to find a most suitable, but here we are setting parameters rationally to set accurate values. The MEM uses an automatic value for gamma of 1./n_features.  
    :param model: predictive targets
    :param features: predicted targets
    :param targets: predictive targets
    :param Cs: list of C configurations
    :param reg_params: list of reg_param configurations
    :param kernel_configs: kernels configurations
    :param intersects (type(intersects) == list): list of feature column(s) to explain
    :param logical (type(logical) == list): list of bool values expressing logical observations
    :returns:                                                                                                         
      - the best estimator values (its accuracy score, its C and its reg_param value)
    """
    if func == np.max():
        if value == threshold:
            return 0.25     
        else:
            return 0.1    
    else:
        if value == threshold:
            return 0.25 
        else:
            return 0.1    

async def stress_test_red_data(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, hyp, pidx):
    for adversary in adversaries:
        adversary_info = []
        
async def forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, targets_to_perturb, features, context):
    #print("LOADING", fname1)
    if os.path.exists(fname1):
        targets = np.load(fname1)
        context = np.load(fname2)
        mse = np.load(fname3)
        diff = np.load(fname4)
        weights = np.load(fname5)
        print(msg4, weights)
        #print(msg5, diff)   
        print("Y", np.shape(targets))        
    else:    
        print(msg1)
        min_len = min(asize, bsize)
        attention_function, targets, attention_scores, weights, context = asyncio.run(parallel([1], 1, (targets_to_perturb[:min_len], features[:min_len], context[:min_len], 250, True), func=continuous_decode, shared=False, index=False))[0]
        print(msg2, np.shape(np.array(targets)/np.array(targets).max(), np.shape(targets_to_perturb)))
        try:
            mse = mean_squared_error(np.array(np.float64(targets))[:min_len].reshape(min_len,-1)/np.array(np.float64(targets))[:min_len].max(), np.array(np.float64(targets_to_perturb[:min_len].reshape(min_len,-1))))
        except Exception as e:
            min_lenj = min(len(targets[0]), len(targets_to_perturb.T[0]))
            #print("MIN LENJ COMPARED TO MIN LEN", min_lenj, min_len)
            #mse = mean_squared_error(np.array(np.float64(targets))[:min_len].reshape(min_lenj,-1)/np.array(np.float64(targets))[:min_len].max(), np.array(np.float64(targets_to_perturb[:min_len].reshape(min_len,-1))))
            mse = 1
        try:
            diff = np.array([np.array(np.float64(targets))[i] - np.array(np.float64(targets_to_perturb))[i] for i in range(min_len)])
        except Exception as e:
            min_lenj = min(len(targets[0]), len(targets_to_perturb.T[0]))
            try:
                diff = np.array([np.array(np.float64(targets))[i][:min_lenj] - np.array(np.float64(targets_to_perturb.T))[i][:min_lenj] for i in range(len(targets))]).T
            except Exception as e:
                diff = targets
        print(msg4, np.shape(weights))
        print(msg5, np.shape(diff))
        np.save(fname1, targets[:min_len])
        np.save(fname2, context)
        np.save(fname3, mse)
        np.save(fname4, diff)
        np.save(fname5, weights)
    print(msg3, np.shape(mse))
    return targets, context, mse, diff, weights

async def process_perturbation(step, hyp, pidx, context, targets, features, msg_suffix, prefix, exp_data, adversary, locally_features):
    fname1 = f"{prefix} {step} {msg_suffix} TARGETS {context.upper()} HYPOTHESIS {hyp} FEATURE {pidx}.npy"
    fname2 = f"{prefix} {step} {msg_suffix} CONTEXT {context.upper()} HYPOTHESIS {hyp} FEATURE {pidx}.npy"
    fname3 = f"{prefix} {step} {msg_suffix} TARGETS {context.upper()} MSE HYPOTHESIS {hyp} FEATURE {pidx}.npy"
    fname4 = f"{prefix} {step} {msg_suffix} TARGETS {context.upper()} WEIGHTS HYPOTHESIS {hyp} FEATURE {pidx}.npy"
    fname5 = f"{prefix} {step} {msg_suffix} TARGETS {context.upper()} DIFFERENCE HYPOTHESIS {hyp} FEATURE {pidx}.npy"

    asize = len(context)
    bsize = len(targets)
    msg1 = ""
    msg2 = f"{msg_suffix} OUTPUTS {context.upper()} IN CONTEXT"
    msg3 = f"{msg_suffix} targets {context.lower()} mse:"
    msg4 = f"{msg_suffix} WEIGHTS {context.upper()}:"
    msg5 = f"{msg_suffix} DIFFERENCE {context.upper()}:"

    targets_out, context_out, mse, diff, weights = await forward_perturbation(
        fname1, fname2, fname3, fname4, fname5, asize, bsize, 
        msg1, msg2, msg3, msg4, msg5, targets, features, context
    )

    exp_data.append([adversary, locally_features, targets_out, weights, hyp, True, mse, diff])

async def run_perturbations_on_jth_features(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, step, hyp, leakage, leaking_targets):
    perturbation_row = [] 
    for pidx in range(features_train.shape[1]):
        perturbation_row.append(asyncio.run(perturb_feature(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, step, hyp, pidx, leakage, leaking_targets)))
    return perturbation_row    

async def hypothesis_perturbation(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, hyp, leakage, leaking_targets, perturbations, gathered_mse, error):
    hypothesis_info = []
    for step in range(perturbations):
        hypothesis_info.append(asyncio.run(run_perturbations_on_jth_features(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, step, hyp, leakage, leaking_targets))) 
    #hypothesis_info.append([features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, mse])
    least_parity = [0]
    droppedout_context = asyncio.run(dropout(hypothesis_info[0], protected_groups=least_parity, protected_lime = leakage, protected_mse=gathered_mse, error=error, protected_features= features_train))                 
    return droppedout_context     

async def multiple_hypothesis_testing(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, leakage, leaking_targets, perturbations, hypotheses, gathered_mse, error):
    adversary_info = []
    for hyp in hypotheses:
        #print("SHAPE X", features_train.shape)
        #print("Perturbing with hypothesis", hyp)
        adversary_info.append(asyncio.run(hypothesis_perturbation(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, hyp, leakage, leaking_targets, perturbations, gathered_mse, error)))      
    return adversary_info    

async def parallel_adversary_perturbation(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, hyp, leakage, leaking_targets, perturbations, hypotheses):
    exp_data = []
    for adversary in adversaries:
        exp_data.append(asyncio.run(multiple_hypothesis_testing(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, hyp, leakage, leaking_targets, perturbations, hypotheses)))  
    return exp_data    

async def perturb_feature(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, step, hyp, pidx, leakage, leaking_targets):
    exp_data = []   
    if os.path.exists("PERTURBATION "+str(step)+" FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"):
        features_train = np.load("PERTURBATION "+str(step)+" FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        features_test = np.load("PERTURBATION "+str(step)+" FEATURES TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        leaking_features = np.load("PERTURBATION "+str(step)+" LEAKING FEATURES HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        adversary['xtrain'] = np.load("PERTURBATION "+str(step)+" ADVERSARY FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        adversary['xtest'] = np.load("PERTURBATION "+str(step)+" ADVERSARY FEATURES TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
    else:    
        np.random.shuffle(features_train[:,pidx])
        np.random.shuffle(features_test[:,pidx])
        np.random.shuffle(leaking_features[:,pidx])
        np.random.shuffle(adversary['xtrain'][:,pidx])
        np.random.shuffle(adversary['xtest'][:,pidx])
        np.save("PERTURBATION "+str(step)+" FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", features_train)
        np.save("PERTURBATION "+str(step)+" FEATURES TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", features_test)
        np.save("PERTURBATION "+str(step)+" LEAKING FEATURES HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", leaking_features)
        np.save("PERTURBATION "+str(step)+" ADVERSARY FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", adversary['xtrain'])
        np.save("PERTURBATION "+str(step)+" ADVERSARY FEATURES TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", adversary['xtest'])                        
    if os.path.exists("PERTURBATION "+str(step)+" IMPUTED FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"):
        locally_imputed_features_train = np.load("PERTURBATION "+str(step)+" IMPUTED FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        locally_imputed_targets_train = np.load("PERTURBATION "+str(step)+" IMPUTED TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        locally_imputed_features_test = np.load("PERTURBATION "+str(step)+" IMPUTED FEATURES TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        locally_imputed_targets_test = np.load("PERTURBATION "+str(step)+" IMPUTED TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        locally_adversary_features_train = np.load("PERTURBATION "+str(step)+" ADVERSARY FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        locally_adversary_targets_train = np.load("PERTURBATION "+str(step)+" ADVERSARY TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        locally_adversary_features_test = np.load("PERTURBATION "+str(step)+" ADVERSARY FEATURES TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        locally_adversary_targets_test = np.load("PERTURBATION "+str(step)+" ADVERSARY TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        external_imputed_features_train = np.load("PERTURBATION "+str(step)+" EXTERNAL FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        external_imputed_targets_train = np.load("PERTURBATION "+str(step)+" EXTERNAL TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        external_imputed_features_test = np.load("PERTURBATION "+str(step)+" EXTERNAL FEATURES TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
        external_imputed_targets_test = np.load("PERTURBATION "+str(step)+" EXTERNAL TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy")
    else:    
        locally_imputed_features_train, locally_imputed_targets_train = define_adversary_data(features_train, correctly_decoded_targets, unequal_treatment_factor = 1-leakage, hyp = hyp, impute=True, regression=True)
        np.save("PERTURBATION "+str(step)+" IMPUTED FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", locally_imputed_features_train)
        np.save("PERTURBATION "+str(step)+" IMPUTED TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", locally_imputed_targets_train)
        #print("LOCALLY IMPUTED TARGETS TRAIN", locally_imputed_features_train, np.shape(locally_imputed_features_train), "LOCALLY IMPUTED TARGETS TRAIN", np.shape(locally_imputed_targets_train))
        locally_imputed_features_test, locally_imputed_targets_test = define_adversary_data(features_test, targets_test, unequal_treatment_factor = 1-leakage, hyp = hyp, impute=True, regression=True)
        np.save("PERTURBATION "+str(step)+" IMPUTED FEATURES TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", locally_imputed_features_test)
        np.save("PERTURBATION "+str(step)+" IMPUTED TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", locally_imputed_targets_test)
        #print("LOCALLY IMPUTED TARGETS TEST", locally_imputed_features_test, np.shape(locally_imputed_features_test), "LOCALLY IMPUTED TARGETS TEST", np.shape(locally_imputed_targets_test))    
        locally_adversary_features_train, locally_adversary_targets_train = define_adversary_data(leaking_features, leaking_targets, unequal_treatment_factor = 1-leakage, hyp = hyp, impute=False, regression=True)
        np.save("PERTURBATION "+str(step)+" ADVERSARY FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", locally_adversary_features_train)
        np.save("PERTURBATION "+str(step)+" ADVERSARY TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", locally_adversary_targets_train)
        #print("LOCALLY ADVERSARY TARGETS TRAIN", locally_adversary_features_train, np.shape(locally_adversary_features_train), "LOCALLY ADVERSARY TARGETS TRAIN", np.shape(locally_adversary_targets_train))
        locally_adversary_features_test, locally_adversary_targets_test = define_adversary_data(features_test, targets_test, unequal_treatment_factor = 1-leakage, hyp = hyp, impute=False, regression=True)
        np.save("PERTURBATION "+str(step)+" ADVERSARY FEATURES TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", locally_adversary_features_test)
        np.save("PERTURBATION "+str(step)+" ADVERSARY TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", locally_adversary_targets_test)
        #print("LOCALLY ADVERSARY TARGETS TEST", locally_adversary_features_test, np.shape(locally_adversary_features_test), "LOCALLY ADVERSARY TARGETS TEST", np.shape(locally_adversary_targets_test))
        print("ADVERSARY FEATURES TRAIN", np.shape(adversary['xtrain']))
        print("ADVERSARY TARGETS TRAIN", np.shape(adversary['ytrain']))
        min_len = min(len(adversary['xtrain']), len(adversary['ytrain']))
        external_imputed_features_train, external_imputed_targets_train = define_adversary_data(adversary['xtrain'][:min_len], adversary['ytrain'][:min_len], unequal_treatment_factor = 1-leakage, hyp = hyp, impute=True,regression=True)
        np.save("PERTURBATION "+str(step)+" EXTERNAL FEATURES TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", external_imputed_features_train)
        np.save("PERTURBATION "+str(step)+" EXTERNAL TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", external_imputed_targets_train)
        print("EXTERNAL IMPUTED TARGETS TRAIN", external_imputed_features_train, np.shape(external_imputed_features_train), "EXTERNAL IMPUTED TARGETS TRAIN", np.shape(external_imputed_targets_train))
        min_len = min(len(adversary['xtest']), len(adversary['ytest']))
        external_imputed_features_test, external_imputed_targets_test = define_adversary_data(adversary['xtest'][:min_len], adversary['ytest'][:min_len], unequal_treatment_factor = 1-leakage, hyp = hyp, impute=True, regression=True)
        np.save("PERTURBATION "+str(step)+" EXTERNAL FEATURES TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", external_imputed_features_test)
        np.save("PERTURBATION "+str(step)+" EXTERNAL TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy", external_imputed_targets_test)
        print("EXTERNAL IMPUTED TARGETS TEST", external_imputed_features_test, np.shape(external_imputed_features_test), "EXTERNAL IMPUTED TARGETS TEST", np.shape(external_imputed_targets_test))
    external_adversary_features_train, external_adversary_targets_train = adversary['xtrain'], adversary['ytrain']
    external_adversary_features_test, external_adversary_targets_test = adversary['xtest'], adversary['ytest']
    fname1 = "PERTURBATION "+str(step)+" LOCALLY IMPUTED ADVERSARY TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" LOCALLY IMPUTED ADVERSARY CONTEXT TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" LOCALLY IMPUTED ADVERSARY TARGETS TRAIN MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" LOCALLY IMPUTED ADVERSARY TARGETS TRAIN WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" LOCALLY IMPUTED ADVERSARY TARGETS TRAIN DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(correctly_decoded_targets) 
    bsize = len(locally_imputed_features_train) 
    msg1 = "COMPUTING LOCALLY IMPUTED RESULTS"
    msg2 = "LOCALLY IMPUTED OUTPUTS TRAIN IN CONTEXT" 
    msg3 = "Locally imputed train mse:"
    msg4 = "LOCALLY IMPUTED WEIGHTS TRAIN:"
    msg5 = "LOCALLY IMPUTED DIFFERENCE TRAIN:"
    imputed_targets_train, imputed_context_train, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, locally_imputed_targets_train, locally_imputed_features_train, cross_validation_context)
    #print("IMPUTED CONTEXT TRAIN", imputed_context_train, np.shape(imputed_context_train))
    exp_data.append([adversary, features_train, imputed_targets_train, weights, hyp, True, mse, diff])
    fname1 = "PERTURBATION "+str(step)+" LOCALLY IMPUTED ADVERSARY TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" LOCALLY IMPUTED ADVERSARY CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" LOCALLY IMPUTED ADVERSARY TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" LOCALLY IMPUTED ADVERSARY TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" LOCALLY IMPUTED ADVERSARY TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(correctly_decoded_targets) 
    bsize = len(locally_imputed_features_test) 
    msg1 = ""
    msg2 = "LOCALLY IMPUTED OUTPUTS TEST IN CONTEXT" 
    msg3 = "Locally imputed test mse:"
    msg4 = "LOCALLY IMPUTED WEIGHTS TEST:"
    msg5 = "LOCALLY IMPUTED DIFFERENCE TEST:"
    try:
        imputed_targets_test, imputed_context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, locally_imputed_targets_test, locally_imputed_features_test, cross_validation_context)
        exp_data.append([adversary, features_test, targets_test, weights, hyp, True, mse, diff])
    except Exception as e:
        pass
    fname1 = "PERTURBATION "+str(step)+" LOCALLY ADVERSARY ADVERSARY TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" LOCALLY ADVERSARY ADVERSARY CONTEXT TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" LOCALLY ADVERSARY ADVERSARY TARGETS TRAIN MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" LOCALLY ADVERSARY ADVERSARY TARGETS TRAIN WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" LOCALLY ADVERSARY ADVERSARY TARGETS TRAIN DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(correctly_decoded_targets) 
    bsize = len(locally_adversary_features_train) 
    msg1 = ""
    msg2 = "LOCALLY ADVERSARY OUTPUTS TRAIN IN CONTEXT" 
    msg3 = "Locally adversary train mse:"
    msg4 = "LOCALLY ADVERSARY WEIGHTS TRAIN:"
    msg5 = "LOCALLY ADVERSARY DIFFERENCE TRAIN:"
    adversary_targets_train, adversary_context_train, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, locally_adversary_targets_train, locally_adversary_features_train, cross_validation_context)
    exp_data.append([adversary, features_train, adversary_targets_train, weights, hyp, True, mse, diff])
    fname1 = "PERTURBATION "+str(step)+" LOCALLY ADVERSARY ADVERSARY TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" LOCALLY ADVERSARY ADVERSARY CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" LOCALLY ADVERSARY ADVERSARY TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" LOCALLY ADVERSARY ADVERSARY TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" LOCALLY ADVERSARY ADVERSARY TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(correctly_decoded_targets) 
    bsize = len(locally_adversary_features_test) 
    msg1 = ""
    msg2 = "LOCALLY ADVERSARY OUTPUTS TEST IN CONTEXT" 
    msg3 = "Locally adversary test mse:"
    msg4 = "LOCALLY ADVERSARY WEIGHTS TEST:"
    msg5 = "LOCALLY ADVERSARY DIFFERENCE TEST:"
    adversary_targets_test, adversary_context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, locally_adversary_targets_test, locally_adversary_features_test, cross_validation_context)
    exp_data.append([adversary, features_test, adversary_targets_test, weights, hyp, True, mse, diff])
    fname1 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY IMPUTED TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY IMPUTED CONTEXT TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY IMPUTED TARGETS TRAIN MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY IMPUTED TARGETS TRAIN WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY IMPUTED TARGETS TRAIN DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(imputed_context_train) 
    bsize = len(locally_imputed_targets_train) 
    msg1 = ""
    msg2 = "FORWARDED LOCALLY IMPUTED OUTPUTS TRAIN IN CONTEXT" 
    msg3 = "Forwarded locally imputed train mse:"
    msg4 = "FORWARDED LOCALLY IMPUTED WEIGHTS TRAIN:"
    msg5 = "FORWARDED LOCALLY IMPUTED DIFFERENCE TRAIN:"
    targets_train, context_train, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, locally_imputed_targets_train, locally_imputed_features_train, imputed_context_train)
    exp_data.append([adversary, locally_imputed_features_train, targets_train, weights, hyp, True, mse, diff])
    fname1 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY IMPUTED TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY IMPUTED CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY IMPUTED TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY IMPUTED TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY IMPUTED TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    try:
        asize = len(imputed_context_test) 
        bsize = len(locally_imputed_targets_test)  
        msg1 = ""      
        msg2 = "FORWARDED LOCALLY IMPUTED OUTPUTS TEST IN CONTEXT" 
        msg3 = "Forwarded locally imputed test mse:"
        msg4 = "FORWARDED LOCALLY IMPUTED WEIGHTS TEST:"
        msg5 = "FORWARDED LOCALLY IMPUTED DIFFERENCE TEST:"
        targets_test, context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, locally_imputed_targets_test, locally_imputed_features_test, imputed_context_test)
        exp_data.append([adversary, locally_imputed_features_test, targets_test, weights, hyp, True, mse, diff])
    except Exception as e:
        pass
    fname1 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY ADVERSARY TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY ADVERSARY CONTEXT TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY ADVERSARY TARGETS TRAIN MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY ADVERSARY TARGETS TRAIN WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY ADVERSARY TARGETS TRAIN DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(adversary_context_train) 
    bsize = len(locally_adversary_targets_train) 
    msg1 = ""
    msg2 = "FORWARDED LOCALLY ADVERSARY OUTPUTS TRAIN IN CONTEXT" 
    msg3 = "Forwarded locally adversary train mse:"
    msg4 = "FORWARDED LOCALLY ADVERSARY WEIGHTS TRAIN:"
    msg5 = "FORWARDED LOCALLY ADVERSARY DIFFERENCE TRAIN:"
    targets_train, context_train, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, locally_adversary_targets_train, locally_adversary_features_train, adversary_context_train)
    exp_data.append([adversary, locally_adversary_features_train, targets_train, weights, hyp, True, mse, diff])
    fname1 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY ADVERSARY TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY ADVERSARY CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY ADVERSARY TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY ADVERSARY TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" REFORWARDED LOCALLY ADVERSARY TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(adversary_context_test) 
    bsize = len(locally_adversary_targets_test) 
    msg1 = ""
    msg2 = "FORWARDED LOCALLY ADVERSARY OUTPUTS TEST IN CONTEXT" 
    msg3 = "Forwarded locally adversary test mse:"
    msg4 = "FORWARDED LOCALLY ADVERSARY WEIGHTS TEST:"
    msg5 = "FORWARDED LOCALLY ADVERSARY DIFFERENCE TEST:"
    targets_test, context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, locally_adversary_targets_test, locally_adversary_features_test, adversary_context_test)
    exp_data.append([adversary, locally_adversary_features_test, targets_test, weights, hyp, True, mse, diff])
    fname1 = "PERTURBATION "+str(step)+" EXTERNAL IMPUTED TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" EXTERNAL IMPUTED CONTEXT TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" EXTERNAL IMPUTED TARGETS TRAIN MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" EXTERNAL IMPUTED TARGETS TRAIN WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" EXTERNAL IMPUTED TARGETS TRAIN DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(correctly_decoded_targets) 
    bsize = len(external_imputed_targets_train) 
    msg1 = ""
    msg2 = "EXTERNAL IMPUTED OUTPUTS TRAIN IN CONTEXT" 
    msg3 = "External imputed targets train mse:"
    msg4 = "EXTERNAL IMPUTED WEIGHTS TRAIN:"
    msg5 = "EXTERNAL IMPUTED DIFFERENCE TRAIN:"
    targets_train, imputed_context_train, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_imputed_targets_train, external_imputed_features_train, cross_validation_context)
    exp_data.append([adversary, locally_adversary_features_train, targets_train, weights, hyp, True, mse, diff])
    fname1 = "PERTURBATION "+str(step)+" EXTERNAL IMPUTED TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" EXTERNAL IMPUTED CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" EXTERNAL IMPUTED TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" EXTERNAL IMPUTED TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" EXTERNAL IMPUTED TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(correctly_decoded_targets) 
    bsize = len(external_imputed_targets_test) 
    msg1 = ""
    msg2 = "EXTERNAL IMPUTED OUTPUTS TEST IN CONTEXT" 
    msg3 = "External imputed targets test mse:"
    msg4 = "EXTERNAL IMPUTED WEIGHTS TEST:"
    msg5 = "EXTERNAL IMPUTED DIFFERENCE TEST:"
    targets_test, imputed_context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_imputed_targets_test, external_imputed_features_test, cross_validation_context)
    exp_data.append([adversary, locally_adversary_features_test, targets_test, weights, hyp, True, mse, diff])                   
    fname1 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY CONTEXT TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TRAIN MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TRAIN WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TRAIN DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(correctly_decoded_targets) 
    bsize = len(external_adversary_targets_train) 
    msg1 = ""
    msg2 = "EXTERNAL ADVERSARY OUTPUTS TRAIN IN CONTEXT" 
    msg3 = "External adversary targets train mse:"
    msg4 = "EXTERNAL ADVERSARY WEIGHTS TRAIN:"
    msg5 = "EXTERNAL ADVERSARY DIFFERENCE TRAIN:"
    targets_train, adversary_context_train, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_adversary_targets_train, external_adversary_features_train, cross_validation_context)
    exp_data.append([adversary, locally_adversary_features_train, targets_train, weights, hyp, True, mse, diff])                      
    fname1 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(cross_validation_context) 
    bsize = len(external_adversary_targets_test) 
    msg1 = ""
    msg2 = "EXTERNAL ADVERSARY OUTPUTS TEST IN CONTEXT" 
    msg3 = "External adversary targets test mse:"
    msg4 = "EXTERNAL ADVERSARY WEIGHTS TEST:"
    msg5 = "EXTERNAL ADVERSARY DIFFERENCE TEST:"
    targets_test, adversary_context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_adversary_targets_test, external_adversary_features_test, cross_validation_context)
    exp_data.append([adversary, locally_adversary_features_test, targets_test, weights, hyp, True, mse, diff])                      
    fname1 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED CONTEXT TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TRAIN MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TRAIN WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TRAIN DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(imputed_context_train) 
    bsize = len(external_imputed_targets_train) 
    msg1 = ""
    msg2 = "FORWARDED EXTERNAL IMPUTED OUTPUTS TRAIN IN CONTEXT" 
    msg3 = "Forwarded External imputed targets train mse:"
    msg4 = "FORWARDED EXTERNAL IMPUTED WEIGHTS TRAIN:"
    msg5 = "FORWARDED EXTERNAL IMPUTED DIFFERENCE TRAIN:"
    targets_train, context_train, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_imputed_targets_train, external_imputed_features_train, imputed_context_train)
    exp_data.append([adversary, locally_adversary_features_train, targets_train, weights, hyp, True, mse, diff])                
    fname1 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(imputed_context_test) 
    bsize = len(external_imputed_targets_test) 
    msg1 = ""
    msg2 = "FORWARDED EXTERNAL IMPUTED OUTPUTS TEST IN CONTEXT" 
    msg3 = "Forwarded External imputed targets test mse:"
    msg4 = "FORWARDED EXTERNAL IMPUTED WEIGHTS TEST:"
    msg5 = "FORWARDED EXTERNAL IMPUTED DIFFERENCE TEST:"
    try:
        targets_test, context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_imputed_targets_test, external_imputed_features_test, imputed_context_test)
        exp_data.append([adversary, locally_adversary_features_test, targets_test, weights, hyp, True, mse, diff])                
    except Exception as e:
        pass
    fname1 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(adversary_context_test) 
    bsize = len(external_adversary_targets_test) 
    msg1 = ""
    msg2 = "FORWARDED EXTERNAL ADVERSARY OUTPUTS TEST IN CONTEXT" 
    msg3 = "FORWARDED External adversary targets test mse:"
    msg4 = "FORWARDED EXTERNAL ADVERSARY WEIGHTS TEST:"
    msg5 = "FORWARDED EXTERNAL ADVERSARY DIFFERENCE TEST:"
    targets_test, context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_adversary_targets_test, external_adversary_features_test, cross_validation_context)
    exp_data.append([adversary, locally_adversary_features_test, targets_test, weights, hyp, True, mse, diff])                
    fname1 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY CONTEXT TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TRAIN MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TRAIN WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TRAIN DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(correctly_decoded_targets) 
    bsize = len(external_adversary_targets_train) 
    msg1 = ""
    msg2 = "EXTERNAL ADVERSARY OUTPUTS TRAIN IN CONTEXT" 
    msg3 = "External adversary targets train mse:"
    msg4 = "EXTERNAL ADVERSARY WEIGHTS TRAIN:"
    msg5 = "EXTERNAL ADVERSARY DIFFERENCE TRAIN:"
    targets_train, context_train, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_adversary_targets_train, external_adversary_features_train, cross_validation_context)
    exp_data.append([adversary, locally_adversary_features_train, targets_train, weights, hyp, True, mse, diff])                   
    fname1 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" EXTERNAL ADVERSARY TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(cross_validation_context) 
    bsize = len(external_adversary_targets_test) 
    #print("ASIZE", asize)
    #print("BSIZE", bsize)
    msg1 = ""
    msg2 = "EXTERNAL ADVERSARY OUTPUTS TEST IN CONTEXT" 
    msg3 = "External adversary targets test mse:"
    msg4 = "EXTERNAL ADVERSARY WEIGHTS TEST:"
    msg5 = "EXTERNAL ADVERSARY DIFFERENCE TEST:"
    targets_test, context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_adversary_targets_test, external_adversary_features_test, cross_validation_context)
    exp_data.append([adversary, locally_adversary_features_test, targets_test, weights, hyp, True, mse, diff]) 
    fname1 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED CONTEXT TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TRAIN MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TRAIN WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TRAIN DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(context_train) 
    bsize = len(external_imputed_targets_train) 
    msg1 = ""
    msg2 = "FORWARDED EXTERNAL IMPUTED OUTPUTS TRAIN IN CONTEXT" 
    msg3 = "Forwarded external imputed targets train mse:"
    msg4 = "FORWARDED EXTERNAL IMPUTED WEIGHTS TRAIN:"
    msg5 = "FORWARDED EXTERNAL IMPUTED DIFFERENCE TRAIN:"
    targets_train, context_train, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_imputed_targets_train, external_imputed_features_train, imputed_context_train)
    exp_data.append([adversary, locally_adversary_features_train, targets_train, weights, hyp, True, mse, diff])                        
    fname1 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL IMPUTED TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(context_test) 
    bsize = len(external_imputed_targets_test) 
    msg1 = ""
    msg2 = "FORWARDED EXTERNAL IMPUTED OUTPUTS TEST IN CONTEXT" 
    msg3 = "Forwarded external imputed targets test mse:"
    msg4 = "FORWARDED EXTERNAL IMPUTED WEIGHTS TEST:"
    msg5 = "FORWARDED EXTERNAL IMPUTED DIFFERENCE TEST:"
    try:
        targets_test, context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_imputed_targets_test, external_imputed_features_test, imputed_context_test)
        exp_data.append([adversary, locally_adversary_features_test, targets_test, weights, hyp, True, mse, diff])     
    except Exception as e:
        pass
    fname1 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY CONTEXT TRAIN HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TRAIN MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TRAIN WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TRAIN DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(context_train) 
    bsize = len(external_adversary_targets_train) 
    msg1 = ""
    msg2 = "FORWARDED EXTERNAL ADVERSARY OUTPUTS TRAIN IN CONTEXT" 
    msg3 = "Forwarded external adversary targets train mse:"
    msg4 = "FORWARDED EXTERNAL ADVERSARY WEIGHTS TRAIN:"
    msg5 = "FORWARDED EXTERNAL ADVERSARY DIFFERENCE TRAIN:"
    targets_train, context_train, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_adversary_targets_train, external_adversary_features_train, adversary_context_train)
    exp_data.append([adversary, locally_adversary_features_train, targets_train, weights, hyp, True, mse, diff])                   
    fname1 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy" 
    fname2 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY CONTEXT TEST HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname3 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TEST MSE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname4 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TEST WEIGHTS HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    fname5 = "PERTURBATION "+str(step)+" FORWARDED EXTERNAL ADVERSARY TARGETS TEST DIFFERENCE HYPOTHESIS "+str(hyp)+" FEATURE "+str(pidx)+".npy"
    asize = len(context_test) 
    bsize = len(external_adversary_targets_test) 
    msg1 = ""
    msg2 = "FORWARDED EXTERNAL ADVERSARY OUTPUTS TEST IN CONTEXT" 
    msg3 = "Forwarded external adversary targets test mse:"
    msg4 = "FORWARDED EXTERNAL ADVERSARY WEIGHTS TEST:"
    msg5 = "FORWARDED EXTERNAL ADVERSARY DIFFERENCE TEST:" 
    targets_test, context_test, mse, diff, weights = await forward_perturbation(fname1, fname2, fname3, fname4, fname5, asize, bsize, msg1, msg2, msg3, msg4, msg5, external_adversary_targets_test, external_adversary_features_test, adversary_context_test)
    exp_data.append([adversary, locally_adversary_features_test, targets_test, weights, hyp, True, mse, diff])                   
    return exp_data

async def GridSearch(model, features, targets, Cs, reg_params, kernel_configs, last, criteria, intersects, logical, fmins_train = [], fmaxs_train = [], fmins_test = [], fmaxs_test = [], tmins_train = [], tmaxs_train = [], tmins_test = [], tmaxs_test = [], track_conflict=None,p_thresh=1e-4, regression = False, adversaries=[1], model_name= 'default'):
    """
    Perform Cross-Validation using search to find out which is the best configuration for a layer. Ideally a full cross-validation process would need at least +4000 configurations to find a most suitable, but here we are setting parameters rationally to set accurate values. The MEM uses an automatic value for gamma of 1./n_features.  
    :param model: predictive targets
    :param features: predicted targets
    :param targets: predictive targets
    :param Cs: list of C configurations
    :param reg_params: list of reg_param configurations
    :param kernel_configs: kernels configurations
    :param intersects (type(intersects) == list): list of feature column(s) to explain
    :param logical (type(logical) == list): list of bool values expressing logical observations
    :returns:                                                                                                         
      - the best estimator values (its accuracy score, its C and its reg_param value)
    """
    n = len(features)  
    attention_contexts = [] 
    attention_weights = [] 
    accepted_targets = [] 
    features_train = []
    features_test = []
    targets_train = []
    targets_test = []
    if not targets[0].size > 0:
        for i in range(len(np.unique(targets))):
            n = len(features[np.where(targets == i)])
            test_n = int((20 * n)/100)
            train_n = n - test_n
            features_train.append(features[targets == i][:train_n])
            features_test.append(features[targets == i][train_n:])
            targets_train.append(targets[targets == i][:train_n])
            targets_test.append(targets[targets == i][train_n:])
        features_test = np.vstack(features_test)
        features_train = np.vstack(features_train)
        targets_train = np.hstack(targets_train)
        targets_test = np.hstack(targets_test)
        print((np.unique))
    else:
        n = len(features)
        test_n = int((20 * n)/100)
        train_n = n - test_n
        features_train.append(features[:train_n])
        features_train = features_train[0]
        features_test.append(features[train_n:])
        features_test = features_test[0]
        targets_train.append(targets[:train_n])
        targets_train = targets_train[0]
        #print("TARGETS TRAIN", np.shape(targets))
        targets_test.append(targets[train_n:])
        targets_test = targets_test[0]
    params = defaultdict(list)
    timing = defaultdict(list)
    scores = defaultdict(list)
    train_scores = defaultdict(list)
    test_scores = defaultdict(list)
    best_estimator = defaultdict(list)
    models = defaultdict(list)
    grid_conflicts = []
    model_weights = []
    params['C'].append(Cs)
    params['reg_param'].append(reg_params)
    np.save("xtrain", features_train) 
    np.save("features_test", features_test) 
    np.save("targets_test", targets_test)    
    print("CS", params['C'])    
    #adversary_data, adversary_targets = define_adversary_data(dataset, np.arange(2), targets, unequal_treatment_factor = 0.2)
    for i in range(len(kernel_configs)):
        for j in range(len(params['C'][i])):
            if os.path.exists('CONTINUOUS MULTI LORAX ATTENTION'+str(i)+'.npy') == True:
                continuous_attention_y = np.load("CONTINUOUS MULTI LORAX Y"+str(i)+'.npy')
                continuous_attention_function_weights_proba = np.load("CONTINUOUS MULTI LORAX WEIGHTS PROBA"+str(i)+'.npy')
                continuous_attention_context = np.load("CONTINUOUS MULTI LORAX ATTENTION CONTEXT"+str(i)+'.npy')
                attention_contexts.append(continuous_attention_context)
                attention_weights.append(continuous_attention_function_weights_proba)
                accepted_targets.append(continuous_attention_y)
                continue
            else:
                training_time = time.time()
                if last is True:
                    if len(model.best_layer) > 0:
                        model.apply(model.best_layer[-1])
                model.gamma = .8
                print("FEATURES TRAIN", np.shape(features_train))
                print("TARGETS TRAIN", np.shape(targets_train))
                try:
                    await model.fit_model(np.float64(features_train[:len(targets_train)]), np.float64(targets_train), kernel_configs[i][0], kernel_configs[
                                i][1], params['C'][0][j], params['reg_param'][0][j], 1./features_train.shape[1], 0.8, regression=regression)
                except Exception as e:
                    pass
                training_time = np.abs(training_time - time.time())
                timing['training_times'].append(training_time)
                #print(str().join(("Training time is: ", str(training_time))))
                if not targets_train[0].size > 1:
                    clf_predictions_train = model.predictions(
                        features_train, targets_train)
                    train_scores['scorings'].append(
                        score(targets_train, clf_predictions_train))
                    # Train statistical parity    
                    protection_val_rule_train = p_rule(
                        np.float64(clf_predictions_train), np.float64(targets_train), model.w, features_train, model.proba,p_thresh)
                else:
                    #print("MODEL W", model.w)
                    try:
                        clf_predictions_train = model.w[0] * np.float64(features_train) + model.bias
                    except Exception as e:
                        model.w = np.array([model.w])
                        clf_predictions_train = model.w[0] * np.float64(features_train) + model.bias
                    print("TRAIN SCORES", train_scores['scorings'])
                    print("MSE", mean_squared_error(np.float64((np.nan_to_num(targets_train))), np.float64((np.nan_to_num(clf_predictions_train)))) )
                    train_scores['scorings'].append(
                        mean_squared_error(np.float64((np.nan_to_num(targets_train))), np.float64((np.nan_to_num(clf_predictions_train)))) )
                    model.proba = sigmoid(model.w[0] * features_train + model.bias)
                    # Train statistical parity    
                    #try:
                    #    protection_val_rule_train = p_rule(clf_predictions_train.T, targets_train.T, model.w[0], features_train, model.proba,p_thresh)
                    #except Exception as e:
                    #    protection_val_rule_train = p_rule(clf_predictions_train.T, targets_train.T, model.w[0], features_train.T, model.proba, p_thresh)
                    protection_val_rule_train = 1
                    print("TARGETS TRAIN", np.shape(targets_train))
                    continuous_attention_function, continuous_attention_y, continuous_attention_scores, continuous_attention_function_weights_proba, continuous_attention_context = continuous_multi_lorax(np.float64(features_train), np.float64(targets_train).sum(axis=1), np.nan_to_num(np.array([model.a[i]])).T, batch_size=250, do_forward = True)[0]
                    print("CONTINUOUS MULTI LORAX ATTENTION", continuous_attention_function)
                    print("CONTINUOUS MULTI LORAX Y", continuous_attention_y)
                    print("CONTINUOUS MULTI LORAX ATTENTION SCORES", continuous_attention_scores)
                    print("CONTINUOUS MULTI LORAX WEIGHTS PROBA", continuous_attention_function_weights_proba)
                    print("CONTINUOUS MULTI LORAX ATTENTION CONTEXT", continuous_attention_context)
                    np.save("CONTINUOUS MULTI LORAX Y"+str(i), continuous_attention_y)
                    np.save("CONTINUOUS MULTI LORAX WEIGHTS PROBA"+str(i), continuous_attention_function_weights_proba)
                    np.save("CONTINUOUS MULTI LORAX ATTENTION CONTEXT"+str(i), continuous_attention_context)
                    attention_contexts.append(continuous_attention_context)
                    attention_weights.append(continuous_attention_function_weights_proba)
                    accepted_targets.append(continuous_attention_y)
                    test_scores['scorings'].append(1)
                try:
                    if type(protection_val_rule_train) != bool and protection_val_rule_train >= .8:
                        #print(('Statistical parity at instance training is',
                        #      protection_val_rule_train))
                        train_scores['demographic_parity'].append(
                            protection_val_rule_train)
                        pass
                    else:
                        print("Discrimination and false information at training instance with parity value of", protection_val_rule_train)
                        scores['scorings'].append(1)
                        scores['demographic_parity'].append(-1)
                        models['models'].append(None)
                        model_weights.append(np.array(model.w))
                        continue
                    #pex, cex, vis = explain(
                    #    model, features_train, targets_train, criteria, intersects, logical)
                    # depends on linked classes
                    #print("Train parent explanation: ", pex)
                    # a subgroup
                    #print("Train child explanation: ", cex)
                    #print("Training BTC with ", targets_train.shape, clf_predictions_train.shape)
                    #print("Train BTC: ", BTC(targets_train, clf_predictions_train))
                    bec, cons = BEC(targets_train, clf_predictions_train)
                    #print("Train BEC: ", bec)
                    #print(str().join(("Train Scoring Error is: ",
                    #      str(train_scores['scorings'][i][j]))))
                    if not targets_train[0].size > 1:
                        clf_predictions_test = model.predictions(
                            features_test, targets_test)
                        test_scores['scorings'].append(
                            score(targets_test, clf_predictions_test))
                        # Test statistical parity    
                        protection_val_rule_test = p_rule(
                            clf_predictions_test, targets_test, model.w, features_test, model.proba,p_thresh)
                    else:
                        clf_predictions_test = model.w[0] * features_test + model.bias
                        test_scores['scorings'].append(
                            mean_squared_error(np.float64(targets_test), np.float64(clf_predictions_test)))
                        model.proba = sigmoid(np.array(model.w) * features_test + model.bias)
                        # Train statistical parity    
                        protection_val_rule_test = p_rule(
                            clf_predictions_test.T, targets_test.T, model.w[0], features_test, model.proba,p_thresh)
                    if type(protection_val_rule_test) != bool and protection_val_rule_test >= .8:
                        print(('Statistical parity at instance testing is',protection_val_rule_test))
                        test_scores['demographic_parity'].append(
                            protection_val_rule_test)
                        model_weights.append(model.w)
                        pass
                    else:
                        print("Discrimination and false information at test instance with parity value of", protection_val_rule_test)
                        scores['scorings'].append(1)
                        scores['demographic_parity'].append(-1)
                        models['models'].append(None)
                        model_weights.append(model.w)
                        continue
                    #pex, cex, vis = explain(
                    #    model, features_test, targets_test, criteria, intersects, logical)
                    #print("Test parent explanation: ", pex)
                    #print("Test child explanation: ", cex)
                    print(("Test BTC: ", BTC(targets_test, clf_predictions_test)))
                    bec, cons = BEC(targets_test, clf_predictions_test)
                    #print("Test BEC: ", bec)
                    try:
                        print((str().join(("Test Scoring Error is: ",
                          str(test_scores['scorings'][j])))))
                    except Exception as e:
                        print((str().join(("Test Scoring Error is: ",
                          str(test_scores['scorings'][-1])))))
                    clf_time = time.time()
                    # apply weights after having found a fit for previous config
                    try:
                        model.apply(model.best_layer[-1])
                    except Exception as e:
                        pass
                    model.written = True
                    # undo apply for later
                    clf_time = np.abs(clf_time - time.time())
                    timing['classifier times'].append(clf_time)
                    print((str().join(("Classification time is: ", str(clf_time)))))
                    if not targets_train[0].size > 1:
                        clf_predictions = model.predictions(features, targets)
                        scores['scorings'].append(
                            score(targets, clf_predictions))
                        #Grid statistical parity    
                        protection_val_rule = p_rule(
                            clf_predictions, targets, model.w, features, model.proba,p_thresh)
                    else:
                        clf_predictions = model.w[0] * features + model.bias
                        scores['scorings'].append(
                            mean_squared_error(np.float64(targets), np.float64(clf_predictions)))
                        model.proba = sigmoid(model.w[0] * features + model.bias)
                        #Grid statistical parity    
                        protection_val_rule = p_rule(
                            np.float64(clf_predictions.T), np.float64(targets.T), np.float64(model.w[0]), features, model.proba,p_thresh)
                    if type(protection_val_rule) != bool and protection_val_rule >= .8:
                        print(('Statistical parity at instance is',
                              protection_val_rule))
                        scores['demographic_parity'].append(protection_val_rule)
                        pass
                    else:
                        print("Discrimination and false information at test instance with parity value of", protection_val_rule)
                        scores['scorings'].append(1)
                        scores['demographic_parity'].append(-1)
                        models['models'].append(None)
                        model_weights.append(model.w)
                        continue
                    models['models'].append(model)
                    #pex, cex, vis = explain(
                    #    model, features, targets, criteria, intersects, logical)
                    #print("Parent search explanation: ", pex)
                    #print("Child search explanation: ", cex)
                    if track_conflict == None:
                        #print("BTC: ", BTC(targets, clf_predictions))
                        bec, cons = BEC(targets, clf_predictions)
                        #print("Instance BEC: ", bec)
                    else:
                        btc = BTC(targets, clf_predictions)
                        bec, conflicts = BEC(
                            targets, clf_predictions, track_conflict, True)
                        #print("BTC: ", btc)
                        #print("Instance BEC: ", bec)
                        grid_conflicts.append(conflicts)
                    print((str().join(("Scoring error is: ",
                          str(scores['scorings'][j])))))
                except Exception as e:
                    print(("Something happened! Continuing...", logger.exception(e), test_scores['scorings']))
                    if len(train_scores['scorings'])-1 == j:
                        train_scores['scorings'].pop(j)
                    if len(train_scores['scorings'])-1 != j:
                        train_scores['scorings'].append(2)
                    if len(test_scores['scorings'])-1 == j:
                        test_scores['scorings'].pop(j)
                    if len(test_scores['scorings'])-1 != j:
                        test_scores['scorings'].append(2)
                    if len(scores['scorings'])-1 == j:
                        scores['scorings'].pop(j)
                    if len(scores['scorings'])-1 != j:
                        scores['scorings'].append(2)
                    if len(scores['demographic_parity'])-1 == j:
                        scores['scorings'].pop(j)
                    #if len(scores['demographic_parity'])-1 != j:
                    #    scores['scorings'].append(100)
                    if len(train_scores['demographic_parity'])-1 != j:
                        train_scores['demographic_parity'].append(2)
                    if len(train_scores['demographic_parity'])-1 == j:
                        train_scores['demographic_parity'].pop(j)
                    if len(test_scores['demographic_parity'])-1 != j:
                        test_scores['demographic_parity'].append(2)
                    if len(test_scores['demographic_parity'])-1 == j:
                        test_scores['demographic_parity'].pop(j)
                        models['models'].append(None)
                        continue
        try:
            good_layers = np.min(np.array(test_scores['scorings']))
            accepted_layers = np.argmin(np.array(test_scores['scorings']))
            best_estimator['score'].append(good_layers)
            print(('Best scores', str(',').join([str(score) for score in best_estimator['score']]), 'Scorings', str(',').join([str(scoring) for scoring in scores['scorings']])))
            print("MODELS", models)
            print("ACCEPTED LAYERS", accepted_layers)
            best_model = np.array(models['models'])[np.array(accepted_layers)]
            best_estimator['C'].append(
                params['C'][0][accepted_layers])
            best_estimator['reg_param'].append(
                params['reg_param'][0][accepted_layers])
            print("Best estimators are: C: ", 
                best_estimator['C'], " Regularization parameter: ", best_estimator['reg_param'])
            print((str().join(("Mean training error scorings: ",
                  str(',').join(np.mean(train_scores['scorings']))))))
            print((str(',').join(("Mean test error scorings: ",
                  str(np.mean(test_scores['scorings']))))))
            print((str(',').join(("Mean training demographic parity scorings: ", str(
                np.mean(train_scores['demographic_parity']))))))
            print((str(',').join(("Mean test demographic parity scorings: ",
                  str(np.mean(test_scores['demographic_parity']))))))            
            print((str(',').join(("Standard dev train error scorings: ",
                  str(np.std(train_scores['scorings']))))))
            print((str(',').join(("Standard dev test error scorings: ",
                  str(np.std(test_scores['scorings']))))))
            print((str(',').join(("Mean error scorings: ",
                  str(np.mean(scores['scorings']))))))
            print((str(',').join(("Standard dev error scorings: ",
                  str(np.std(scores['scorings']))))))
            print((str(',').join(("Mean demographic parity scorings: ",
                  str(np.mean(scores['demographic_parity']))))))
        except Exception as e:
            logger.exception(e)
            continue
    #Contract all decoded outputs to a single layer target       
    print("Gathering layers outputs with number of accepted targets", len(accepted_targets))
    raw_decoded_targets = np.array(asyncio.run(parallel([1], 1, ((np.array(accepted_targets))),func = gather_layers_outputs, index = False, shared=False, fifo = False, lifc = False, continuous = False, as_singular_matrix = True))).T 
    np.save("raw_decoded_targets", raw_decoded_targets)
    continuous_attention_y = np.copy(raw_decoded_targets)
    best_fit = 0
    cross_validation_context = attention_contexts[best_fit]    
    decoded_targets = raw_decoded_targets / raw_decoded_targets.max()
    np.save("decoded_targets", decoded_targets)
    print("Gathered targets", np.shape(decoded_targets))
    min_len = min(len(decoded_targets), len(targets_train))
    gathered_mse = mean_squared_error(np.float64(decoded_targets[:min_len].reshape(min_len,-1)), np.float64(targets_train[:min_len].reshape(min_len,-1)))
    print("Gathered mse", gathered_mse)
    print("Targets during leakage test", np.shape(targets_train))    
    #print("DECODED CROSS VALIDATION", decoded_targets, np.shape(decoded_targets)) 
    decoded_targets = decoded_targets.reshape((decoded_targets.shape[0], decoded_targets.shape[1]))
    leakage, correctly_decoded_targets, leaking_targets, leaking_idxs, error = scan_data_leakage(np.float64(targets_train[:min_len]), np.float64(decoded_targets[:min_len]), error_threshold=gathered_mse)
    print("Correctly decoded targets", np.shape(correctly_decoded_targets)) 
    leaking_features = features_train[np.unique(leaking_idxs)]
    print("Leaking features", leaking_features, np.shape(leaking_features)) 
    print("Leaking targets", leaking_targets, np.shape(leaking_targets)) 
    #hypotheses = ['var','mean','std','median'] 
    hypotheses = ['var'] 
    decoded_targets = np.float64(decoded_targets)[:min_len]      
    min_len = min(len(correctly_decoded_targets), len(targets_train))
    correctly_decoded_targets = correctly_decoded_targets[:min_len]
    np.save("correctly_decoded_targets", correctly_decoded_targets)
    np.save("leaking_features", leaking_features) 
    np.save("leaking_targets", leaking_targets)
    #Explain and dropout
    #perturbations = 10
    #perturbations = 2
    perturbations = 1
    exp_data = []
    for adversary in adversaries:
        exp_data.append(asyncio.run(multiple_hypothesis_testing(features_train, features_test, leaking_features, adversary, correctly_decoded_targets, targets_test, cross_validation_context, leakage, leaking_targets, perturbations, hypotheses, gathered_mse, error))[0])    
    stress_becs = []
    stress_btcs = []
    stress_conflicts = []  
    stress_instances = [] 
    for instance in range(len(exp_data)):
        for i in range(len(exp_data[instance])):
            #Backward analysis of red team data  
            #print("EXP DATA:", exp_data[instance][i])
            try:
                bec_decoded, conflicts_decoded = BEC(exp_data[instance][i][1].sum(axis=1), exp_data[instance][i][2].sum(axis=1), track_conflict, True, attention=True)
            except Exception as e:
                logger.exception(e)
                if instance == 0:
                    stress_becs.append(1)
                    stress_instances.append(instance)
                    stress_btcs.append(0)
                    stress_conflicts.append([])
                continue
            print("Stress test BEC:", bec_decoded, np.mean(bec_decoded))
            btc_decoded = BTC(exp_data[instance][i][1].sum(axis=1), exp_data[instance][i][2].sum(axis=1))
            print("Stress test BTC:", btc_decoded, np.mean(btc_decoded))
            print("Stress test Conflicts:", len(conflicts_decoded))
            if instance == 0:
                stress_becs.append(np.mean(bec_decoded))
                stress_btcs.append(np.mean(btc_decoded))
                stress_conflicts.append(conflicts_decoded)
    try:
    	  #grid_conflicts.append(conflicts)
    	  max_btc = np.argmax(stress_btcs)                    
    	  min_bec = np.argmin(stress_becs) 
    	  max_conflicts = np.argmax(stress_conflicts)        
    	  stress_features_train = exp_data[stress_instances[min_bec]][min_bec][0]
    	  stress_features_train = exp_data[stress_instances[min_bec]][min_bec][0]
    	  stress_features_test = exp_data[stress_instances[min_bec]][min_bec][0]
    	  stress_targets_train = exp_data[stress_instances[min_bec]][min_bec][1] 
    	  stress_targets_test = exp_data[stress_instances[min_bec]][min_bec][1]
    	  stress_context = exp_data[stress_instances[min_bec]][min_bec][3]
    	  stress_context = np.vstack((cross_validation_context, stress_context))  
    	  np.save("MODEL CONTEXT", stress_context)  	  
    	  return stress_context
    except Exception as e:
    	  logger.exception(e)
    	  np.save("Model "+str(model_name)+" context", cross_validation_context)  	
    	  return cross_validation_context 
     

     
