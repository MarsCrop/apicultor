import numpy as np
from sklearn.metrics import accuracy_score
from .visuals import plot_regression
from random import sample, randint, shuffle
import logging
import warnings
import os
import glob
from .error import *
from ..gradients.subproblem import *
from .fairness import *
from .dependency import *
from sklearn.metrics import mean_squared_error, accuracy_score

warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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

def get_intersection(noisy_x, limit):
    if limit == 'mean':
        intersection = noisy_x.mean()
        #intersection = noisy_x[:,idxs[yi]].mean()*thresh
    if limit == 'median':
        rob_noise = np.median(noisy_x) * noisy_x         	
        intersection = np.mean(rob_noise)
        #intersection = noisy_x[:,idxs[yi]].mean()*thresh
    elif limit == 'var':
        intersection = np.var(noisy_x)
        #intersection = np.var(noisy_x[:,idxs[yi]])*thresh
    elif limit == 'std':
        intersection = np.std(noisy_x)
        #intersection = np.std(noisy_x[:,idxs[yi]])*thresh
    else:
        intersection = limit
    return intersection        

async def compute_feature_importance(dset, inst=0): 
    """
    This method performs layer dropouts via explanations. Getting accuracy scores
    for a set of layer variables will provide probability values in order to dropout
    in a set of layers. Being based on accuracy to provide feature importance in Deep
    Learning explanation is called when using layers is called Layer-wise
    Relevance Propagation. The contribution of each feature across another
    feature type can be consider a Shapley Additive Explanation. The use of
    Linear Regression to explain predictions to highlight the contributions
    of features in an agnostic way way is called Local Interpretable Model-agnostic
    Explanation. A noticeable variance in a feature's column indicates its importance.
    Attention weights and contexts can be used in explanations to highlight relevant
    features 
    :param model (type(model) == object): a usable model
    :param x (type(x) == np.darray): dataset that is going to be explained
    :param ys (type(ys) == np.darray): related targets
    :param limit (type(limit) == list): if a list of strings is given only 'mean', 'var' or 'std' are supported,
        it also supports other types. This argument sets the intersection value (eg.: [x[:,0] > 900, x[:,4] > 70])
    :param Idxs (type(intersects) == list): indexes of important features to perturb
    :param logical (type(logical) == bool or type(logical) == list): bool or a list of bools (must be of the size of limits).
        If True is passed, observability is based on positive limit. If False is passed, observability is based on not
        positive limit. If None is passed, the current limit is not observable and therefore it is contrasted against the 
        explaining values and the next limit is used against the explaining data (limit_i < data < limit_i+1)  	
    :param thresh: a threshold for robust explanation
    :returns:                                                                                                         
      - pos_explanation_scores: accuracies for data following the instances criterias
      - neg_explanation_scores: accuracies for data not following the instances criterias
      - plt: a plot of the explanation
    """
    importances = []
    #mse, x, scores
    scores = [i[-2] for i in dset if (not len(i[-2]) == 1)]
    sizes = [len(i[-2]) for i in dset  if (not len(i[-2]) == 1)]
    min_len = min(sizes)
    x = [i[1] for i in dset]
    unique_idxs = np.unique(np.array([i[-1] for i in dset]))
    mse_idxs = np.array([i[-1] for i in dset])
    pmses = np.array([i[0] for i in dset])
    mses = np.array([np.mean(pmses[np.where(mse_idxs == i)] + dset[-1][0]) for i in unique_idxs if i != -1])
    np.save("IMPORTANT MSES.npy", pmses+ dset[-1][0])    
    #print("MSES", mses)
    #print("X", x)
    #print("SIZES", sizes)
    #print("MEAN MSE", np.mean(np.array(mses))) 
    passing_scores = []   
    passing_xs = []
    for i in range(len(scores)):
        for j in range(len(mses)):
            if len(scores[i]) == len(mses):
                scores[i] = scores[i].T
            scores[i][:min_len][:,j] -= np.array(mses[j])
            passing_xs.append(x[i])
            passing_scores.append(i)
        #print("CURRENT IMPORTANCE", scores[i])    
        importances.append( scores[i] ) #feature-wise      mean_importance = np.mean(np.array(importances),axis=1)
        np.save("IMPORTANCE "+str(i)+' INSTANCE '+str(i)+'.npy',  scores[i])
    sizes = [len(i) for i in importances if len(i) != 46]
    min_len = min(sizes)
    #print("SIZES", sizes)
    #print("MIN LEN", min_len)
    mean_importance = np.mean(np.array([i[:min_len] for i in importances  if len(i) != 46] ),axis=1)  
    std_importance = np.std(np.array([i[:min_len] for i in importances  if len(i) != 46]),axis=1)  
    print("MEAN IMPORTANCE", np.mean(mean_importance,axis=0), np.shape(mean_importance))     
    print("STD IMPORTANCE", np.mean(std_importance,axis=0),  np.shape(mean_importance))
    print("IMPORTANCES", np.array(np.array([i[:min_len] for i in importances  if len(i) != 46] ))) 
    return np.array([i[:min_len] for i in importances  if len(i) != 46] ), mean_importance, std_importance, len(mses), np.unique(passing_scores), passing_xs

def forward_with_dropout(data_set, protected_groups=[], protected_lime = 0.8, importance_threshold = 0.2, protected_mse=0.2, error = [], protected_features = []):
    mses = np.load("IMPORTANT MSES.npy")
    print("DATA SET", data_set)
    importances = []
    fdir = '/home/mc/sfs-python/sfs/'
    for f in list(os.walk(fdir))[0][-1]:
        print("LOADING",f)
        if 'IMPORTANCE' in f:
            importances.append(np.load(fdir+f))
    print("IMPORTANCES", importances)    
    importances = [i for i in importances  if len(i) != 1]
    min_len = np.min([len(i) for i in importances  if len(i) != 46])
    mean_importance_thresholds = np.mean(np.array([i[:min_len] for i in importances  if len(i) != 46] ),axis=1)
    std_importance_thresholds = np.std(np.array([i[:min_len] for i in importances  if len(i) != 46]),axis=1)     
    #print("STD THRESHOLDS", std_importance_thresholds)
    importance_idx = 0
    droppedout_sets = []
    relevant_weights = [] 
    wset = data_set[-2]
    try:
        #importance_threshold = min(mean_importance_thresholds[importance], std_importance_thresholds[importance])
        importance_threshold = mean_importance_thresholds[importance_idx]
        #print("CURRENT WEIGHTS", wset[instance][i])
        #print("IMPORTANCE THRESHOLD", importance_threshold)
        if  (len(wset) == 1):
            pass
        elif len(wset) == mses_size:
            pass
        else:
            importance_mask = importances[instance][importance_idx] >= importance_threshold
            if np.all([False == i for i in importance_mask]) == True:
                importance_idx += 1
                pass
            else:
                wset *= importance_mask  
                explaining_mask = importances[instance][most_important_feature] >= importance_threshold
                x = important_features_per_model_perturbation[instance][-1][importance_idx]
                if protected_groups in neg_classes:
                    not_protected_group = np.where(x[:,most_important_feature] > intersection)
                else:
                    not_protected_group = np.where(np.logical_not(x[:,most_important_feature] > intersection))                    
                wset[not_protected_group] *= explaining_mask         
    except Exception as e:
        pass
    x, y = data_set[0], data_set[1]    
    attention_function, targets, attention_scores, weights, context = asyncio.run(parallel([1], 1, (y[:min_len], x[:min_len], wset[:min_len], 250, True), func=continuous_decode, shared=False, index=False))[0]
    decoded_robustness = validate_non_discrimination_and_robustness(targets, context, attention_scores, y, weights, 1e-4, 0)
    mse_decoded = mse(targets, y)  
    return context    

def validate_non_discrimination_and_robustness(y, y_predicted, theta, x, proba, mse, statistical_parity_thresh=1e-4, local_parity_thresh = 0.2, class_parity_thresh = 0.2, conditional_parity_thresh = 0.2):
    """
    This method solves robustness and non discrimination by
    analyzing a set of targets, model weights, functions and 
    parity thresholds and analyze the model in terms of suboptimization
    :param y: y
    :param y_predicted: y_test
    :param theta: attention weights
    :returns:                                                                                                         
      - robustness: robustness score from 0 to 1
    """
    statistical_parity = np.float64(p_rule(y_predicted, y, theta, x, proba, thresh = 1e-4))
    #print("STATISTICAL PARITY", statistical_parity)
    xcontext = np.nan_to_num(asyncio.run(parallel([1], 1, (x, theta), func = context_vector, index = False, shared=False, fifo = False, lifc = False, continuous = False)))
    conditional_procedure_accuracy_equality = unprotection_score(mse, xcontext, y_predicted)
    if type(conditional_procedure_accuracy_equality) == np.complex128:
        conditional_procedure_accuracy_equality = np.float64(conditional_procedure_accuracy_equality)
    #print("CONDITIONAL PROCEDURE ACCURACY EQUALITY", conditional_procedure_accuracy_equality, type(conditional_procedure_accuracy_equality))
    local_parity = np.float64(ind_fairness(x, xcontext.T, y))
    #print("LOCAL PARITY", local_parity)
    class_parity = np.float64(group_fairness(local_parity))
    #print("CLASS PARITY", class_parity)
    robustness = 0
    if not statistical_parity > statistical_parity_thresh:
        print("Autorregressive discrimination in statistical parity:", statistical_parity)
    else:
        robustness += 0.25
    if not conditional_procedure_accuracy_equality > conditional_parity_thresh:
        print("Autorregressive discrimination in conditional procedure accuracy equality:", conditional_procedure_accuracy_equality)
    else:
        robustness += 0.25
    if not local_parity > local_parity_thresh:
        print("Autorregressive discrimination in local equality:",local_parity)
    else:
        robustness += 0.25
    if not class_parity > class_parity_thresh:
        print("Autorregressive discrimination in class equality:",class_parity)  
    else:
        robustness += 0.25
    return robustness     

async def dropout(data_set, protected_groups=[], protected_lime = 0.8, importance_threshold = 0.2, protected_mse=0.2, error = [], protected_features = [], logical = [True], most_important_feature = 0 ):
    """
    This method performs layer dropouts via explanations. Getting accuracy scores
    for a set of layer variables will provide probability values in order to dropout
    in a set of layers. Being based on accuracy to provide feature importance in Deep
    Learning explanation is called when using layers is called Layer-wise
    Relevance Propagation. The contribution of each feature across another
    feature type can be consider a Shapley Additive Explanation. The use of
    Linear Regression to explain predictions to highlight the contributions
    of features in an agnostic way way is called Local Interpretable Model-agnostic
    Explanation. A noticeable variance in a feature's column indicates its importance.
    Attention weights and contexts can be used in explanations to highlight relevant
    features 
    :param model (type(model) == object): a usable model
    :param x (type(x) == np.darray): dataset that is going to be explained
    :param ys (type(ys) == np.darray): related targets
    :param limit (type(limit) == list): if a list of strings is given only 'mean', 'var' or 'std' are supported,
        it also supports other types. This argument sets the intersection value (eg.: [x[:,0] > 900, x[:,4] > 70])
    :param Idxs (type(intersects) == list): indexes of important features to perturb
    :param logical (type(logical) == bool or type(logical) == list): bool or a list of bools (must be of the size of limits).
        If True is passed, observability is based on positive limit. If False is passed, observability is based on not
        positive limit. If None is passed, the current limit is not observable and therefore it is contrasted against the 
        explaining values and the next limit is used against the explaining data (limit_i < data < limit_i+1)  	
    :param thresh: a threshold for robust explanation
    :returns:                                                                                                         
      - pos_explanation_scores: accuracies for data following the instances criterias
      - neg_explanation_scores: accuracies for data not following the instances criterias
      - plt: a plot of the explanation
    """
    pos_limes = []
    neg_limes = []
    pos_idxs_set = []
    neg_idxs_set = []
    pos_classes_set = []
    neg_classes_set = []
    #Compute feature importance of attention weights
    important_features_per_model_perturbation = []
    #print("DATA SET", len(data_set))
    even_train_sizes = [ 0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 23, 25, 27]
    odd_test_sizes = [ 1, 3, 5, 7, 9, 11, 13, 14, 16, 18, 20, 22, 25]
    idx = 0
    sizes = []
    for dset in data_set:
        for j in range(len(dset)):
            try:
                sizes.append(len(dset[j][1]))
            except Exception as e:
                #logger.exception(e)
                pass
        idx += 1  
    max_train_size = np.mean(sizes)    
    #print("TRUE TRAINING SET MAX SIZE", max_train_size)    
    idx = 0
    training_importances = []     
    training_wset = []  
    passing_training_features = []   
    training_instances_to_drop = [] 
    instances_to_drop = []      
    for dset in data_set:
        for j in range(len(dset)):
            try:
                if len(dset[j][1]) >= max_train_size:
                    training_importances.append([dset[j][-2], dset[j][1], dset[j][-1], idx])
                    training_wset.append(dset[j][-5])
                    passing_training_features.append(j)
                    #print("TARGETS OUT", np.shape(dset[j][2]))
                    training_instances_to_drop.append([dset[j][1], dset[j][2] ])
            except Exception as e:
                #logger.exception(e)
                pass
        idx += 1             
    training_importances.append([protected_mse, protected_features, error, -1])        
    instances_to_drop.append(training_instances_to_drop)  
    testing_importances = []
    testing_wset = [] 
    idx = 0          
    passing_testing_features = []  
    testing_instances_to_drop = []                   
    for dset in data_set:
        for j in range(len(dset)):
            try:
                if len(dset[j][1]) < max_train_size:
                    testing_importances.append([dset[j][-2], dset[j][1], dset[j][-1], idx])
                    testing_wset.append(dset[j][-5])
                    passing_testing_features.append(j)
                    testing_instances_to_drop.append([dset[j][1], dset[j][2] ])
            except Exception as e:
                #logger.exception(e)
                pass
        idx += 1
    testing_importances.append([protected_mse, protected_features, error, -1])  
    instances_to_drop.append(testing_instances_to_drop) 
    wset = []
    important_features_per_model_perturbation.append(asyncio.run(compute_feature_importance(training_importances)))
    important_features_per_model_perturbation.append(asyncio.run(compute_feature_importance(testing_importances,1)))
    wset.append(training_wset)
    wset.append(testing_wset)
    #dataset[3][importances] >= protected_lime
    neg_classes = [0,2,4,6,8,10]
    pos_classes = [1,3,5,7,9,11]
    protected_groups = protected_groups[0]
    mses_size = important_features_per_model_perturbation[0][-3]
    droppedout_sets_per_instance = []
    relevant_weights_per_instance = []    
    relevant_weights_idx_per_instance = []  
    remaining_weights_arguments = [] 
    intersection = np.std(training_instances_to_drop[-1][0])       
    for instance in range(len(important_features_per_model_perturbation)):
        importances = important_features_per_model_perturbation[instance][0]
        mean_importance_thresholds = important_features_per_model_perturbation[instance][1]
        std_importance_thresholds = important_features_per_model_perturbation[instance][-4]
        #print("STD THRESHOLDS", std_importance_thresholds)
        if instance == 0:
            dset = training_importances
        else:
            dset = testing_importances
        importance_idx = 0
        droppedout_sets = []
        relevant_weights = [] 
        relevant_weights_idx = []
        widx = 0
        for i in range(len(wset[instance])):
            try:
                #importance_threshold = min(mean_importance_thresholds[importance], std_importance_thresholds[importance])
                importance_threshold = mean_importance_thresholds[importance_idx]
                #print("CURRENT WEIGHTS", wset[instance][i])
                #print("IMPORTANCE THRESHOLD", importance_threshold)
                if  (len(wset[instance][i]) == 1):
                    continue
                elif len(wset[instance][i]) == mses_size:
                    continue
                else:
                    importance_mask = importances[instance][importance_idx] >= importance_threshold
                    if np.all([False == i for i in importance_mask]) == True:
                        importance_idx += 1
                        continue
                    else:
                        wset[instance][i] *= importance_mask  
                        explaining_mask = importances[instance][most_important_feature] >= importance_threshold
                        x = important_features_per_model_perturbation[instance][-1][importance_idx]
                        if protected_groups in neg_classes:
                            not_protected_group = np.where(x[:,most_important_feature] > intersection)
                        else:
                            not_protected_group = np.where(np.logical_not(x[:,most_important_feature] > intersection))                    
                        wset[instance][i][not_protected_group] *= explaining_mask     
                        #print("IMPORTANCE MASK", importance_mask)
                        print("Explaining mask:", explaining_mask)
                        #print("DROPPED OUT WEIGHTS", wset[instance][i])
                        droppedout_sets.append(wset[instance][i])
                        relevant_weights.append(widx)
                        relevant_weights_idx.append(np.where([np.all([0 == j for j in i]) == False for i in wset[instance][i]]))
                        #relevant_weights_vals
                        importance_idx += 1
                        widx += 1
            except Exception as e:
                #logger.exception(e)
                pass
        droppedout_sets_per_instance.append(droppedout_sets)        
        relevant_weights_per_instance.append(relevant_weights)
        relevant_weights_idx_per_instance.append(relevant_weights_idx)
        #relevant_weights_vals_per_instance
    passing_training_scores = important_features_per_model_perturbation[0][-2]    
    passing_testing_scores = important_features_per_model_perturbation[1][-2]  
    min_len = np.min(np.array([len(i[0]) for i in instances_to_drop[0]]))
    passed_training_features = np.array([i[0][:min_len] for i in instances_to_drop[0]])[np.array(passing_training_scores)]     
    #print("TEST INSTANCES TO DROP SIZES", np.array([len(i[1]) for i in instances_to_drop[0]]))
    sizes = np.array([len(i[1]) for i in instances_to_drop[0] if len(i) != 1])
    min_len = np.min(np.array([i for i in sizes if i != mses_size]))
    passed_training_targets = [i[1] for i in instances_to_drop[0] if len(i[1]) != 1]  
    passed_training_targets = [i for i in passed_training_targets if len(i) != mses_size]
    #print("SIZE OF PASSED TRAINING TARGETS", [len(i) for i in passed_training_targets] )  
    sizes = np.array([len(i[1]) for i in instances_to_drop[1] if len(i) != 1])
    min_len = np.min(np.array([len(i[1]) for i in instances_to_drop[1]]))
    passed_testing_targets  = [i[1] for i in instances_to_drop[1] if len(i[1]) != 1]
    passed_testing_targets = [i for i in passed_testing_targets if len(i) != mses_size]  
    #print("SIZE OF PASSED TESTING TARGETS", [len(i) for i in passed_testing_targets] ) 
    passed_testing_features = [[i[0] for i in instances_to_drop[1]][i] for i in passing_testing_scores]     
    #passed_training_targets = np.array([i[1][:min_len] for i in instances_to_drop[0]])[np.array(passing_testing_scores)] 
    #print("PASSED TRAINING FEATURES", passed_training_features)
    #print("PASSED TESTING FEATURES", passed_testing_features)
    #print("PASSED TESTING TARGETS", passed_testing_targets)
    #print("RELEVANT WEIGHTS PER INSTANCE 0", relevant_weights_per_instance[0])
    #print("RELEVANT WEIGHTS PER INSTANCE 1", relevant_weights_per_instance[1])
    np.save("MOST IMPORTANT FEATURE.npy", most_important_feature)
    remaining_features_train_idxs = [passed_training_features[i] for i in relevant_weights_per_instance[0]]
    remaining_features_test_idxs = [passed_testing_features[i] for i in relevant_weights_per_instance[1]]
    remaining_targets_train_idxs = [passed_training_targets[i] for i in relevant_weights_per_instance[0]]
    remaining_targets_test_idxs = [passed_testing_targets[i] for i in relevant_weights_per_instance[1]]
    remaining_weights_arguments.append([remaining_features_train_idxs, remaining_targets_train_idxs])
    remaining_weights_arguments.append([remaining_features_test_idxs, remaining_targets_test_idxs])
    #if not os.path.exists("REMAINING WEIGHTS ARGUMENTS"):
    #    np.save("REMAINING WEIGHTS ARGUMENTS", remaining_weights_arguments)
    #else:
    #    remaining_weights_arguments = np.load("REMAINING WEIGHTS ARGUMENTS", remaining_weights_arguments)
    #Dropout of unimportant values   
    droppedout_context = []
    attention_scores_txt = 'DROPPEDOUT SCORES INSTANCE '
    targets_txt = 'TARGETS INSTANCE '
    context_txt = 'CONTEXT INSTANCE '
    weights_txt = 'WEIGHTS INSTANCE '
    for remaining in range(len(remaining_weights_arguments)):
        for instance in range(len(remaining_weights_arguments[remaining][0])):
            try:
                x = remaining_weights_arguments[remaining][0][instance]
                y = remaining_weights_arguments[remaining][1][instance]
                weights = droppedout_sets_per_instance[remaining-1][instance]
                weights_droppedout = np.copy(weights)
                if not os.path.exists(attention_scores_txt + str(remaining) + ' DISCRIMINATED SET '+ str(instance) + '.npy'):
                    if len(y) == 2:
                        continue
                    #print("X", np.shape(x))
                    #print("Y", len(y))
                    #print("WEIGHTS", np.shape(weights))
                    min_len = min(len(weights), len(x))
                    attention_function, targets, attention_scores, weights, context = asyncio.run(parallel([1],1, (np.array(x)[:min_len], np.array(y)[:min_len], weights[:min_len], 250, True), func=continuous_multi_lorax, shared=False, index=False))[0][0]
                    np.save(attention_scores_txt + str(remaining) + ' DISCRIMINATED SET '+ str(instance) + '.npy', attention_scores)
                    np.save(targets_txt + str(remaining) + ' DISCRIMINATED SET '+ str(instance) + '.npy', targets)
                    np.save(context_txt + str(remaining) + ' DISCRIMINATED SET '+ str(instance) + '.npy', context)
                    np.save(weights_txt + str(remaining) + ' DISCRIMINATED SET '+ str(instance) + '.npy', weights)
                else:
                    attention_scores = np.load(attention_scores_txt + str(remaining) + ' DISCRIMINATED SET '+ str(instance) + '.npy')
                    targets = np.load(targets_txt + str(remaining) + ' DISCRIMINATED SET '+ str(instance) + '.npy')
                    context = np.load(context_txt + str(remaining) + ' DISCRIMINATED SET '+ str(instance) + '.npy')
                    weights = np.load(weights_txt + str(remaining) + ' DISCRIMINATED SET '+ str(instance) + '.npy')
                min_len = min(len(weights), len(targets))
                min_len = min(len(context), min_len)
                mse = mean_squared_error(np.array(np.float64(targets))[:min_len].reshape(min_len,-1)/np.array(np.float64(targets))[:min_len].max(), np.array(np.float64(y[:min_len].reshape(min_len,-1))))
                if mse > 0.2:
                    continue
                print("mse with dropout:", mse)
                print("TARGETS SIZE", np.shape(targets))
                print("TARGETS SIZE", np.shape(y))
                decoded_robustness = validate_non_discrimination_and_robustness(targets[:min_len], context[:min_len], attention_scores[:min_len], y[:min_len], weights[:min_len], mse, 1e-4, 0)
                print("Robustness:", decoded_robustness, 'size', len(x))
                droppedout_context.append([x, y, targets, weights, len(x)])
            except Exception as e:
                logger.exception(e)
                continue
    return droppedout_context


async def explain(model, x, ys, limit, Idxs, logical=None, plot=None, fig=None, thresh=None, attention=False, diffs=None):
    """
    Since the number of relevant features in a dataset must be less than the number of  targets,
    this method follows given criteria to split the dataset into subsets to predict accuracy 
    the hypothetic way (for a expected feature criteria accuracies and relevant  features must be given
    without using the complete dataset)
    :param model (type(model) == object): a usable model
    :param x (type(x) == np.darray): dataset that is going to be explained
    :param ys (type(ys) == np.darray): related targets
    :param limit (type(limit) == list): if a list of strings is given only 'mean', 'var' or 'std' are supported,
        it also supports other types. This argument sets the intersection value (eg.: [x[:,0] > 900, x[:,4] > 70])
    :param Idxs (type(intersects) == list): indexes of important features to perturb
    :param logical (type(logical) == bool or type(logical) == list): bool or a list of bools (must be of the size of limits).
        If True is passed, observability is based on positive limit. If False is passed, observability is based on not
        positive limit. If None is passed, the current limit is not observable and therefore it is contrasted against the 
        explaining values and the next limit is used against the explaining data (limit_i < data < limit_i+1)  	
    :param thresh: a threshold for robust explanation
    :returns:                                                                                                         
      - pos_explanation_scores: accuracies for data following the instances criterias
      - neg_explanation_scores: accuracies for data not following the instances criterias
      - plt: a plot of the explanation
    """
    # 0vsall ->
    # 1vsall ->
    pos_explanation_scores = []
    neg_explanation_scores = []
    if attention == False:
        truek = model.kernel1
        model.kernel1 = 'rbf'
    noisy_x = x.copy()
    idxs = []
    for yi in range(len(limit)):
        if Idxs[yi] == None:
            idxs.append(randint(0,x.shape[1]))
        else:
            idxs.append(Idxs[yi])
    for yi in range(len(limit)):
        if attention == False:
            if ys[0].size > 1:
                targets = model.w[0] * noisy_x + model.bias
            else:
                targets = model.predictions(noisy_x, ys)
                shuffle(noisy_x.T[idxs[yi]].T)
        else:
            min_len = min(len(noisy_x), len(ys))
            min_len = min(len(diffs), min_len)
        intersection = get_intersection(noisy_x[:, idxs[yi]], limit[yi])    
        try:
            if logical != None and logical != False:
                if len(logical) != len(limit):
                    raise ValueError('Missing logic')
                if logical[yi] != True:
                    if not ys[0].size > 1:
                        pos_explanation_scores.append(acc_score(ys[np.where(np.logical_not(
                            noisy_x[:, idxs[yi]] > intersection))], targets[np.where(np.logical_not(noisy_x[:, idxs[yi]] > intersection))]))
                        neg_explanation_scores.append(acc_score(ys[np.where(np.logical_not(
                            noisy_x[:, idxs[yi]] < intersection))], targets[np.where(np.logical_not(noisy_x[:, idxs[yi]] < intersection))]))
                    else:
                        pos_mse = mean_squared_error(np.float64(ys[np.where(np.logical_not(
                            noisy_x[:, idxs[yi]] > intersection))]).reshape(min_len,-1), np.float64(targets[np.where(np.logical_not(noisy_x[:, idxs[yi]] > intersection))]).reshape(min_len,-1))
                        neg_mse = mean_squared_error(np.float64(ys[np.where(np.logical_not(
                            noisy_x[:, idxs[yi]] < intersection))]).reshape(min_len,-1), np.float64(targets[np.where(np.logical_not(noisy_x[:, idxs[yi]] < intersection))]).reshape(min_len,-1))
                        pos_explanation_scores.append(pos_mse)
                        neg_explanation_scores.append(neg_mse)
                    # data drift
                    noisy_x = noisy_x[np.where(np.logical_not(
                        noisy_x[:, idxs[yi]] > intersection))]
                    # concept drift
                    ys = ys[np.where(np.logical_not(
                        noisy_x[:, idxs[yi]] > intersection))]
                    pos_idxs = np.where(noisy_x[:, idxs[yi]] > intersection)
                    neg_idxs = np.where(noisy_x[:, idxs[yi]] < intersection)
                else:
                    print(('WHERE', np.where(noisy_x[:min_len][:, idxs[yi]] > intersection)))
                    print('INTERSECTION', intersection)
                    print('IDXS',idxs)
                    if attention == False:
                        if not ys[0].size > 1:
                            pos_explanation_scores.append(acc_score(ys[np.where(
                                noisy_x[:min_len][:, idxs[yi]] > intersection)[0]], targets[np.where(noisy_x[:min_len][:, idxs[yi]] > intersection)[0]]))
                            neg_explanation_scores.append(acc_score(ys[np.where(
                                noisy_x[:min_len][:, idxs[yi]] < intersection)[0]], targets[np.where(noisy_x[:min_len][:, idxs[yi]] < intersection)[0]]))
                        else:
                            pos_mse = mean_squared_error(ys[np.where(
                                noisy_x[:min_len][:, idxs[yi]] > intersection)[0]], targets[np.where(noisy_x[:min_len][:, idxs[yi]] > intersection)[0]])
                            neg_mse = mean_squared_error(ys[np.where(
                                noisy_x[:min_len][:, idxs[yi]] < intersection)[0]], targets[np.where(noisy_x[:min_len][:, idxs[yi]] < intersection)[0]])
                            pos_explanation_scores.append(pos_mse)
                            neg_explanation_scores.append(neg_mse)
                    else:
                        if not ys[0].size > 1:
                            pos_explanation_scores.append(acc_score(ys[np.where(
                                noisy_x[:min_len][:, idxs[yi]] > intersection)[0]], targets[np.where(noisy_x[:min_len][:, idxs[yi]] > intersection)[0]]))
                            neg_explanation_scores.append(acc_score(ys[np.where(
                                noisy_x[:min_len][:, idxs[yi]] < intersection)[0]], targets[np.where(noisy_x[:min_len][:, idxs[yi]] < intersection)[0]]))
                        else:
                            try:
                                pos_mse = mean_squared_error(ys[np.where(noisy_x[:min_len][:, idxs[yi]] > intersection)].reshape((len(targets),-1)), targets[np.where(targets > intersection)].reshape((len(targets),-1)))
                            except Exception as e:
                                logger.exception(e)
                                pos_mse = 1
                            try:
                                neg_mse = mean_squared_error(ys[np.where(noisy_x[:min_len][:, idxs[yi]] < intersection)].reshape((len(targets),-1)), targets[np.where(targets < intersection)].reshape((len(targets),-1)))
                            except Exception as e:
                                logger.exception(e)
                                neg_mse = 1
                            pos_explanation_scores.append(pos_mse)
                            neg_explanation_scores.append(neg_mse)
                            if (pos_mse == 0) and (neg_mse == 0):
                                #logger.exception(e)
                                diffs[:] = 0
                    # data drift
                    #noisy_x = noisy_x[np.where(
                    #    noisy_x[:, idxs[yi]] > intersection)[0]]
                    # concept drift
                    print("YS", np.shape(targets))
                    pos_classes = targets[np.where(targets > intersection)]
                    print("POS CLASSES", pos_classes)
                    neg_classes = targets[np.where(np.logical_not(targets > intersection))]
                    pos_idxs = np.where(noisy_x[:min_len][:, idxs[yi]] > intersection)
                    neg_idxs = np.where(noisy_x[:min_len][:, idxs[yi]] < intersection)
            else:
                if not ys[0].size > 1:
                    pos_explanation_scores.append(acc_score(ys[np.where(
                        noisy_x[:, idxs[yi]] > intersection)], targets[np.where(noisy_x[:, idxs[yi]] > intersection)]))
                    neg_explanation_scores.append(acc_score(ys[np.where(
                        noisy_x[:, idxs[yi]] < intersection)], targets[np.where(noisy_x[:, idxs[yi]] < intersection)]))
                else:
                    pos_mse = np.mean(diffs[np.where(noisy_x[:, idxs[yi]] > intersection)]**2)
                    neg_mse = np.mean(diffs[np.where(noisy_x[:, idxs[yi]] < intersection)]**2)
                    pos_explanation_scores.append(pos_mse)
                    neg_explanation_scores.append(neg_mse) 
                # data drift
                noisy_x = noisy_x[np.where(
                    noisy_x[:, idxs[yi]] > intersection)]
                # concept drift
                pos_importances = diffs[np.where(noisy_x[:, idxs[yi]] > intersection)]
                neg_importances = diffs[np.where(np.logical_not(noisy_x[:, idxs[yi]] > intersection))]
                pos_idxs = np.where(noisy_x[:, idxs[yi]] > intersection)
                neg_idxs = np.where(noisy_x[:, idxs[yi]] < intersection)
            if plot is True:
                plt = plot_regression(model, noisy_x, pos_classes, idxs)
            else:
                plt = None
            if fig != None:
                plt = plot_regression(model, noisy_x, pos_classes, idxs)
                plt.savefig(fig)
            try:
                plt
            except Exception as e:
                plt = None
        except Exception as e:
            print('Explanation Error:', logger.exception(e))
    if attention == False:
        model.kernel1 = truek
    # add code for effect size or feature transformation
    # given expected criteria, return accuracies per instance
    if np.any(diffs) == None:
        return pos_explanation_scores, neg_explanation_scores, plt
    else:
        if np.any(diffs) == None:
                return pos_explanation_scores, neg_explanation_scores, None, pos_idxs, neg_idxs
        else:
                return pos_explanation_scores, neg_explanation_scores, None, pos_idxs, neg_idxs, pos_classes, neg_classes, diffs
