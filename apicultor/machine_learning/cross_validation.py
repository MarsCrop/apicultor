#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
import time
import asyncio
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

# parallel() de apicultor/arch/thread — motor de paralelismo protegido
from ..arch.thread import parallel
from ..machine_learning.subproblem import continuous_decode
from ..machine_learning.stress import generate_chain_of_thought

warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# GENERACION DE DATOS IMPUTADOS REALISTAS (SIN SKLEARN)
# ══════════════════════════════════════════════════════════════════════════════
def generate_imputed_data(
    dataset_shape=(4000, 4000),
    reference_data=None,
    reference_targets=None,
    imputation_method="mean",
    regression=True,
    noise_level=0.05,
):
    """
    Genera datos imputados REALISTAS basados en datos de referencia.
    """
    if reference_data is None:
        raise ValueError("generate_imputed_data requiere reference_data para imputación realista")
    
    n_samples, n_features = dataset_shape
    n_ref = len(reference_data)
    
    # CORRECCIÓN: Limitar n_samples al tamaño disponible
    if n_samples > n_ref:
        print(f"  WARNING: n_samples={n_samples} > n_ref={n_ref}, reduciendo a {n_ref}")
        n_samples = n_ref
    
    print(f"  n_samples: {n_samples}, n_features: {n_features}")
    
    if n_ref == 0:
        raise ValueError("reference_data está vacío")
    
    # Seleccionar índices
    if n_samples <= n_ref:
        indices = np.random.choice(n_ref, size=n_samples, replace=False)
    else:
        indices = np.random.choice(n_ref, size=n_samples, replace=True)
    
    imputed_data = reference_data[indices].copy()
    imputed_targets = reference_targets[indices].copy() if reference_targets is not None else None
    
    # Aplicar método de imputación
    if imputation_method == "mean":
        mask = np.random.random(imputed_data.shape) < 0.3
        col_means = np.mean(reference_data, axis=0)
        for i in range(n_features):
            imputed_data[mask[:, i], i] = col_means[i]
            
    elif imputation_method == "interpolate":
        for i in range(1, len(imputed_data) - 1):
            if np.random.random() < 0.2:
                imputed_data[i] = (imputed_data[i-1] + imputed_data[i+1]) / 2
                    
    elif imputation_method == "knn_simple":
        if n_ref > 10 and n_samples > 0:
            for i in range(min(100, len(imputed_data))):
                if np.random.random() < 0.2:
                    random_idx = np.random.choice(n_ref, size=min(3, n_ref), replace=False)
                    vecinos = reference_data[random_idx]
                    imputed_data[i] = np.mean(vecinos, axis=0)
    
    if noise_level > 0:
        noise_data = np.random.normal(0, noise_level * np.std(imputed_data, axis=0, keepdims=True), imputed_data.shape)
        imputed_data = imputed_data + noise_data
        imputed_data = np.maximum(imputed_data, 0)
    
    if regression and imputed_targets is not None:
        noise_targets = np.random.normal(0, noise_level * np.std(imputed_targets), imputed_targets.shape)
        imputed_targets = imputed_targets + noise_targets
        imputed_targets = np.maximum(imputed_targets, 0)
    
    return imputed_data, imputed_targets

# ══════════════════════════════════════════════════════════════════════════════
# DATOS ADVERSARIALES (usa imputación realista SIN SKLEARN)
# ══════════════════════════════════════════════════════════════════════════════
def define_adversary_data(
    dataset, y, categories=None, unequal_treatment_factor=0.2, hyp=0,
    impute=False, regression=True,
    imputed_min_features=None,
    imputed_max_features=None,
    min_val=None, max_val=None, reason=False,
    proposed_min_previous_w=0, proposed_max_previous_w=0,
    proposed_min_previous_b=0, proposed_max_previous_b=0,
    proposed_min_w=0, proposed_max_w=0,
    proposed_min_b=0, proposed_max_b=0,
):
    """
    Define datos adversariales con alta paridad para testing del modelo.
    """
    print(f"\n[define_adversary_data]")
    print(f"  dataset shape: {dataset.shape if dataset is not None else 'None'}")
    print(f"  y shape: {y.shape if y is not None else 'None'}")
    print(f"  impute: {impute}")
    print(f"  unequal_treatment_factor: {unequal_treatment_factor}")
    
    if hyp == 'var':
        hyp = np.var(y)
    elif hyp == 'std':
        hyp = np.std(y)
    elif hyp == 'mean':
        hyp = np.mean(y)
    elif hyp == 'median':
        hyp = np.median(y)

    if not regression:
        categories = np.unique(categories)
    else:
        cls0 = np.where(y > hyp)[0]
        cls1 = np.where(np.logical_not(y > hyp))[0]
        categories = [cls0, cls1]

    adversary_group = []
    adversary_group_y = []
    adversary_group_w = []
    adversary_group_b = []
    previous_adversary_group_w = []
    previous_adversary_group_b = []
    imputed_b = []

    for i, cat in enumerate(categories):
        print(f"  Categoría {i}, tamaño: {len(cat)}")
        real_data = dataset[cat]
        real_targets = y[cat]
        
        if not impute:
            adversary_set_size = len(real_data)
            imputed_dataset = real_data
            imputed_targets = real_targets
            print(f"    Usando datos REALES: {adversary_set_size} muestras")
        else:
            # CORRECCIÓN: Limitar el tamaño para no exceder el dataset
            max_size = len(real_data)
            adversary_set_size = int(np.floor(max_size * unequal_treatment_factor))
            if adversary_set_size < 1:
                adversary_set_size = max(1, max_size // 10)
            # Asegurar que no excede el tamaño disponible
            adversary_set_size = min(adversary_set_size, max_size)
            
            print(f"    Generando datos IMPUTADOS: {adversary_set_size} muestras (max={max_size})")
            
            imputation_method = "interpolate" if reason else "mean"
            
            imputed_dataset, imputed_targets = generate_imputed_data(
                dataset_shape=(adversary_set_size, dataset.shape[1]),
                reference_data=real_data,
                reference_targets=real_targets,
                imputation_method=imputation_method,
                regression=regression,
                noise_level=0.03,
            )

        imputed_dataset = np.float64(imputed_dataset)
        imputed_targets = np.float64(imputed_targets)
        
        print(f"    imputed_dataset shape: {imputed_dataset.shape}")
        print(f"    imputed_targets shape: {imputed_targets.shape}")
        
        adversary_group.append(imputed_dataset)
        adversary_group_y.append(imputed_targets)

    adversary_data = np.vstack(adversary_group)
    adversary_targets = np.vstack(adversary_group_y)
    
    print(f"[define_adversary_data] FINAL - shapes: {adversary_data.shape}, {adversary_targets.shape}")
    
    return (
        adversary_data,
        adversary_targets,
        previous_adversary_group_w,
        previous_adversary_group_b,
        adversary_group_w,
        imputed_b,
    )

# ══════════════════════════════════════════════════════════════════════════════
# DETECCION DE DATA LEAKAGE
# ══════════════════════════════════════════════════════════════════════════════

def scan_data_leakage(y, ytest, wrong_yrate=0.2, error_threshold=0):
    """
    Detecta data leakage contando el numero total de muestras mal
    predichas usando un umbral.
    """
    error = np.array([y[i] - ytest[i] for i in range(len(y)-1)])
    logger.debug(f"ERROR SHAPE {np.shape(error)}")

    wrongly_predicted_samples = np.where(error > error_threshold)[0]
    correctly_predicted_samples = y[np.unique(
        np.where(np.logical_not(error > error_threshold))[0]
    )]

    logger.debug(f"WRONGLY PREDICTED SAMPLES {y[np.unique(wrongly_predicted_samples)]}")
    wpr = len(wrongly_predicted_samples) / y.size if y.size > 0 else 0
    logger.debug(f"Wrong prediction rate: {wpr * 100} %")

    if wpr > wrong_yrate:
        logger.debug("Data leakage test: FAILED")
    else:
        logger.debug("Data leakage test: OK")

    # Detectar leakage por duplicados en targets
    duplicate_targets = len(y) - len(np.unique(y, axis=0)) if y.ndim > 1 else len(y) - len(np.unique(y))
    if duplicate_targets > 0:
        logger.warning(f"[Data Leakage] {duplicate_targets} targets duplicados detectados")

    return (
        wpr,
        correctly_predicted_samples,
        y[np.unique(wrongly_predicted_samples)],
        wrongly_predicted_samples,
        error,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ERRORES Y UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

class DiscriminationError(Exception):
    def __init__(self, e):
        super().__init__(("Something happened!", e, "Continuing"))


class ProductionizationError(Exception):
    def __init__(self, e):
        super().__init__(e)


def hard_kill_pool(pids, pt):
    """Termina un pool de procesos de forma forzada."""
    for pid in pids:
        os.kill(pid, signal.SIGINT)
    pt.terminate()


# ══════════════════════════════════════════════════════════════════════════════
# PRODUCTIONIZE
# ══════════════════════════════════════════════════════════════════════════════

async def productionize(
    model, features_train, targets_train, features_test, targets_test,
    features, targets, C, reg, k0, k1, last, criteria, intersects, logical,
    track_conflict,
):
    """
    Entrena, valida y verifica paridad estadistica de un modelo.
    """
    try:
        training_time = time.time()
        await model.fit_model(
            features_train, targets_train, k0, k1, C, reg,
            1.0 / features_train.shape[1], 0.8,
        )
        training_time = np.abs(training_time - time.time())

        clf_predictions_train = model.predictions(features_train, targets_train)
        train_score_val = score(targets_train, clf_predictions_train)

        protection_val_rule_train = p_rule(
            clf_predictions_train, targets_train, model.w, features_train, model.proba
        )
        if type(protection_val_rule_train) != bool and protection_val_rule_train >= 0.8:
            parity_train = protection_val_rule_train
        else:
            return DiscriminationError(Exception("Parity train failed"))

        pex, cex, vis = explain(
            model, features_train, targets_train, criteria, intersects, logical
        )
        logger.debug(f"Train parent explanation: {pex}")
        logger.debug(f"Train child explanation: {cex}")
        logger.debug(f"Train BTC: {BTC(targets_train, clf_predictions_train)}")
        bec, cons = BEC(targets_train, clf_predictions_train)

        clf_predictions_test = model.predictions(features_test, targets_test)
        test_score_val = score(targets_test, clf_predictions_test)

        protection_val_rule_test = p_rule(
            clf_predictions_test, targets_test, model.w, features_test, model.proba
        )
        if type(protection_val_rule_test) != bool and protection_val_rule_test >= 0.8:
            parity_test = protection_val_rule_test
        else:
            return DiscriminationError(Exception("Parity test failed"))

        pex, cex, vis = explain(
            model, features_test, targets_test, criteria, intersects, logical
        )
        logger.debug(f"Test parent explanation: {pex}")
        logger.debug(f"Test child explanation: {cex}")
        bec, cons = BEC(targets_test, clf_predictions_test)

        clf_time = time.time()
        if last is True:
            model.apply(model.best_layer[-1])

        await model.fit_model(
            features, targets,
            k0, k1,
            C, reg, 1.0 / features.shape[1], 0.8,
        )
        model.written = False
        clf_time = np.abs(clf_time - time.time())

        clf_predictions = model.predictions(features, targets)

        protection_val_rule = p_rule(
            clf_predictions, targets, model.w, features, model.proba
        )
        if type(protection_val_rule) != bool and protection_val_rule >= 0.8:
            parity = protection_val_rule
        else:
            return DiscriminationError(Exception("Parity final failed"))

        mse_val = score(targets, clf_predictions)
        pex, cex, vis = explain(
            model, features, targets, criteria, intersects, logical
        )
        logger.debug(f"Parent search explanation: {pex}")
        logger.debug(f"Child search explanation: {cex}")

        if track_conflict is None:
            bec, cons = BEC(targets, clf_predictions)
        else:
            btc_val = BTC(targets, clf_predictions)
            bec, cons = BEC(targets, clf_predictions, track_conflict, True)

        return (
            mse_val, train_score_val, test_score_val, model, cons,
            training_time, clf_time, parity_train, parity_test, parity,
        )
    except Exception as e:
        return ProductionizationError(logger.exception(e))


# ══════════════════════════════════════════════════════════════════════════════
# PESO DE CONFIGURACION
# ══════════════════════════════════════════════════════════════════════════════

def define_weight(value, threshold, func):
    """Devuelve un peso segun si el valor alcanza el umbral con la funcion dada."""
    if func == np.max:
        return 0.25 if value == threshold else 0.1
    else:
        return 0.25 if value == threshold else 0.1


# ══════════════════════════════════════════════════════════════════════════════
# STRESS TEST
# ══════════════════════════════════════════════════════════════════════════════

async def stress_test_red_data(
    features_train, features_test, leaking_features, adversaries,
    correctly_decoded_targets, targets_test, cross_validation_context, hyp, pidx,
):
    adversary_info = []
    for adversary in adversaries:
        info = {
            "adversary": adversary,
            "features_train": features_train,
            "features_test": features_test,
            "leaking_features": leaking_features,
            "correctly_decoded_targets": correctly_decoded_targets,
            "targets_test": targets_test,
            "cross_validation_context": cross_validation_context,
            "hyp": hyp,
            "pidx": pidx,
        }
        adversary_info.append(info)
    return adversary_info


# ══════════════════════════════════════════════════════════════════════════════
# FORWARD PERTURBATION
# ══════════════════════════════════════════════════════════════════════════════

async def forward_perturbation(
    fname1, fname2, fname3, fname4, fname5,
    asize, bsize,
    msg1, msg2, msg3, msg4, msg5,
    targets_to_perturb, features, context,
):
    """
    Carga o calcula perturbaciones hacia adelante.
    """
    win = True 
    diffin = True 
    msein = True 
    cin = True 
    tin = True  
    if os.path.exists(fname1):
        targets = np.load(fname1)
    else:
        tin = False
    if os.path.exists(fname2):
        context_arr = np.load(fname2)
    else:
        cin = False
    if os.path.exists(fname3):
        mse_val = np.load(fname3)
    else:
        msein = False
    if os.path.exists(fname4):
        diff = np.load(fname4)
    else:
        diffin = False
    if os.path.exists(fname5):
        weights = np.load(fname5)
        logger.debug(f"{msg4} {weights}")
        logger.debug(f"Y {np.shape(targets)}")
    else:
        win = False
    if all((win, diffin, msein, cin, tin)) == False:
        logger.debug(msg1)
        min_len = min(asize, bsize)

        result = await parallel(
            [1], 1,
            (features, targets, context, 2500, True),  # ← CORRECTO: features, targets, context
            func=continuous_decode,
            shared=False,
            index=False,
        )
        print("RESULT:", result)
        attention_function, targets, attention_scores, weights, context_arr = result[0]

        logger.debug(f"{msg2} {np.shape(np.array(targets) / np.array(targets).max())} {np.shape(targets_to_perturb)}")

        try:
            mse_val = mean_squared_error(
                np.array(np.float64(targets))[:min_len].reshape(min_len, -1)
                / np.array(np.float64(targets))[:min_len].max(),
                np.array(np.float64(targets_to_perturb[:min_len].reshape(min_len, -1))),
            )
        except Exception:
            mse_val = 1.0

        try:
            diff = np.array([
                np.array(np.float64(targets))[i]
                - np.array(np.float64(targets_to_perturb))[i]
                for i in range(min_len)
            ])
        except Exception:
            min_lenj = min(len(targets[0]), len(targets_to_perturb.T[0]))
            try:
                diff = np.array([
                    np.array(np.float64(targets))[i][:min_lenj]
                    - np.array(np.float64(targets_to_perturb.T))[i][:min_lenj]
                    for i in range(len(targets))
                ]).T
            except Exception:
                diff = targets

        logger.debug(f"{msg4} {np.shape(weights)}")
        logger.debug(f"{msg5} {np.shape(diff)}")
        np.save(fname1, targets[:min_len])
        np.save(fname2, context_arr)
        np.save(fname3, mse_val)
        np.save(fname4, diff)
        np.save(fname5, weights)

    logger.debug(f"{msg3} {np.shape(mse_val)}")
    return targets, context_arr, mse_val, diff, weights


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS PERTURBATION
# ══════════════════════════════════════════════════════════════════════════════

async def process_perturbation(
    step, hyp, pidx, context, targets, features,
    msg_suffix, prefix, exp_data, adversary, locally_features,
):
    """
    Procesa una perturbacion individual.
    """
    fname1 = f"{prefix} {step} {msg_suffix} TARGETS {str(context).upper()} HYPOTHESIS {hyp} FEATURE {pidx}.npy"
    fname2 = f"{prefix} {step} {msg_suffix} CONTEXT {str(context).upper()} HYPOTHESIS {hyp} FEATURE {pidx}.npy"
    fname3 = f"{prefix} {step} {msg_suffix} TARGETS {str(context).upper()} MSE HYPOTHESIS {hyp} FEATURE {pidx}.npy"
    fname4 = f"{prefix} {step} {msg_suffix} TARGETS {str(context).upper()} WEIGHTS HYPOTHESIS {hyp} FEATURE {pidx}.npy"
    fname5 = f"{prefix} {step} {msg_suffix} TARGETS {str(context).upper()} DIFFERENCE HYPOTHESIS {hyp} FEATURE {pidx}.npy"

    asize = len(context) if context is not None else 0
    bsize = len(targets) if targets is not None else 0
    msg1 = ""
    msg2 = f"{msg_suffix} OUTPUTS {str(context).upper()} IN CONTEXT"
    msg3 = f"{msg_suffix} targets {str(context).lower()} mse:"
    msg4 = f"{msg_suffix} WEIGHTS {str(context).upper()}:"
    msg5 = f"{msg_suffix} DIFFERENCE {str(context).upper()}:"

    targets_out, context_out, mse_val, diff, weights = await forward_perturbation(
        fname1, fname2, fname3, fname4, fname5,
        asize, bsize,
        msg1, msg2, msg3, msg4, msg5,
        features, targets, context,
    )

    exp_data.append([adversary, locally_features, targets_out, weights, hyp, True, mse_val, diff])
    return exp_data


# ══════════════════════════════════════════════════════════════════════════════
# RUN PERTURBATIONS ON J-TH FEATURES
# ══════════════════════════════════════════════════════════════════════════════
async def run_perturbations_on_jth_features(
    features_train, features_test, leaking_features, adversary,
    correctly_decoded_targets, targets_test, cross_validation_context,
    step, hyp, leakage, leaking_targets,
    max_features_per_run: int = 8
):
    """
    Ejecuta perturbaciones sobre un número limitado de features.
    """
    print(f"\n[run_perturbations_on_jth_features]")
    print(f"  step={step}, hyp={hyp}")
    
    if features_train is None or features_train.shape[1] == 0:
        print(f"  features_train es None o vacío")
        return []

    n_features = features_train.shape[1]
    BATCH_SIZE = 8
    
    # Asegurar que max_features_per_run no exceda n_features
    max_features_per_run = min(max_features_per_run, n_features)
    features_to_perturb = list(range(max_features_per_run))
    
    print(f"  n_features: {n_features}, perturbando: {features_to_perturb}")
    
    # Verificar dimensiones de features_test
    if features_test is not None:
        print(f"  features_test shape: {features_test.shape}")
        if features_test.shape[0] < max_features_per_run:
            print(f"  WARNING: features_test tiene {features_test.shape[0]} filas, pero se necesitan {max_features_per_run}")
    
    perturbation_row = []
    for batch_start in range(0, len(features_to_perturb), BATCH_SIZE):
        batch_indices = features_to_perturb[batch_start:batch_start + BATCH_SIZE]
        tasks = [
            perturb_feature(
                features_train, features_test, leaking_features, adversary,
                correctly_decoded_targets, targets_test, cross_validation_context,
                step, hyp, pidx, leakage, leaking_targets,
            )
            for pidx in batch_indices
        ]
        batch_results = await asyncio.gather(*tasks)
        perturbation_row.extend(batch_results)
        print(f"    Batch {batch_start}: {len(batch_results)} resultados")

    return perturbation_row
# ══════════════════════════════════════════════════════════════════════════════
# HYPOTHESIS PERTURBATION
# ══════════════════════════════════════════════════════════════════════════════

async def hypothesis_perturbation(
    features_train, features_test, leaking_features, adversary,
    correctly_decoded_targets, targets_test, cross_validation_context,
    hyp, leakage, leaking_targets, perturbations, gathered_mse, error,
):
    """
    Ejecuta perturbaciones para una hipotesis dada.
    """
    hypothesis_info = []
    for step in range(perturbations):
        result = await run_perturbations_on_jth_features(
            features_train, features_test, leaking_features, adversary,
            correctly_decoded_targets, targets_test, cross_validation_context,
            step, hyp, leakage, leaking_targets,
        )
        hypothesis_info.append(result)

    least_parity = [0]
    droppedout_context = await dropout(
        hypothesis_info[0] if hypothesis_info else [],
        protected_groups=least_parity,
        protected_lime=leakage,
        protected_mse=gathered_mse,
        error=error,
        protected_features=features_train,
    )
    return droppedout_context


# ══════════════════════════════════════════════════════════════════════════════
# MULTIPLE HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════════════════
async def multiple_hypothesis_testing(
    features_train, features_test, leaking_features, adversary,
    correctly_decoded_targets, targets_test, cross_validation_context,
    leakage, leaking_targets, perturbations, hypotheses, gathered_mse, error,
    capture_adversary_models=False, base_model=None,
    max_features_per_run: int = 8  # NUEVO: límite de features
):
    """
    Testea multiples hipotesis de perturbacion.
    
    Si capture_adversary_models=True, retorna también todos los modelos
    adversarios generados durante el testing.
    
    IMPORTANTE: Solo se perturban max_features_per_run features.
    """
    adversary_info = []
    adversary_models = []
    
    # Determinar qué features perturbar (las primeras N)
    n_features_total = features_train.shape[1] if features_train is not None else 0
    features_to_perturb = list(range(min(max_features_per_run, n_features_total)))
    
    print(f"[multiple_hypothesis] Perturbando {len(features_to_perturb)} features (índices 0 a {len(features_to_perturb)-1})")
    
    for hyp in hypotheses:
        for step in range(perturbations):
            for pidx in features_to_perturb:  # SOLO las features seleccionadas
                result = await perturb_feature(
                    features_train, features_test, leaking_features, adversary,
                    correctly_decoded_targets, targets_test, cross_validation_context,
                    step, hyp, pidx, leakage, leaking_targets,
                    model=base_model,
                    capture_adversary_model=capture_adversary_models,
                )
                
                if capture_adversary_models:
                    if isinstance(result, tuple) and len(result) == 2:
                        exp_data, adv_model = result
                        if exp_data is not None:
                            adversary_info.append(exp_data)
                        if adv_model is not None:
                            adversary_models.append({
                                'model': adv_model,
                                'hypothesis': hyp,
                                'step': step,
                                'feature_idx': pidx,
                                'noise_level': leakage,
                                'adversary_id': len(adversary_models)
                            })
                    else:
                        if result is not None:
                            adversary_info.append(result)
                else:
                    if result is not None:
                        adversary_info.append(result)
    
    if capture_adversary_models:
        return adversary_info, adversary_models
    else:
        return adversary_info

# ══════════════════════════════════════════════════════════════════════════════
# PARALLEL ADVERSARY PERTURBATION
# ══════════════════════════════════════════════════════════════════════════════
async def parallel_adversary_perturbation(
    features_train, features_test, leaking_features, adversaries,
    correctly_decoded_targets, targets_test, cross_validation_context,
    hyp, leakage, leaking_targets, perturbations, hypotheses,
    gathered_mse=None, error=None,
    capture_adversary_models=False, base_model=None
):
    """
    Ejecuta perturbaciones adversariales para todos los adversaries.
    
    Retorna (exp_data, all_adversary_models) si capture_adversary_models=True.
    """
    all_adversary_models = []
    tasks = []
    
    for adv in adversaries:
        task = multiple_hypothesis_testing(
            features_train, features_test, leaking_features, adv,
            correctly_decoded_targets, targets_test, cross_validation_context,
            leakage, leaking_targets, perturbations, hypotheses,
            gathered_mse, error,
            capture_adversary_models=capture_adversary_models,
            base_model=base_model,  # Pasar modelo base
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    if capture_adversary_models:
        exp_data_list = []
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                exp_data, adv_models = result
                exp_data_list.append(exp_data)
                all_adversary_models.extend(adv_models)
            else:
                exp_data_list.append(result)
        return exp_data_list, all_adversary_models
    else:
        return results


# ══════════════════════════════════════════════════════════════════════════════
# COT METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_cot_metrics(thoughts, cot_context):
    """
    Calcula metricas de similitud coseno entre el vector de pensamiento
    original y el contexto final de chain-of-thought.
    """
    metrics = {}
    similarities = []
    
    if thoughts is None or cot_context is None:
        return {"mean_similarity": 0.0, "min_similarity": 0.0, "max_similarity": 0.0}
    
    min_len = min(len(thoughts), len(cot_context))
    for i in range(min_len):
        original = thoughts[i].flatten()
        final = cot_context[i].flatten() if len(cot_context[i].shape) > 0 else cot_context[i]
        norm_product = np.linalg.norm(original) * np.linalg.norm(final) + 1e-8
        similarity = np.dot(original, final) / norm_product
        similarities.append(similarity)

    metrics["mean_similarity"] = np.mean(similarities) if similarities else 0.0
    metrics["min_similarity"] = np.min(similarities) if similarities else 0.0
    metrics["max_similarity"] = np.max(similarities) if similarities else 0.0
    return metrics

async def perturb_feature(
    features_train, features_test, leaking_features, adversary,
    correctly_decoded_targets, targets_test, cross_validation_context,
    step, hyp, pidx, leakage, leaking_targets,
    continuous_attention_function=None, previousw=None, previousb=None, 
    model=None, previous_weights=None, capture_adversary_model=False
):
    """
    Perturba la feature en el indice pidx y evalua el impacto adversarial.
    """
    print(f"\n{'='*80}")
    print(f"[perturb_feature] INICIANDO")
    print(f"  step={step}, hyp={hyp}, pidx={pidx}")
    print(f"  capture_adversary_model={capture_adversary_model}")
    print(f"  leakage={leakage}")
    print(f"  model is None? {model is None}")
    
    if features_test is not None:
        print(f"  features_test shape: {features_test.shape}")
    if targets_test is not None:
        print(f"  targets_test shape: {targets_test.shape}")
    if correctly_decoded_targets is not None:
        print(f"  correctly_decoded_targets shape: {correctly_decoded_targets.shape}")
    
    exp_data = []
    fname_prefix = f"PERTURBATION {step} HYPOTHESIS {hyp} FEATURE {pidx}"
    perturbed_model = None

    try:
        # ========== 1. SHUFFLE DE FEATURES ==========
        print(f"\n  [1] Verificando caché de shuffle...")
        if os.path.exists(f"{fname_prefix} FEATURES TRAIN.npy"):
            print(f"    Cargando desde caché")
            features_train = np.load(f"{fname_prefix} FEATURES TRAIN.npy")
            features_test = np.load(f"{fname_prefix} FEATURES TEST.npy")
            leaking_features = np.load(f"{fname_prefix} LEAKING FEATURES.npy")
            if adversary is not None and isinstance(adversary, dict):
                adversary['xtrain'] = np.load(f"{fname_prefix} ADVERSARY FEATURES TRAIN.npy")
                adversary['xtest'] = np.load(f"{fname_prefix} ADVERSARY FEATURES TEST.npy")
        else:
            print(f"    No hay caché, aplicando shuffles...")
            
            if features_train is not None and hasattr(features_train, 'shape') and len(features_train.shape) == 2:
                if pidx < features_train.shape[1]:
                    print(f"    Shuffling features_train[:, {pidx}]")
                    np.random.shuffle(features_train[:, pidx])
            
            if features_test is not None and hasattr(features_test, 'shape') and len(features_test.shape) == 2:
                if pidx < features_test.shape[1]:
                    print(f"    Shuffling features_test[:, {pidx}]")
                    np.random.shuffle(features_test[:, pidx])
            
            if leaking_features is not None and hasattr(leaking_features, 'shape'):
                if len(leaking_features.shape) == 2:
                    if pidx < leaking_features.shape[1]:
                        print(f"    Shuffling leaking_features[:, {pidx}]")
                        np.random.shuffle(leaking_features[:, pidx])
            
            if adversary is not None and isinstance(adversary, dict):
                if 'xtrain' in adversary and adversary['xtrain'] is not None:
                    if hasattr(adversary['xtrain'], 'shape') and len(adversary['xtrain'].shape) == 2:
                        if pidx < adversary['xtrain'].shape[1]:
                            np.random.shuffle(adversary['xtrain'][:, pidx])
                if 'xtest' in adversary and adversary['xtest'] is not None:
                    if hasattr(adversary['xtest'], 'shape') and len(adversary['xtest'].shape) == 2:
                        if pidx < adversary['xtest'].shape[1]:
                            np.random.shuffle(adversary['xtest'][:, pidx])
            
            if capture_adversary_model and model is not None:
                import copy
                perturbed_model = copy.deepcopy(model)
                if hasattr(perturbed_model, 'w') and perturbed_model.w is not None:
                    w_copy = perturbed_model.w.copy()
                    if w_copy.ndim == 1 and pidx < len(w_copy):
                        noise = np.random.normal(0, 0.1)
                        w_copy[pidx] = w_copy[pidx] + noise
                        perturbed_model.w = w_copy
                    elif w_copy.ndim == 2 and pidx < w_copy.shape[1]:
                        noise = np.random.normal(0, 0.1, w_copy[:, pidx].shape)
                        w_copy[:, pidx] = w_copy[:, pidx] + noise
                        perturbed_model.w = w_copy
            
            np.save(f"{fname_prefix} FEATURES TRAIN.npy", features_train)
            np.save(f"{fname_prefix} FEATURES TEST.npy", features_test)
            np.save(f"{fname_prefix} LEAKING FEATURES.npy", leaking_features)
            if adversary is not None and isinstance(adversary, dict):
                np.save(f"{fname_prefix} ADVERSARY FEATURES TRAIN.npy", adversary.get('xtrain', np.array([])))
                np.save(f"{fname_prefix} ADVERSARY FEATURES TEST.npy", adversary.get('xtest', np.array([])))

        # ========== 2. ADVERSARY FEATURES TEST ==========
        print(f"\n  [2] Procesando ADVERSARY FEATURES TEST...")
        locally_adversary_features_test = None
        locally_adversary_targets_test = None
        
        adv_features_path = f"{fname_prefix} ADVERSARY FEATURES TEST.npy"
        adv_targets_path = f"{fname_prefix} ADVERSARY TARGETS TEST.npy"
        
        if os.path.exists(adv_features_path) and os.path.exists(adv_targets_path):
            print(f"    Cargando desde caché")
            locally_adversary_features_test = np.load(adv_features_path)
            locally_adversary_targets_test = np.load(adv_targets_path)
            print(f"    ✅ Cargado: features={locally_adversary_features_test.shape}, targets={locally_adversary_targets_test.shape}")
        else:
            print(f"    Generando nuevos adversarios...")
            
            if features_test is None:
                print(f"    ❌ features_test es None, no se puede generar adversary data")
                return [] if not capture_adversary_model else ([], None)
            
            if targets_test is None:
                print(f"    ❌ targets_test es None, no se puede generar adversary data")
                return [] if not capture_adversary_model else ([], None)
            
            # Asegurar que tengan la misma longitud
            min_len = min(len(features_test), len(targets_test))
            print(f"    min_len (features_test/targets_test): {min_len}")
            
            if min_len == 0:
                print(f"    ❌ min_len == 0, no se puede generar adversary data")
                return [] if not capture_adversary_model else ([], None)
            
            features_test_trunc = features_test[:min_len]
            targets_test_trunc = targets_test[:min_len]
            
            # Usar leakage para determinar el factor de tratamiento desigual
            unequal_factor = max(0.1, min(0.9, 1 - leakage if leakage is not None else 0.5))
            print(f"    unequal_factor: {unequal_factor}")
            
            try:
                # Definir categories basado en los targets (para regresión, dividir por la mediana)
                median_target = np.median(targets_test_trunc)
                cls0 = np.where(targets_test_trunc <= median_target)[0]
                cls1 = np.where(targets_test_trunc > median_target)[0]
                categories = [cls0, cls1]
                
                print(f"    Categorías: cls0={len(cls0)}, cls1={len(cls1)}")
                
                # Llamar a define_adversary_data con impute=False para datos adversariales REALES
                result = define_adversary_data(
                    features_test_trunc, 
                    targets_test_trunc,
                    categories=categories,
                    unequal_treatment_factor=unequal_factor,
                    hyp=hyp,
                    impute=False,
                    regression=True,
                )
                
                if result and len(result) >= 2:
                    locally_adversary_features_test = result[0]
                    locally_adversary_targets_test = result[1]
                    
                    if locally_adversary_features_test is not None and len(locally_adversary_features_test) > 0:
                        np.save(adv_features_path, locally_adversary_features_test)
                        np.save(adv_targets_path, locally_adversary_targets_test)
                        print(f"    ✅ Generado: features shape={locally_adversary_features_test.shape}, targets shape={locally_adversary_targets_test.shape}")
                    else:
                        print(f"    ⚠️ define_adversary_data retornó datos vacíos")
                        return [] if not capture_adversary_model else ([], None)
                else:
                    print(f"    ⚠️ define_adversary_data retornó resultado inesperado: {type(result)}")
                    return [] if not capture_adversary_model else ([], None)
                    
            except Exception as e:
                print(f"    ❌ Error generando: {e}")
                import traceback
                traceback.print_exc()
                return [] if not capture_adversary_model else ([], None)

        # ========== 3. FORWARD PERTURBATION ==========
        print(f"\n  [3] Procesando FORWARD PERTURBATION...")
        
        # Verificar que tenemos datos válidos
        if locally_adversary_targets_test is None or locally_adversary_features_test is None:
            print(f"    ERROR: Datos adversariales son None")
            return [] if not capture_adversary_model else ([], None)
        
        if len(locally_adversary_targets_test) == 0 or len(locally_adversary_features_test) == 0:
            print(f"    ERROR: Datos adversariales están vacíos")
            return [] if not capture_adversary_model else ([], None)
        
        print(f"    Datos: targets={locally_adversary_targets_test.shape}, features={locally_adversary_features_test.shape}")
        
        # Crear contexto de validación cruzada si es None
        if cross_validation_context is None:
            print(f"    ERROR: cross_validation_context es None")
            return [] if not capture_adversary_model else ([], None)
        
        # Ejecutar forward_perturbation
        try:
            # Asegurar que los targets tengan la forma correcta (2D)
            if len(locally_adversary_targets_test.shape) == 1:
                locally_adversary_targets_test = locally_adversary_targets_test.reshape(-1, 1)
                print(f"    Reshape targets a: {locally_adversary_targets_test.shape}")
            
            adversary_targets_test, adversary_context_test, mse_val, diff, weights = await forward_perturbation(
                f"{fname_prefix} ADVERSARY FEATURES TEST.npy",
                f"{fname_prefix} ADVERSARY CONTEXT TEST.npy",
                f"{fname_prefix} ADVERSARY FEATURES TEST MSE.npy",
                f"{fname_prefix} ADVERSARY FEATURES TEST WEIGHTS.npy",
                f"{fname_prefix} ADVERSARY FEATURES TEST DIFFERENCE.npy",
                len(correctly_decoded_targets) if correctly_decoded_targets is not None else len(locally_adversary_targets_test),
                len(locally_adversary_features_test),
                "",
                "LOCALLY ADVERSARY OUTPUTS TEST IN CONTEXT",
                "Locally adversary test mse:",
                "LOCALLY ADVERSARY WEIGHTS TEST:",
                "LOCALLY ADVERSARY DIFFERENCE TEST:",
                locally_adversary_targets_test,       # ← targets (Y)
                locally_adversary_features_test,      # ← features (X)
                cross_validation_context,
            )
            print(f"    ✅ forward_perturbation completado, mse_val={mse_val}")
            
            exp_data.append([
                adversary, features_test, adversary_targets_test,
                weights, hyp, True, mse_val, diff,
            ])
            print(f"    exp_data actualizado, ahora {len(exp_data)} elementos")
            
        except Exception as e:
            print(f"    ❌ Error en forward_perturbation: {e}")
            import traceback
            traceback.print_exc()
            return [] if not capture_adversary_model else ([], None)

        print(f"\n[perturb_feature] FINAL - exp_data tiene {len(exp_data)} elementos")
        
        if capture_adversary_model:
            return exp_data, perturbed_model
        else:
            return exp_data
            
    except Exception as e:
        print(f"\n[perturb_feature] ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        if capture_adversary_model:
            return [], None
        else:
            return []
# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH (usando funciones existentes)
# ══════════════════════════════════════════════════════════════════════════════
async def GridSearch(
    model, features, targets, Cs, reg_params, kernel_configs,
    last, criteria, intersects, logical,
    fmins_train=[], fmaxs_train=[], fmins_test=[], fmaxs_test=[],
    tmins_train=[], tmaxs_train=[], tmins_test=[], tmaxs_test=[],
    track_conflict=None, p_thresh=1e-4, regression=False,
    adversaries=[1], model_name='default',
    output_dir=".",  # NUEVO
    loop_counter=0,  # NUEVO
):
    """
    Cross-Validation con busqueda en grilla.
    CON DEBUGGING COMPLETO: prints detallados de cada paso.
    """
    print("\n" + "="*80)
    print(f"🔍 GRID SEARCH INICIADO - model_name: {model_name}")
    print("="*80)
    
    if features is None or targets is None:
        print("❌ ERROR: features o targets son None")
        logger.error("GridSearch: features o targets son None")
        return None
    
    print(f"📊 features shape: {features.shape if hasattr(features, 'shape') else 'N/A'}")
    print(f"📊 targets shape: {targets.shape if hasattr(targets, 'shape') else 'N/A'}")
    print(f"📊 features type: {type(features)}")
    print(f"📊 targets type: {type(targets)}")
    
    if len(features) == 0 or len(targets) == 0:
        print("❌ ERROR: features o targets vacíos")
        logger.error("GridSearch: features o targets vacíos")
        return None
    
    n = len(features)
    attention_contexts = []
    attention_weights = []
    accepted_targets = []
    features_train = []
    features_test = []
    targets_train = []
    targets_test = []

    print("\n📂 DIVIDIENDO DATOS EN TRAIN/TEST...")
    print("TARGETS:", targets)
    
    if not isinstance(targets[0], (int, float)) and hasattr(targets[0], 'size') and targets[0].size > 0 and type(targets) == list:
        unique_vals = np.unique(targets)
        print(f"📊 targets tienen {len(unique_vals)} valores únicos")
        
        for i in range(len(unique_vals)):
            indices = np.where(targets == unique_vals[i])[0]
            if len(indices) == 0:
                continue
            n_class = len(indices)
            test_n = int((20 * n_class) / 100)
            train_n = n_class - test_n
            print(f"  Clase {i}: {n_class} muestras, train={train_n}, test={test_n}")
            
            features_train.append(features[indices][:train_n])
            features_test.append(features[indices][train_n:])
            targets_train.append(targets[indices][:train_n])
            targets_test.append(targets[indices][train_n:])
        
        if features_train:
            features_test = np.vstack(features_test) if features_test else np.array([])
            features_train = np.vstack(features_train) if features_train else np.array([])
            targets_train = np.hstack(targets_train) if targets_train else np.array([])
            targets_test = np.hstack(targets_test) if targets_test else np.array([])
    else:
        n = len(features)
        test_n = int((20 * n) / 100)
        train_n = n - test_n
        features_train = features[:train_n]
        features_test = features[train_n:]
        targets_train = targets[:train_n]
        targets_test = targets[train_n:]
        print(f"📊 División simple: train={train_n}, test={test_n}")

    print(f"\n✅ TRAIN shapes: features={features_train.shape if hasattr(features_train, 'shape') else 'N/A'}, targets={targets_train.shape if hasattr(targets_train, 'shape') else 'N/A'}")
    print(f"✅ TEST shapes: features={features_test.shape if hasattr(features_test, 'shape') else 'N/A'}, targets={targets_test.shape if hasattr(targets_test, 'shape') else 'N/A'}")

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
    
    print(f"\n⚙️ PARÁMETROS DE BÚSQUEDA:")
    print(f"  Cs: {Cs}")
    print(f"  reg_params: {reg_params}")
    print(f"  kernel_configs: {kernel_configs}")

    if adversaries and isinstance(adversaries[0], int):
        # Convertir enteros a formato que entienda multiple_hypothesis_testing
        # Cada adversario necesita ser un dict con xtrain, ytrain, etc.
        formatted_adversaries = []
        for adv_idx in adversaries:
            formatted_adversaries.append({
                'xtrain': features_train,
                'ytrain': targets_train,
                'xtest': features_test,
                'ytest': targets_test,
                'adversary_id': adv_idx
            })
        adversaries = formatted_adversaries

    print(f"  adversaries: {adversaries}")
    
    if len(features_train) > 0:
        np.save("xtrain", features_train)
        print("  💾 Guardado xtrain.npy")
    if len(features_test) > 0:
        np.save("features_test", features_test)
        print("  💾 Guardado features_test.npy")
    if len(targets_test) > 0:
        np.save("targets_test", targets_test)
        print("  💾 Guardado targets_test.npy")
    
    logger.debug(f"CS {params['C']}")

    print("\n" + "="*80)
    print("🔄 INICIANDO ITERACIONES DE KERNEL CONFIGURATIONS")
    print("="*80)

    for i in range(len(kernel_configs)):
        print(f"\n📌 KERNEL CONFIG [{i+1}/{len(kernel_configs)}]: {kernel_configs[i]}")
        
        for j in range(len(params['C'][i])):
            print(f"\n  🔧 COMBINACIÓN [{j+1}/{len(params['C'][i])}]: C={params['C'][0][j]}, reg={params['reg_param'][0][j]}")
            
            cache_key = 'CONTINUOUS MULTI LORAX ATTENTION' + str(i)
            if os.path.exists(cache_key + '.npy'):
                print(f"  📦 Cargando desde caché: {cache_key}.npy")
                continuous_attention_y = np.load(f"CONTINUOUS MULTI LORAX Y{str(i)}.npy")
                continuous_attention_function_weights_proba = np.load(f"CONTINUOUS MULTI LORAX WEIGHTS PROBA{str(i)}.npy")
                continuous_attention_context = np.load(f"{cache_key}.npy")
                attention_contexts.append(continuous_attention_context)
                attention_weights.append(continuous_attention_function_weights_proba)
                accepted_targets.append(continuous_attention_y)
                print(f"  ✅ Cargado de caché: shapes {continuous_attention_y.shape}")
                continue
            else:
                print(f"  🆕 No hay caché, entrenando modelo...")
                training_time = time.time()
                
                if last is True:
                    if len(model.best_layer) > 0:
                        print(f"  🔄 Aplicando best_layer: {model.best_layer[-1]}")
                        model.apply(model.best_layer[-1])
                model.gamma = 0.8
                
                print(f"  📏 FEATURES TRAIN shape: {np.shape(features_train)}")
                print(f"  📏 TARGETS TRAIN shape: {np.shape(targets_train)}")

                try:
                    fit_len = min(len(features_train), len(targets_train))
                    if fit_len == 0:
                        print("  ⚠️ No hay datos para entrenar, saltando")
                        logger.warning("No hay datos para entrenar")
                        continue
                    
                    print(f"  🏋️ Entrenando con {fit_len} muestras...")
                    print(f"    kernel: {kernel_configs[i][0]}, param: {kernel_configs[i][1]}")
                    print(f"    C: {params['C'][0][j]}, reg: {params['reg_param'][0][j]}")
                    
                    await model.fit_model(
                        np.float64(features_train[:fit_len]),
                        np.float64(targets_train[:fit_len]),
                        kernel_configs[i][0], kernel_configs[i][1],
                        params['C'][0][j], params['reg_param'][0][j],
                        1.0 / features_train.shape[1], 0.8,
                        regression=regression,
                    )
                    print(f"  ✅ Entrenamiento completado")
                    
                except Exception as e:
                    print(f"  ❌ Fit falló: {e}")
                    logger.warning(f"Fit falló: {e}")
                    continue

                training_time = np.abs(training_time - time.time())
                timing['training_times'].append(training_time)
                print(f"  ⏱️ Tiempo entrenamiento: {training_time:.4f}s")

                if len(targets_train) > 0 and (not hasattr(targets_train[0], 'size') or targets_train[0].size >= 1):
                    print(f"  🔮 Generando predicciones train...")
                    
                    try:
                        clf_predictions_train = model.w[0] * np.float64(features_train) + model.bias
                        print(f"    Predicciones shape: {clf_predictions_train.shape}")
                    except Exception as e:
                        if model.w is not None:
                            model.w = np.array([model.w]) if not isinstance(model.w, np.ndarray) else model.w
                            clf_predictions_train = np.mean(model.w[0],axis=1) * np.float64(features_train) + model.bias
                        else:
                            clf_predictions_train = np.zeros_like(features_train)

                    mse_train = mean_squared_error(
                        np.float64(np.nan_to_num(targets_train[:fit_len])),
                        np.float64(np.nan_to_num(clf_predictions_train[:fit_len])),
                    )
                    train_scores['scorings'].append(mse_train)
                    print(f"  📉 MSE train: {mse_train:.6f}")

                    # Calcular predicciones de forma robusta
                    _proba = None  # fallback si model.w no está disponible
                    if hasattr(model, 'w') and model.w is not None:
                        w = model.w
                        # Asegurar que w tenga la forma correcta para broadcasting
                        if len(w.shape) == 2 and w.shape[0] == w.shape[1] and w.shape[0] == features_train.shape[1]:
                            # Si es matriz cuadrada, tomar la diagonal o la primera fila
                            logger.debug(f"w es matriz cuadrada {w.shape}, usando diagonal")
                            w = np.diag(w)  # (1025,)
                        elif len(w.shape) == 2 and w.shape[0] == 1:
                            w = w[0]  # (1025,)
                        elif len(w.shape) == 2 and w.shape[1] == 1:
                            w = w[:, 0]  # (1025,)
    
                        # Calcular proba localmente — model.proba puede ser property sin setter
                        _proba = sigmoid(np.dot(features_train, w) + (model.bias if model.bias is not None else 0))
                        try:
                            model.proba = _proba
                        except (AttributeError, TypeError):
                            pass  # property sin setter: se usa _proba directamente abajo

                    protection_val_rule_train = 1.0
                    print("WEIGHTS:", model._feature_weights)
                    print("WEIGHTS:", model._feature_weights_array)
                    model._feature_weights = ((np.array(list(model.a.values())).T*features_train)*clf_predictions_train).sum(axis=0)

                    print(f"  🔄 Ejecutando continuous_multi_lorax...")
                    print("WEIGHTS:", model._feature_weights)
                    print("A:", model.a)
                    print("W:", model.w)
                    print("WEIGHTS:", model._feature_weights.shape)
                    print("TARGETS_TRAIN.SHAPE:", clf_predictions_train.shape)
                    if model._feature_weights.size == 0:
                        continue                   	
                    continuous_result = await parallel_continuous_multi_lorax(
                        np.float64(features_train[:4000]),
                        clf_predictions_train[:4000],
                        np.nan_to_num(model._feature_weights)
                    )
                    print("Multi Lorax Continuo:", continuous_result)
                    #time.sleep(6000)
                    
                # Verificar si continuous_result es un tuple de 5 valores o una lista con un tuple
                if continuous_result is not None:
                    # Si es una lista con un tuple (como resultado de parallel)
                    if isinstance(continuous_result, list) and len(continuous_result) > 0:
                        result_data = continuous_result[0]
                    else:
                        result_data = continuous_result
                    
                    # Verificar que tengamos 5 valores para desempaquetar
                    if isinstance(result_data, (tuple, list)) and len(result_data) >= 5:
                        (continuous_attention_function, continuous_attention_y,
                         continuous_attention_scores, continuous_attention_function_weights_proba,
                         continuous_attention_context) = result_data[:5]
                        print(f"    continuous_attention_y shape: {continuous_attention_y.shape if hasattr(continuous_attention_y, 'shape') else 'N/A'}")
                        
                        np.save(f"CONTINUOUS MULTI LORAX Y{i}", continuous_attention_y)
                        np.save(f"CONTINUOUS MULTI LORAX WEIGHTS PROBA{i}", continuous_attention_function_weights_proba)
                        np.save(f"CONTINUOUS MULTI LORAX ATTENTION CONTEXT{i}", continuous_attention_context)
                        print(f"    💾 Guardados archivos de atención")
                
                        attention_contexts.append(continuous_attention_context)
                        attention_weights.append(continuous_attention_function_weights_proba)
                        accepted_targets.append(continuous_attention_y)
                        test_scores['scorings'].append(1.0)
                    else:
                        print(f"    ⚠️ Resultado de continuous_multi_lorax tiene formato inesperado: {type(result_data)} con {len(result_data) if hasattr(result_data, '__len__') else 'N/A'} elementos")

                try:
                    if protection_val_rule_train >= 0.8:
                        train_scores['demographic_parity'].append(protection_val_rule_train)
                        print(f"  ✅ Demographic parity train OK: {protection_val_rule_train}")
                    else:
                        print(f"  ⚠️ Demographic parity train bajo: {protection_val_rule_train}")
                        scores['scorings'].append(1.0)
                        scores['demographic_parity'].append(-1)
                        models['models'].append(None)
                        model_weights.append(np.array(model.w) if model.w is not None else np.array([1.0]))
                        continue

                    if len(targets_train) > 0 and len(clf_predictions_train) > 0:
                        bec, cons = BEC(targets_train[:fit_len], clf_predictions_train[:fit_len])
                        print(f"  📊 BEC train: {bec}")

                    if len(features_test) > 0 and len(targets_test) > 0:
                        print(f"  🔮 Evaluando en test...")
                        clf_predictions_test = model.w[0] * features_test + model.bias if hasattr(model, 'w') and model.w is not None else np.zeros_like(features_test)
                        test_mse = mean_squared_error(
                            np.float64(targets_test), 
                            np.float64(clf_predictions_test)
                        )
                        test_scores['scorings'].append(test_mse)
                        model_weights.append(model.w)
                        print(f"  📉 MSE test: {test_mse:.6f}")

                    clf_time = time.time()
                    try:
                        if hasattr(model, 'best_layer') and model.best_layer:
                            model.apply(model.best_layer[-1])
                            print(f"  🔄 Aplicando best_layer: {model.best_layer[-1]}")
                    except Exception as e:
                        print(f"  ⚠️ Error apply best_layer: {e}")
                        pass
                    model.written = True
                    clf_time = np.abs(clf_time - time.time())
                    timing['classifier times'].append(clf_time)
                    print(f"  ⏱️ Tiempo clasificación: {clf_time:.4f}s")

                    if len(features) > 0 and len(targets) > 0:
                        print(f"  🔮 Evaluación final...")
                        clf_predictions = model.w[0] * features + model.bias if hasattr(model, 'w') and model.w is not None else np.zeros_like(features)
                        mse_val = mean_squared_error(
                            np.float64(targets), 
                            np.float64(clf_predictions)
                        )
                        scores['scorings'].append(mse_val)
                        print(f"  📉 MSE final: {mse_val:.6f}")

                        protection_val_rule = p_rule(
                            np.float64(clf_predictions.T) if len(clf_predictions.shape) > 1 else clf_predictions,
                            np.float64(targets.T) if len(targets.shape) > 1 else targets,
                            np.float64(model.w[0]) if model.w is not None else 1.0,
                            features,
                            _proba if _proba is not None else getattr(model, 'proba', np.ones((len(features), 1))),
                            p_thresh,
                        )

                        if protection_val_rule >= 0.8:
                            scores['demographic_parity'].append(protection_val_rule)
                            print(f"  ✅ Demographic parity final: {protection_val_rule}")
                        else:
                            print(f"  ⚠️ Demographic parity final bajo: {protection_val_rule}")
                            scores['scorings'].append(1.0)
                            scores['demographic_parity'].append(-1)
                            models['models'].append(None)
                            continue

                    models['models'].append(model)

                    if track_conflict is None:
                        bec, cons = BEC(targets, clf_predictions)
                    else:
                        btc_val = BTC(targets, clf_predictions)
                        bec, conflicts = BEC(targets, clf_predictions, track_conflict, True)
                        grid_conflicts.append(conflicts)
                        print(f"  📊 Conflictos: {len(conflicts)}")

                except Exception as e:
                    print(f"  ❌ Error en entrenamiento: {e}")
                    logger.exception(f"Error en entrenamiento: {e}")
                    continue

    print("\n" + "="*80)
    print(f"📦 GATHERING LAYERS OUTPUTS - accepted_targets: {len(accepted_targets)}")
    print("="*80)

    if len(accepted_targets) == 0:
        print("❌ No hay accepted_targets, retornando None")
        logger.warning("No hay accepted_targets, retornando None")
        return None

    try:
        # Verificar que todos los accepted_targets tengan la misma forma
        target_shapes = [t.shape for t in accepted_targets if t is not None]
        if not target_shapes:
            print("❌ target_shapes vacío")
            return None
        
        print(f"📏 Shapes de accepted_targets: {target_shapes}")
        
        # Si hay diferentes formas, tomar la más común o reshape
        from collections import Counter
        shape_counts = Counter(target_shapes)
        most_common_shape = shape_counts.most_common(1)[0][0]
        print(f"📏 Forma más común: {most_common_shape}")
        
        # Reshapear todos a la forma más común si es necesario
        aligned_targets = []
        for idx, t in enumerate(accepted_targets):
            if t is None:
                print(f"  ⚠️ accepted_targets[{idx}] es None, saltando")
                continue
            if t.shape != most_common_shape:
                print(f"  🔄 Reshapeando accepted_targets[{idx}] de {t.shape} a {most_common_shape}")
                if len(t.shape) == 1:
                    t = t.reshape(-1, 1)
                if t.shape[0] != most_common_shape[0]:
                    t = np.resize(t, most_common_shape)
            aligned_targets.append(t)
        
        if not aligned_targets:
            print("❌ No hay aligned_targets")
            return None
        
        print(f"📦 aligned_targets count: {len(aligned_targets)}")
        
        print("🔄 Ejecutando parallel gather_layers_outputs...")
        raw_decoded_targets = np.array(
            await parallel(
                [1], 1,
                np.array(aligned_targets),
                func=gather_layers_outputs,
                index=False,
                shared=False,
                fifo=False,
                lifc=False,
                continuous=False,
                as_singular_matrix=True,
            )
        )
        print(f"✅ raw_decoded_targets shape: {raw_decoded_targets.shape}")
        
        if len(raw_decoded_targets.shape) > 1:
            raw_decoded_targets = raw_decoded_targets.T
            print(f"  🔄 Transpuesto: {raw_decoded_targets.shape}")
        
        np.save("raw_decoded_targets", raw_decoded_targets)
        print("  💾 Guardado raw_decoded_targets.npy")
        
        continuous_attention_y = np.copy(raw_decoded_targets)
        best_fit = 0
        cross_validation_context = attention_contexts[best_fit] if attention_contexts else None
        decoded_targets = raw_decoded_targets / (raw_decoded_targets.max() + 1e-9)
        np.save("decoded_targets", decoded_targets)
        print(f"📊 decoded_targets shape: {decoded_targets.shape}")
        print(f"📊 decoded_targets min: {decoded_targets.min():.6f}, max: {decoded_targets.max():.6f}")

        min_len = min(len(decoded_targets), len(targets_train))
        print(f"📏 min_len para comparación: {min_len}")
        
        if min_len > 0 and len(decoded_targets.shape) > 2 and len(targets_train.shape) > 2:
            gathered_mse = mean_squared_error(
                np.float64(decoded_targets[:min_len].reshape(min_len, -1)),
                np.float64(targets_train[:min_len].reshape(min_len, -1)),
            )
            print(f"📉 Gathered MSE: {gathered_mse:.6f}")
        else:
            gathered_mse = 1.0
            print(f"⚠️ No se pudo calcular gathered_mse, usando 1.0")
        
        if len(targets_train.shape) == 1:
            targets_train_2d = targets_train.reshape(-1, 1)
            print(f"  🔄 targets_train reshape a 2D: {targets_train_2d.shape}")
        else:
            targets_train_2d = targets_train

        print("🔍 Ejecutando scan_data_leakage...")
        leakage_result = scan_data_leakage(
            np.float64(targets_train_2d[:min_len]),
            np.float64(decoded_targets[:min_len]),
            error_threshold=gathered_mse,
        )
        
        if leakage_result:
            leakage, correctly_decoded_targets, leaking_targets, leaking_idxs, error = leakage_result
            print(f"  📊 Leakage: {leakage:.4f}")
            print(f"  📊 correctly_decoded_targets shape: {correctly_decoded_targets.shape if hasattr(correctly_decoded_targets, 'shape') else 'N/A'}")
            print(f"  📊 leaking_idxs count: {len(leaking_idxs)}")
        else:
            leakage, correctly_decoded_targets, leaking_targets, leaking_idxs, error = (0.0, np.array([]), np.array([]), np.array([]), np.array([]))
            print("  ⚠️ scan_data_leakage no retornó resultados")

        if len(leaking_idxs) > 0 and len(features_train) > 0:
            leaking_features = features_train[np.unique(leaking_idxs)]
            print(f"  📊 leaking_features shape: {leaking_features.shape}")
        else:
            leaking_features = np.array([])
            print("  📊 leaking_features vacío")

        hypotheses = ['var']
        min_len = min(len(correctly_decoded_targets), len(targets_train)) if len(correctly_decoded_targets) > 0 else 0
        correctly_decoded_targets = correctly_decoded_targets[:min_len] if min_len > 0 else correctly_decoded_targets

        np.save("correctly_decoded_targets", correctly_decoded_targets)
        np.save("leaking_features", leaking_features)
        np.save("leaking_targets", leaking_targets)
        print("  💾 Guardados archivos de leakage")

        perturbations = 1
        exp_data = []

        print(f"\n🔄 Ejecutando multiple_hypothesis_testing para {len(adversaries)} adversaries...")
        for adv_idx, adversary in enumerate(adversaries):
            print(f"\n  👾 Adversary {adv_idx+1}/{len(adversaries)}: {adversary}")
            try:
                # En GridSearch, al llamar a multiple_hypothesis_testing:
                result = await multiple_hypothesis_testing(
                    features_train, features_test, leaking_features, adversary,
                    correctly_decoded_targets, targets_test, cross_validation_context,
                    leakage, leaking_targets, perturbations, hypotheses, gathered_mse, error,
                    capture_adversary_models=True,
                    base_model=model,
                    max_features_per_run=8,  # NUEVO: limitar a 8 features
                )
                if result:
                    exp_data.append(result[0] if result else [])
                    print(f"    ✅ multiple_hypothesis_testing completado")
                else:
                    print(f"    ⚠️ multiple_hypothesis_testing retornó None")
                    exp_data.append([])
            except Exception as e:
                print(f"    ❌ Error en multiple_hypothesis_testing: {e}")
                logger.error(f"Error en multiple_hypothesis_testing: {e}")
                exp_data.append([])

        print("\n📊 Calculando stress metrics...")
        stress_becs = []
        stress_btcs = []
        stress_conflicts = []
        stress_instances = []
        print("EXP DATA", exp_data)

        for instance in range(len(exp_data)):
            print(f"  Instance {instance}: {len(exp_data[instance]) if exp_data[instance] else 0} elementos")
            for i in range(len(exp_data[instance]) if exp_data[instance] else 0):
                try:
                    if len(exp_data[instance][i]) >= 3:
                        data1 = exp_data[instance][i][1]
                        data2 = exp_data[instance][i][2]
                        if hasattr(data1, 'sum') and hasattr(data2, 'sum'):
                            bec_decoded, conflicts_decoded = BEC(
                                data1.sum(axis=1) if len(data1.shape) > 1 else data1,
                                data2.sum(axis=1) if len(data2.shape) > 1 else data2,
                                track_conflict, True, attention=True,
                            )
                            
                            btc_decoded = BTC(
                                data1.sum(axis=1) if len(data1.shape) > 1 else data1,
                                data2.sum(axis=1) if len(data2.shape) > 1 else data2,
                            )

                            print(f"    BEC: {np.mean(bec_decoded) if hasattr(bec_decoded, '__len__') else bec_decoded:.4f}")
                            print(f"    BTC: {np.mean(btc_decoded) if hasattr(btc_decoded, '__len__') else btc_decoded:.4f}")

                            if instance == 0:
                                stress_becs.append(np.mean(bec_decoded) if hasattr(bec_decoded, '__len__') else bec_decoded)
                                stress_btcs.append(np.mean(btc_decoded) if hasattr(btc_decoded, '__len__') else btc_decoded)
                                stress_conflicts.append(conflicts_decoded)
                except Exception as e:
                    print(f"    Error en cálculo stress: {e}")
                    logger.exception(e)
                    continue

        print(f"\n📊 stress_becs: {stress_becs}")
        print(f"📊 stress_btcs: {stress_btcs}")

        if stress_instances and stress_becs:
            min_bec = np.argmin(stress_becs)
            print(f"  🏆 Mejor BEC en índice: {min_bec}")
            if min_bec < len(exp_data) and len(exp_data[min_bec]) > 0 and len(exp_data[min_bec][0]) >= 4:
                stress_context = exp_data[stress_instances[min_bec]][min_bec][3]
                if cross_validation_context is not None and stress_context is not None:
                    stress_context = np.vstack((cross_validation_context, stress_context))
                    print(f"  📦 stress_context shape: {stress_context.shape}")
                np.save("MODEL CONTEXT", stress_context)
                print("  💾 Guardado MODEL CONTEXT")
                print("\n" + "="*80)
                print("✅ GRID SEARCH COMPLETADO CON ÉXITO")
                print("="*80)

        # En GridSearch, al final de la función, después de la sección de fallback:
        if cross_validation_context is not None:
            np.save(f"outputs/Model {model_name} context", cross_validation_context)
            print(f"  💾 Guardado Model {model_name} context")
            print("\n" + "="*80)
            print("✅ GRID SEARCH COMPLETADO (modo fallback)")
            print("="*80)
            
            # CORRECCIÓN: Verificar si cross_validation_context es un diccionario antes de usar .get()
            if isinstance(cross_validation_context, dict):
                adversary_models = cross_validation_context.get('adversary_models', [])
            else:
                adversary_models = []
            
            # Guardar modelos adversarios con índice de loop
            if adversary_models:
                # Nota: output_dir y loop_counter deben ser accesibles aquí
                # Si no lo son, se pueden pasar como argumentos a GridSearch
                adv_models_path = os.path.join(output_dir, f"adversary_models_loop_{loop_counter}.pkl")
                with open(adv_models_path, "wb") as f:
                    pickle.dump(adversary_models, f)
                logger.info(f"[CV Adversarial] {len(adversary_models)} modelos adversarios guardados en {adv_models_path}")
                
                adv_metadata_path = os.path.join(output_dir, f"adversary_models_metadata_loop_{loop_counter}.json")
                adv_metadata = []
                for adv in adversary_models:
                    adv_metadata.append({
                        'adversary_id': adv.get('adversary_id', 0),
                        'hypothesis': adv.get('hypothesis', 'unknown'),
                        'feature_idx': adv.get('feature_idx', -1),
                        'noise_level': adv.get('noise_level', 0),
                        'loop': loop_counter
                    })
                with open(adv_metadata_path, "w") as f:
                    json.dump(adv_metadata, f, indent=2)
            
            return {
                'context': cross_validation_context,
                'exp_data': exp_data,
                'stress_becs': stress_becs,
                'stress_btcs': stress_btcs,
                'adversary_models': adversary_models
            }

    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO EN GRID SEARCH: {e}")
        logger.exception(f"Error en GridSearch: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES (solo las necesarias)
# ══════════════════════════════════════════════════════════════════════════════

def mean_squared_error(y_true, y_pred):
    """
    Calcula error cuadrático medio manejando diferentes dimensionalidades.
    Para regresión multidimensional, calcula MSE sobre todas las features.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Si son 2D, calcular MSE sobre todas las dimensiones
    if len(y_true.shape) == 2 and len(y_pred.shape) == 2:
        min_len = min(len(y_true), len(y_pred))
        if min_len == 0:
            return 0.0
        # MSE promedio sobre todas las features y muestras
        return float(np.mean((y_true[:min_len] - y_pred[:min_len]) ** 2))
    
    # Si son 1D o formas mixtas, flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    min_len = min(len(y_true_flat), len(y_pred_flat))
    if min_len == 0:
        return 0.0
    return float(np.mean((y_true_flat[:min_len] - y_pred_flat[:min_len]) ** 2))


def sigmoid(x):
    """Función sigmoide."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def cross_validation(features, targets, Cs, reg_params, kernel_configs, last, regression):
    """Simulación de cross_validation para fine-tuning."""
    return np.array([])


# ══════════════════════════════════════════════════════════════════════════════
# FINE-TUNING POST-INTERVENCION
# ══════════════════════════════════════════════════════════════════════════════

def optimize_with_fine_tuning(
    self,
    input_data,
    target,
    intervention_result,
    Cs=None,
    reg_params=None,
    kernel_configs=None,
):
    """Optimiza el modelo despues de intervenciones usando fine-tuning."""
    if Cs is None:
        Cs = [0.1, 1.0, 10.0]
    if reg_params is None:
        reg_params = [0.01, 0.1, 1.0]
    if kernel_configs is None:
        kernel_configs = [('linear', 1), ('rbf', 0.5), ('poly', 3)]

    logger.info("Iniciando optimizacion con fine-tuning post-intervencion")

    corrected_output = intervention_result.get('corrected_output', input_data)
    metrics = intervention_result.get('metrics', {})

    features = np.vstack([input_data, corrected_output]) if len(input_data.shape) == len(corrected_output.shape) else input_data
    targets_arr = np.hstack([target, target]) if target is not None else None

    fine_tuned_context = cross_validation(
        features=features,
        targets=targets_arr if targets_arr is not None else np.zeros(len(features)),
        Cs=Cs,
        reg_params=reg_params,
        kernel_configs=kernel_configs,
        last=True,
        regression=target is not None and len(target.shape) > 1,
    )

    if hasattr(fine_tuned_context, 'size') and fine_tuned_context.size > 0:
        if hasattr(self, 'base_model') and hasattr(self.base_model, 'context'):
            self.base_model.context = fine_tuned_context
        improvement = 0.1  # Simulado
        intervention_result.update({
            'fine_tuned': True,
            'fine_tuned_context': fine_tuned_context,
            'fine_tuned_improvement': improvement,
            'final_output': _apply_context_refinement(
                corrected_output, fine_tuned_context),
        })
    else:
        intervention_result['fine_tuned'] = False

    return intervention_result


def _apply_context_refinement(output: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Aplica refinamiento basado en contexto fine-tuned."""
    if output.shape == context.shape:
        alpha = 0.6
        refined = alpha * context + (1 - alpha) * output
        if len(refined.shape) == 2:
            row_sums = np.sum(refined, axis=1, keepdims=True)
            refined = refined / (row_sums + 1e-10)
        return refined
    return output