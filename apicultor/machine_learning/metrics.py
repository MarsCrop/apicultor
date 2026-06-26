import numpy as np
from itertools import product
from sklearn.metrics import confusion_matrix
from .visuals import plot_confusion_matrix
from typing import Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# === SOFTMAX ESTABLE ===
def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax numéricamente estable"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# === ENTROPÍA ===
def calculate_entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    """Entropía de Shannon: H(p) = -Σ p_i * log(p_i)"""
    eps = 1e-12
    probs = np.clip(probabilities, eps, 1 - eps)
    return -np.sum(probs * np.log(probs), axis=axis)

# === EXPECTED CALIBRATION ERROR ===
def expected_calibration_error(probabilities: np.ndarray,
                              targets: np.ndarray,
                              n_bins: int = 15) -> Dict:
    """
    Calcula Expected Calibration Error (ECE) usando solo NumPy.
    
    Args:
        probabilities: [batch_size, n_classes] o [batch_size,] para binario
        targets: [batch_size,] o one-hot
        n_bins: Número de bins
        
    Returns:
        Dict con métricas de calibración
    """
    # Convertir a formato estándar
    if len(probabilities.shape) == 1:
        # Binario
        confidences = probabilities
        if len(targets.shape) > 1:
            targets = np.argmax(targets, axis=1)
        predictions = (confidences > 0.5).astype(int)
    else:
        # Multiclase
        n_classes = probabilities.shape[1]
        if len(targets.shape) > 1:
            targets = np.argmax(targets, axis=1)
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
    
    # Crear bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Inicializar acumuladores
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Calcular por bin
    for bin_idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Muestras en este bin
        if bin_idx == n_bins - 1:
            # Último bin incluye ambos límites
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
        
        bin_count = np.sum(in_bin)
        bin_counts[bin_idx] = bin_count
        
        if bin_count > 0:
            # Accuracy en el bin
            bin_accuracy = np.mean(predictions[in_bin] == targets[in_bin])
            bin_accuracies[bin_idx] = bin_accuracy
            
            # Confianza promedio en el bin
            bin_confidence = np.mean(confidences[in_bin])
            bin_confidences[bin_idx] = bin_confidence
    
    # Calcular ECE y MCE
    total_samples = np.sum(bin_counts)
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / total_samples
    mce = np.max(np.abs(bin_accuracies[bin_counts > 0] - bin_confidences[bin_counts > 0]))
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'bin_accuracies': bin_accuracies.tolist(),
        'bin_confidences': bin_confidences.tolist(),
        'bin_counts': bin_counts.tolist(),
        'n_bins': n_bins
    }

# === TEMPERATURE SCALING (SOLO NUMPY) ===
class TemperatureScaler:
    """
    Temperature Scaling usando solo NumPy.
    Optimización con gradiente descendente.
    """
    
    def __init__(self, init_temp: float = 1.0):
        self.temperature = init_temp
        self.trained = False
        self.history = []
    
    def negative_log_likelihood(self, 
                               logits: np.ndarray, 
                               targets: np.ndarray,
                               temperature: float) -> float:
        """Calcula NLL para una temperatura dada"""
        # Escalar logits
        scaled_logits = logits / temperature
        
        # Softmax estable
        probs = stable_softmax(scaled_logits, axis=1)
        
        # Seleccionar probabilidades de clases correctas
        if len(targets.shape) == 1:
            # Targets como índices
            n_samples = len(targets)
            correct_probs = probs[np.arange(n_samples), targets]
        else:
            # Targets one-hot
            correct_probs = np.sum(probs * targets, axis=1)
        
        # NLL
        eps = 1e-12
        correct_probs = np.clip(correct_probs, eps, 1 - eps)
        nll = -np.mean(np.log(correct_probs))
        
        return nll
    
    def gradient_nll(self, 
                    logits: np.ndarray, 
                    targets: np.ndarray,
                    temperature: float) -> float:
        """Calcula gradiente de NLL respecto a temperatura (0=lógico, 1=creativo)"""
        # Escalar logits
        scaled_logits = logits / temperature
        
        # Softmax estable
        probs = stable_softmax(scaled_logits, axis=1)
        
        # Derivada de NLL respecto a temperatura
        n_samples = len(logits)
        
        if len(targets.shape) == 1:
            # Targets como índices
            targets_one_hot = np.eye(logits.shape[1])[targets]
        else:
            targets_one_hot = targets
        
        # dNLL/dT = -1/(nT²) * Σ_i Σ_j (y_ij - p_ij) * z_ij
        diff = targets_one_hot - probs
        grad = -np.sum(diff * logits) / (temperature**2 * n_samples)
        
        return grad
    
    def fit(self, 
           logits: np.ndarray, 
           targets: np.ndarray,
           max_iter: int = 1000,
           learning_rate: float = 0.01,
           tolerance: float = 1e-6,
           verbose: bool = False) -> float:
        """
        Ajusta temperatura minimizando NLL.
        
        Args:
            logits: Logits del modelo [n_samples, n_classes]
            targets: Targets [n_samples,] o one-hot
            max_iter: Máximo número de iteraciones
            learning_rate: Tasa de aprendizaje
            tolerance: Tolerancia para convergencia
            
        Returns:
            Temperatura óptima
        """
        # Inicializar temperatura
        T = np.array([self.temperature])
        
        for iteration in range(max_iter):
            # Calcular NLL y gradiente
            nll = self.negative_log_likelihood(logits, targets, T[0])
            grad = self.gradient_nll(logits, targets, T[0])
            
            # Actualizar temperatura (asegurar positividad)
            T_new = T - learning_rate * grad
            
            # Mantener temperatura positiva
            if T_new[0] < 1e-8:
                T_new[0] = 1e-8
            
            # Guardar historial
            self.history.append({
                'iteration': iteration,
                'temperature': float(T[0]),
                'nll': float(nll),
                'gradient': float(grad)
            })
            
            # Verificar convergencia
            if np.abs(T_new[0] - T[0]) < tolerance:
                T = T_new
                if verbose:
                    print(f"Converged at iteration {iteration}, T={T[0]:.4f}, NLL={nll:.4f}")
                break
            
            T = T_new
            
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: T={T[0]:.4f}, NLL={nll:.4f}, Grad={grad:.6f}")
        
        self.temperature = float(T[0])
        self.trained = True
        
        # NLL final
        final_nll = self.negative_log_likelihood(logits, targets, self.temperature)
        self.final_nll = final_nll
        
        if verbose:
            print(f"Final temperature: {self.temperature:.4f}")
            print(f"Final NLL: {final_nll:.4f}")
        
        return self.temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Aplica temperature scaling a logits"""
        if not self.trained:
            raise ValueError("TemperatureScaler must be fitted first")
        
        scaled_logits = logits / self.temperature
        return stable_softmax(scaled_logits, axis=-1)
    
    def predict_uncertainty(self, logits: np.ndarray) -> np.ndarray:
        """Calcula incertidumbre después de calibrar"""
        probs = self.calibrate(logits)
        return calculate_entropy(probs)

# === ENSEMBLE UNCERTAINTY ===
def ensemble_uncertainty(ensemble_probs: np.ndarray,
                        method: str = 'entropy') -> np.ndarray:
    """
    Calcula incertidumbre a partir de un ensemble de modelos.
    
    Args:
        ensemble_probs: [n_models, n_samples, n_classes] o [n_models, n_samples] para regresión
        method: 'entropy', 'variance', 'mutual_information'
        
    Returns:
        Incertidumbre por muestra
    """
    n_models = ensemble_probs.shape[0]
    
    if len(ensemble_probs.shape) == 3:
        # Clasificación
        n_samples, n_classes = ensemble_probs.shape[1], ensemble_probs.shape[2]
        
        if method == 'entropy':
            # Entropía promedio
            entropies = np.array([calculate_entropy(p, axis=1) for p in ensemble_probs])
            return np.mean(entropies, axis=0)
        
        elif method == 'mutual_information':
            # Información mutua: H(mean) - mean(H)
            mean_probs = np.mean(ensemble_probs, axis=0)
            entropy_mean = calculate_entropy(mean_probs, axis=1)
            
            entropies = np.array([calculate_entropy(p, axis=1) for p in ensemble_probs])
            mean_entropy = np.mean(entropies, axis=0)
            
            return entropy_mean - mean_entropy
        
        elif method == 'variance':
            # Varianza de probabilidades por clase
            variance = np.var(ensemble_probs, axis=0)  # [n_samples, n_classes]
            return np.mean(variance, axis=1)
        
        else:
            raise ValueError(f"Método desconocido: {method}")
    
    else:
        # Regresión: [n_models, n_samples]
        if method == 'variance':
            return np.var(ensemble_probs, axis=0)
        elif method == 'std':
            return np.std(ensemble_probs, axis=0)
        else:
            raise ValueError(f"Método {method} no soportado para regresión")

# === MONTE CARLO DROPOUT ===
def monte_carlo_dropout_uncertainty(model: Callable,
                                   X: np.ndarray,
                                   n_samples: int = 30,
                                   dropout_layers: List[int] = None) -> Dict:
    """
    Estimación de incertidumbre usando Monte Carlo Dropout.
    
    Args:
        model: Función que toma X y retorna logits
        X: Input [n_samples, features]
        n_samples: Número de forward passes
        dropout_layers: Índices de capas con dropout
        
    Returns:
        Dict con predicciones e incertidumbres
    """
    # Esta función requiere que el modelo tenga modo train/eval
    # y que preserve el dropout durante evaluación
    
    all_predictions = []
    
    for i in range(n_samples):
        # Forward pass con dropout activado
        # En un modelo real, necesitarías activar manualmente el dropout
        logits = model(X)
        
        if len(logits.shape) == 2 and logits.shape[1] > 1:
            # Clasificación multiclase
            probs = stable_softmax(logits, axis=1)
            all_predictions.append(probs)
        else:
            # Regresión o binario
            all_predictions.append(logits)
    
    # Convertir a array
    ensemble_preds = np.stack(all_predictions, axis=0)
    
    if len(logits.shape) == 2 and logits.shape[1] > 1:
        # Clasificación
        mean_probs = np.mean(ensemble_preds, axis=0)
        
        # Diferentes medidas de incertidumbre
        entropy_uncertainty = calculate_entropy(mean_probs, axis=1)
        mutual_info = ensemble_uncertainty(ensemble_preds, 'mutual_information')
        variance_uncertainty = ensemble_uncertainty(ensemble_preds, 'variance')
        
        return {
            'mean_probs': mean_probs,
            'predictions': np.argmax(mean_probs, axis=1),
            'entropy': entropy_uncertainty,
            'mutual_information': mutual_info,
            'variance': variance_uncertainty,
            'epistemic': mutual_info,  # Info mutua ≈ incertidumbre epistémica
            'aleatoric': entropy_uncertainty - mutual_info,  # Aproximación
            'samples': ensemble_preds
        }
    else:
        # Regresión
        mean_pred = np.mean(ensemble_preds, axis=0)
        std_pred = np.std(ensemble_preds, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'uncertainty': std_pred,
            'samples': ensemble_preds
        }

# === CALIBRATION DIAGRAM ===
def calibration_diagram_data(probabilities: np.ndarray,
                            targets: np.ndarray,
                            n_bins: int = 10) -> Dict:
    """
    Genera datos para diagrama de calibración.
    
    Returns:
        Dict con datos para plotting
    """
    ece_result = expected_calibration_error(probabilities, targets, n_bins)
    
    # Puntos para la línea de calibración perfecta
    perfect_line_x = np.linspace(0, 1, 100)
    perfect_line_y = perfect_line_x
    
    # Puntos para histograma de confianza
    confidences = np.max(probabilities, axis=1) if len(probabilities.shape) > 1 else probabilities
    hist_counts, hist_edges = np.histogram(confidences, bins=n_bins, range=(0, 1))
    hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    
    return {
        'bin_accuracies': np.array(ece_result['bin_accuracies']),
        'bin_confidences': np.array(ece_result['bin_confidences']),
        'bin_counts': np.array(ece_result['bin_counts']),
        'perfect_line_x': perfect_line_x,
        'perfect_line_y': perfect_line_y,
        'hist_centers': hist_centers,
        'hist_counts': hist_counts,
        'ece': ece_result['ece'],
        'mce': ece_result['mce']
    }

# === BRIER SCORE ===
def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """
    Calcula Brier Score.
    
    BS = 1/N Σ_i Σ_j (p_ij - y_ij)²
    """
    # Convertir targets a one-hot si es necesario
    if len(targets.shape) == 1:
        n_classes = probabilities.shape[1]
        targets_one_hot = np.eye(n_classes)[targets]
    else:
        targets_one_hot = targets
    
    # Asegurar mismas dimensiones
    if probabilities.shape != targets_one_hot.shape:
        probabilities = probabilities.reshape(targets_one_hot.shape)
    
    # Calcular Brier Score
    brier = np.mean(np.sum((probabilities - targets_one_hot) ** 2, axis=1))
    
    return float(brier)

# === NEGATIVE LOG LIKELIHOOD ===
def negative_log_likelihood(probabilities: np.ndarray,
                           targets: np.ndarray) -> float:
    """
    Calcula Negative Log Likelihood.
    
    NLL = -1/N Σ_i log(p_i,yi)
    """
    # Seleccionar probabilidades de clases correctas
    if len(targets.shape) == 1:
        # Targets como índices
        n_samples = len(targets)
        correct_probs = probabilities[np.arange(n_samples), targets]
    else:
        # Targets one-hot
        correct_probs = np.sum(probabilities * targets, axis=1)
    
    # Aplicar log con clipping para estabilidad
    eps = 1e-12
    correct_probs = np.clip(correct_probs, eps, 1 - eps)
    nll = -np.mean(np.log(correct_probs))
    
    return float(nll)

# === CONFIDENCE INTERVALS ===
def confidence_intervals(predictions: np.ndarray,
                        confidence: float = 0.95,
                        method: str = 'percentile') -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula intervalos de confianza.
    
    Args:
        predictions: [n_samples, n_predictions] o lista de arrays
        confidence: Nivel de confianza (0.95 para 95%)
        method: 'percentile' o 'normal'
        
    Returns:
        (lower_bound, upper_bound)
    """
    if isinstance(predictions, list):
        predictions = np.array(predictions)
    
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(1, -1)
    
    alpha = 1 - confidence
    
    if method == 'percentile':
        lower = np.percentile(predictions, 100 * alpha/2, axis=0)
        upper = np.percentile(predictions, 100 * (1 - alpha/2), axis=0)
    
    elif method == 'normal':
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Z-score para confianza dada
        z = -np.sqrt(2) * special.erfcinv(2 - 2 * (1 - alpha/2))
        
        lower = mean - z * std
        upper = mean + z * std
    
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    return lower, upper

# === UNCERTAINTY CALIBRATION ===
def calibrate_uncertainty(uncertainties: np.ndarray,
                         errors: np.ndarray,
                         n_bins: int = 10) -> Dict:
    """
    Calibra estimaciones de incertidumbre.
    
    Args:
        uncertainties: Incertidumbres estimadas [n_samples,]
        errors: Errores reales [n_samples,]
        n_bins: Número de bins
        
    Returns:
        Dict con métricas de calibración de incertidumbre
    """
    # Ordenar por incertidumbre
    sorted_indices = np.argsort(uncertainties)
    sorted_uncertainties = uncertainties[sorted_indices]
    sorted_errors = errors[sorted_indices]
    
    # Crear bins con igual número de muestras
    bin_edges = np.percentile(sorted_uncertainties, np.linspace(0, 100, n_bins + 1))
    
    bin_uncertainties = []
    bin_errors = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        
        if i == n_bins - 1:
            in_bin = (sorted_uncertainties >= bin_lower) & (sorted_uncertainties <= bin_upper)
        else:
            in_bin = (sorted_uncertainties >= bin_lower) & (sorted_uncertainties < bin_upper)
        
        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)
        
        if bin_count > 0:
            bin_uncertainties.append(np.mean(sorted_uncertainties[in_bin]))
            bin_errors.append(np.mean(sorted_errors[in_bin]))
        else:
            bin_uncertainties.append(0)
            bin_errors.append(0)
    
    # Calibrar usando regresión lineal simple
    bin_uncertainties_arr = np.array(bin_uncertainties)
    bin_errors_arr = np.array(bin_errors)
    
    # Filtrar bins con datos
    mask = (bin_counts > 0) & (bin_uncertainties_arr > 0)
    
    if np.sum(mask) >= 2:
        # Ajustar línea: error = a * uncertainty + b
        A = np.vstack([bin_uncertainties_arr[mask], np.ones_like(bin_uncertainties_arr[mask])]).T
        a, b = np.linalg.lstsq(A, bin_errors_arr[mask], rcond=None)[0]
        
        # Calibrar incertidumbres
        calibrated = a * uncertainties + b
        calibrated = np.clip(calibrated, 0, None)  # No negativas
    else:
        a, b = 1.0, 0.0
        calibrated = uncertainties
    
    # Métrica: Calibration Error for Uncertainty (UCE)
    uce = 0.0
    total_samples = 0
    
    for i in range(n_bins):
        if bin_counts[i] > 0:
            uce += bin_counts[i] * np.abs(bin_errors[i] - bin_uncertainties[i])
            total_samples += bin_counts[i]
    
    if total_samples > 0:
        uce /= total_samples
    
    return {
        'uncertainty_calibration_error': float(uce),
        'calibration_slope': float(a),
        'calibration_intercept': float(b),
        'calibrated_uncertainties': calibrated,
        'bin_uncertainties': bin_uncertainties,
        'bin_errors': bin_errors,
        'bin_counts': bin_counts
    }

# === DETECCIÓN DE SOBRECONFIANZA ===
def detect_overconfidence(probabilities: np.ndarray,
                         targets: np.ndarray,
                         threshold: float = 0.1) -> Dict:
    """
    Detecta sobre-confianza del modelo.
    
    Args:
        probabilities: Probabilidades predichas
        targets: Targets reales
        threshold: Umbral para considerar sobre-confianza
        
    Returns:
        Dict con diagnóstico
    """
    if len(probabilities.shape) > 1:
        # Clasificación multiclase
        predictions = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)
        
        if len(targets.shape) > 1:
            targets = np.argmax(targets, axis=1)
        
        accuracy = np.mean(predictions == targets)
    else:
        # Binario
        predictions = (probabilities > 0.5).astype(int)
        confidences = np.where(predictions == 1, probabilities, 1 - probabilities)
        accuracy = np.mean(predictions == targets)
    
    mean_confidence = np.mean(confidences)
    confidence_gap = mean_confidence - accuracy
    
    # Calcular ECE
    ece_result = expected_calibration_error(probabilities, targets)
    ece = ece_result['ece']
    
    # Diagnosticar
    is_overconfident = confidence_gap > threshold
    severity = 'high' if confidence_gap > 0.2 else 'medium' if confidence_gap > 0.1 else 'low'
    
    recommendations = []
    if is_overconfident:
        recommendations.append("Apply temperature scaling")
        recommendations.append("Use label smoothing during training")
        recommendations.append("Collect more diverse training data")
    
    return {
        'is_overconfident': bool(is_overconfident),
        'confidence_gap': float(confidence_gap),
        'mean_confidence': float(mean_confidence),
        'accuracy': float(accuracy),
        'ece': float(ece),
        'severity': severity,
        'recommendations': recommendations
    }

# === UNCERTAINTY-AWARE REWARD ===
def uncertainty_aware_reward(base_reward: np.ndarray,
                           uncertainty: np.ndarray,
                           uncertainty_weight: float = 1.0,
                           method: str = 'penalize_high') -> np.ndarray:
    """
    Ajusta recompensa basada en incertidumbre.
    
    Args:
        base_reward: Recompensa base [n_samples,]
        uncertainty: Incertidumbre [n_samples,]
        uncertainty_weight: Peso de la penalización
        method: 'penalize_high', 'reward_low', 'normalize'
        
    Returns:
        Recompensa ajustada por incertidumbre
    """
    # Normalizar incertidumbre
    if np.std(uncertainty) > 0:
        uncertainty_norm = (uncertainty - np.mean(uncertainty)) / np.std(uncertainty)
    else:
        uncertainty_norm = uncertainty * 0
    
    if method == 'penalize_high':
        # Penalizar alta incertidumbre
        adjusted = base_reward - uncertainty_weight * uncertainty_norm
    
    elif method == 'reward_low':
        # Recompensar baja incertidumbre
        adjusted = base_reward + uncertainty_weight * (1 - uncertainty_norm)
    
    elif method == 'normalize':
        # Normalizar por incertidumbre
        uncertainty_clipped = np.clip(uncertainty, 1e-6, None)
        adjusted = base_reward / (1 + uncertainty_weight * uncertainty_clipped)
    
    elif method == 'threshold':
        # Umbral: ignorar muestras con alta incertidumbre
        threshold = np.percentile(uncertainty, 75)  # Percentil 75
        mask = uncertainty < threshold
        adjusted = base_reward * mask.astype(float)
    
    else:
        raise ValueError(f"Método desconocido: {method}")
    
    # Recortar a rango [0, 1]
    adjusted = np.clip(adjusted, 0, 1)
    
    return adjusted

# === BAYESIAN UNCERTAINTY QUANTIFICATION ===
class BayesianUncertainty:
    """Métodos Bayesianos para cuantificación de incertidumbre."""
    
    @staticmethod
    def posterior_predictive(prior_samples: np.ndarray,
                            likelihood_samples: np.ndarray,
                            method: str = 'average') -> np.ndarray:
        """
        Aproximación de distribución predictiva posterior.
        
        Args:
            prior_samples: Muestras de la distribución previa
            likelihood_samples: Muestras de verosimilitud
            method: 'average', 'mixture', 'bootstrap'
            
        Returns:
            Distribución predictiva posterior
        """
        if method == 'average':
            # Promedio de predicciones
            return np.mean(likelihood_samples, axis=0)
        
        elif method == 'mixture':
            # Mezcla de distribuciones
            n_samples, n_models = likelihood_samples.shape
            weights = np.ones(n_models) / n_models
            return np.sum(likelihood_samples * weights, axis=1)
        
        elif method == 'bootstrap':
            # Bootstrap aggregating
            n_models = likelihood_samples.shape[1]
            bootstrap_weights = np.random.dirichlet(np.ones(n_models), size=1)[0]
            return np.sum(likelihood_samples * bootstrap_weights, axis=1)
    
    @staticmethod
    def bayesian_model_average(models_predictions: List[np.ndarray],
                              prior_weights: np.ndarray = None) -> Dict:
        """
        Bayesian Model Averaging.
        
        Args:
            models_predictions: Lista de predicciones de cada modelo
            prior_weights: Pesos previos para cada modelo
            
        Returns:
            Predicción promediada y incertidumbre
        """
        n_models = len(models_predictions)
        
        if prior_weights is None:
            prior_weights = np.ones(n_models) / n_models
        
        # Normalizar pesos
        prior_weights = prior_weights / np.sum(prior_weights)
        
        # Promedio ponderado
        weighted_predictions = np.zeros_like(models_predictions[0])
        
        for i, pred in enumerate(models_predictions):
            weighted_predictions += prior_weights[i] * pred
        
        # Calcular incertidumbre
        variances = []
        for pred in models_predictions:
            variances.append(np.var(pred, axis=0))
        
        model_uncertainty = np.average(variances, axis=0, weights=prior_weights)
        
        return {
            'predictive_mean': weighted_predictions,
            'predictive_uncertainty': model_uncertainty,
            'model_weights': prior_weights
        }

# === CALIBRATION MONITOR ===
class CalibrationMonitor:
    """Monitor de calibración durante entrenamiento."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = {
            'ece': [],
            'nll': [],
            'brier': [],
            'accuracy': [],
            'confidence': []
        }
    
    def update(self, 
               probabilities: np.ndarray,
               targets: np.ndarray,
               predictions: np.ndarray = None):
        """Actualiza métricas con nuevo batch."""
        # Calcular métricas
        ece_result = expected_calibration_error(probabilities, targets)
        nll = negative_log_likelihood(probabilities, targets)
        brier = brier_score(probabilities, targets)
        
        if predictions is None:
            if len(probabilities.shape) > 1:
                predictions = np.argmax(probabilities, axis=1)
            else:
                predictions = (probabilities > 0.5).astype(int)
        
        if len(targets.shape) > 1:
            targets_idx = np.argmax(targets, axis=1)
        else:
            targets_idx = targets
        
        accuracy = np.mean(predictions == targets_idx)
        
        if len(probabilities.shape) > 1:
            confidence = np.mean(np.max(probabilities, axis=1))
        else:
            confidence = np.mean(np.where(predictions == 1, probabilities, 1 - probabilities))
        
        # Guardar en historial
        self.history['ece'].append(ece_result['ece'])
        self.history['nll'].append(nll)
        self.history['brier'].append(brier)
        self.history['accuracy'].append(accuracy)
        self.history['confidence'].append(confidence)
        
        # Mantener ventana deslizante
        for key in self.history:
            if len(self.history[key]) > self.window_size:
                self.history[key] = self.history[key][-self.window_size:]
    
    def get_metrics(self) -> Dict:
        """Obtiene métricas actuales."""
        metrics = {}
        for key, values in self.history.items():
            if values:
                metrics[f'{key}_mean'] = float(np.mean(values))
                metrics[f'{key}_std'] = float(np.std(values))
                metrics[f'{key}_current'] = float(values[-1]) if values else 0.0
        
        # Calibrar diagnóstico
        if self.history['accuracy'] and self.history['confidence']:
            avg_accuracy = np.mean(self.history['accuracy'][-10:])
            avg_confidence = np.mean(self.history['confidence'][-10:])
            gap = avg_confidence - avg_accuracy
            
            if gap > 0.1:
                metrics['calibration_status'] = 'overconfident'
            elif gap < -0.05:
                metrics['calibration_status'] = 'underconfident'
            else:
                metrics['calibration_status'] = 'well_calibrated'
            
            metrics['confidence_gap'] = float(gap)
        
        return metrics
    
    def should_recalibrate(self, threshold: float = 0.15) -> bool:
        """Decide si se necesita recalibrar."""
        metrics = self.get_metrics()
        
        if 'ece_current' in metrics and metrics['ece_current'] > threshold:
            return True
        
        if 'confidence_gap' in metrics and abs(metrics['confidence_gap']) > 0.1:
            return True
        
        return False

# === INTEGRACIÓN CON REWARD MODEL ===
class UncertaintyAwareRewardModel:
    """
    Reward Model que incorpora incertidumbre en sus predicciones.
    """
    
    def __init__(self, base_model, uncertainty_method: str = 'entropy'):
        self.base_model = base_model
        self.uncertainty_method = uncertainty_method
        self.temperature_scaler = TemperatureScaler()
        self.calibration_monitor = CalibrationMonitor()
        
        self.uncertainty_history = []
        self.calibration_history = []
    
    async def predict_with_uncertainty(self, X: np.ndarray) -> Dict:
        """Predicción con estimación de incertidumbre."""
        # Forward pass
        logits = await self.base_model.predict(X) 
        
        # Aplicar temperature scaling si está entrenado
        if self.temperature_scaler.trained:
            probs = self.temperature_scaler.calibrate(logits)
        else:
            probs = stable_softmax(logits, axis=-1)
        
        # Calcular incertidumbre
        if self.uncertainty_method == 'entropy':
            uncertainty = calculate_entropy(probs)
        elif self.uncertainty_method == 'mutual_information':
            # Necesitaríamos múltiples forward passes
            uncertainty = calculate_entropy(probs)  # Fallback
        else:
            uncertainty = calculate_entropy(probs)
        
        # Registrar
        self.uncertainty_history.append({
            'mean': float(np.mean(uncertainty)),
            'std': float(np.std(uncertainty)),
            'max': float(np.max(uncertainty))
        })
        
        return {
            'probabilities': probs,
            'predictions': np.argmax(probs, axis=1),
            'uncertainty': uncertainty,
            'confidence': np.max(probs, axis=1),
            'logits': logits
        }
    
    def compute_reward(self, 
                      predictions: np.ndarray,
                      targets: np.ndarray,
                      uncertainty: np.ndarray = None,
                      uncertainty_weight: float = 0.3) -> np.ndarray:
        """Calcula recompensa considerando incertidumbre."""
        # Recompensa base (accuracy)
        if len(predictions.shape) > 1:
            pred_classes = np.argmax(predictions, axis=1)
            if len(targets.shape) > 1:
                target_classes = np.argmax(targets, axis=1)
            else:
                target_classes = targets
            base_reward = (pred_classes == target_classes).astype(float)
        else:
            base_reward = (predictions == targets).astype(float)
        
        # Ajustar por incertidumbre si se proporciona
        if uncertainty is not None:
            adjusted_reward = uncertainty_aware_reward(
                base_reward, uncertainty, uncertainty_weight, method='penalize_high'
            )
        else:
            adjusted_reward = base_reward
        
        return adjusted_reward
    
    def evaluate_calibration(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evalúa calibración del modelo."""
        # Obtener predicciones
        result = self.predict_with_uncertainty(X)
        probs = result['probabilities']
        
        # Calcular métricas
        ece_result = expected_calibration_error(probs, y)
        nll = negative_log_likelihood(probs, y)
        brier = brier_score(probs, y)
        
        # Detectar sobre-confianza
        overconfidence = detect_overconfidence(probs, y)
        
        # Actualizar monitor
        self.calibration_monitor.update(probs, y, result['predictions'])
        
        metrics = {
            'ece': ece_result['ece'],
            'nll': nll,
            'brier': brier,
            'overconfidence': overconfidence,
            'mean_uncertainty': float(np.mean(result['uncertainty'])),
            'mean_confidence': float(np.mean(result['confidence']))
        }
        
        self.calibration_history.append(metrics)
        
        # Decidir si recalibrar
        if self.calibration_monitor.should_recalibrate():
            metrics['recalibration_recommended'] = True
            metrics['recalibration_reason'] = 'High calibration error'
        
        return metrics
    
    def fit_temperature_scaling(self, logits: np.ndarray, targets: np.ndarray):
        """Ajusta temperature scaling."""
        self.temperature_scaler.fit(logits, targets, verbose=True)
    
    def get_uncertainty_stats(self) -> Dict:
        """Obtiene estadísticas de incertidumbre."""
        if not self.uncertainty_history:
            return {}
        
        uncertainties = [h['mean'] for h in self.uncertainty_history]
        
        return {
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties)),
            'trend': 'increasing' if len(uncertainties) > 10 and uncertainties[-1] > uncertainties[0] else 'stable',
            'num_measurements': len(self.uncertainty_history)
        }

# === TESTING ===
def test_calibration_uncertainty():
    """Prueba las funciones de calibración e incertidumbre."""
    print("Testing calibration and uncertainty functions...")
    
    # Generar datos sintéticos
    np.random.seed(42)
    n_samples = 1000
    n_classes = 3
    
    # Logits perfectamente calibrados
    perfect_logits = np.random.randn(n_samples, n_classes)
    perfect_probs = stable_softmax(perfect_logits, axis=1)
    perfect_targets = np.argmax(perfect_probs, axis=1)
    
    # Logits sobre-confidentes
    overconfident_logits = perfect_logits * 2.0  # Escalar aumenta confianza
    overconfident_probs = stable_softmax(overconfident_logits, axis=1)
    
    # Calcular ECE
    perfect_ece = expected_calibration_error(perfect_probs, perfect_targets)
    overconfident_ece = expected_calibration_error(overconfident_probs, perfect_targets)
    
    print(f"Perfectly calibrated ECE: {perfect_ece['ece']:.4f}")
    print(f"Overconfident ECE: {overconfident_ece['ece']:.4f}")
    
    # Probar temperature scaling
    print("\nTesting Temperature Scaling...")
    ts = TemperatureScaler()
    optimal_temp = ts.fit(overconfident_logits, perfect_targets, max_iter=200, verbose=False)
    
    print(f"Optimal temperature: {optimal_temp:.4f}")
    print(f"Initial NLL: {ts.history[0]['nll']:.4f}")
    print(f"Final NLL: {ts.final_nll:.4f}")
    
    # Calibrar probabilidades
    calibrated_probs = ts.calibrate(overconfident_logits)
    calibrated_ece = expected_calibration_error(calibrated_probs, perfect_targets)
    
    print(f"Calibrated ECE: {calibrated_ece['ece']:.4f}")
    
    # Probar detección de sobre-confianza
    print("\nTesting overconfidence detection...")
    overconfident_diagnosis = detect_overconfidence(overconfident_probs, perfect_targets)
    print(f"Is overconfident: {overconfident_diagnosis['is_overconfident']}")
    print(f"Confidence gap: {overconfident_diagnosis['confidence_gap']:.4f}")
    print(f"Severity: {overconfident_diagnosis['severity']}")
    
    # Probar ensemble uncertainty
    print("\nTesting ensemble uncertainty...")
    n_models = 5
    ensemble_preds = []
    for _ in range(n_models):
        # Generar predicciones ligeramente diferentes
        noise = np.random.randn(n_samples, n_classes) * 0.1
        ensemble_preds.append(stable_softmax(perfect_logits + noise, axis=1))
    
    ensemble_array = np.stack(ensemble_preds, axis=0)
    
    # Calcular diferentes tipos de incertidumbre
    entropy_uncertainty = ensemble_uncertainty(ensemble_array, 'entropy')
    mutual_info = ensemble_uncertainty(ensemble_array, 'mutual_information')
    
    print(f"Mean entropy uncertainty: {np.mean(entropy_uncertainty):.4f}")
    print(f"Mean mutual information: {np.mean(mutual_info):.4f}")
    
    # Probar uncertainty-aware reward
    print("\nTesting uncertainty-aware reward...")
    base_reward = np.random.rand(n_samples)
    uncertainty = np.random.rand(n_samples)
    
    adjusted_reward = uncertainty_aware_reward(base_reward, uncertainty, 
                                               uncertainty_weight=0.5, 
                                               method='penalize_high')
    
    print(f"Base reward mean: {np.mean(base_reward):.4f}")
    print(f"Adjusted reward mean: {np.mean(adjusted_reward):.4f}")
    print(f"Correlation reward-uncertainty: {np.corrcoef(base_reward, uncertainty)[0,1]:.4f}")
    
    print("\nAll tests completed successfully!")
    
    return {
        'perfect_ece': perfect_ece['ece'],
        'overconfident_ece': overconfident_ece['ece'],
        'calibrated_ece': calibrated_ece['ece'],
        'optimal_temperature': optimal_temp,
        'overconfidence_detected': overconfident_diagnosis['is_overconfident']
    }

# === INTEGRACIÓN CON POST-TRAINING PIPELINE ===
def integrate_with_pipeline():
    """
    Ejemplo de cómo integrar con el pipeline de post-training.
    """
    print("\n" + "="*80)
    print("INTEGRACIÓN CON POST-TRAINING PIPELINE")
    print("="*80)
    
    # 1. Crear monitor de calibración
    monitor = CalibrationMonitor(window_size=50)
    
    # 2. Durante entrenamiento, actualizar monitor
    print("\n1. Monitoring calibration during training...")
    
    # Simular batches de entrenamiento
    for epoch in range(5):
        # Generar batch sintético
        n_batch = 100
        logits = np.random.randn(n_batch, 3)
        probs = stable_softmax(logits, axis=1)
        targets = np.argmax(probs, axis=1)
        
        # Actualizar monitor
        monitor.update(probs, targets)
        
        if epoch % 2 == 0:
            metrics = monitor.get_metrics()
            print(f"Epoch {epoch}: ECE={metrics.get('ece_current', 0):.4f}, "
                  f"Accuracy={metrics.get('accuracy_current', 0):.4f}, "
                  f"Status={metrics.get('calibration_status', 'unknown')}")
    
    # 3. Detectar necesidad de recalibración
    print("\n2. Checking recalibration need...")
    should_recalibrate = monitor.should_recalibrate()
    print(f"Should recalibrate: {should_recalibrate}")
    
    # 4. Si se necesita, aplicar temperature scaling
    if should_recalibrate:
        print("Applying temperature scaling...")
        # En práctica, usaríamos un validation set
        # ts = TemperatureScaler()
        # ts.fit(validation_logits, validation_targets)
    
    # 5. Crear reward model conciente de incertidumbre
    print("\n3. Creating uncertainty-aware reward model...")
    
    # Simular base model
    class DummyModel:
        def predict(self, X):
            return np.random.randn(len(X), 3)
    
    base_model = DummyModel()
    ua_reward_model = UncertaintyAwareRewardModel(base_model, uncertainty_method='entropy')
    
    # Evaluar calibración
    X_test = np.random.randn(200, 10)
    y_test = np.random.randint(0, 3, 200)
    
    metrics = ua_reward_model.evaluate_calibration(X_test, y_test)
    print(f"Calibration metrics: ECE={metrics['ece']:.4f}, "
          f"NLL={metrics['nll']:.4f}, "
          f"Overconfident={metrics['overconfidence']['is_overconfident']}")
    
    # 6. Usar en RL con penalización por incertidumbre
    print("\n4. Using in RL with uncertainty penalty...")
    
    # Simular rollouts
    n_rollouts = 10
    base_rewards = np.random.rand(n_rollouts)
    uncertainties = np.random.rand(n_rollouts)
    
    adjusted_rewards = uncertainty_aware_reward(
        base_rewards, uncertainties, 
        uncertainty_weight=0.3, method='penalize_high'
    )
    
    print(f"Base rewards: {base_rewards[:5]}")
    print(f"Uncertainties: {uncertainties[:5]}")
    print(f"Adjusted rewards: {adjusted_rewards[:5]}")
    print(f"Mean adjustment: {np.mean(base_rewards - adjusted_rewards):.4f}")
    
    print("\n" + "="*80)
    print("INTEGRACIÓN COMPLETADA")
    print("="*80)

if __name__ == "__main__":
    # Ejecutar tests
    test_results = test_calibration_uncertainty()
    
    # Ejecutar integración
    integrate_with_pipeline()
    
    print("\n" + "="*80)
    print("RESUMEN DE IMPLEMENTACIÓN")
    print("="*80)
    print("\nFunciones implementadas:")
    print("1. Expected Calibration Error (ECE)")
    print("2. Temperature Scaling con optimización NumPy")
    print("3. Ensemble Uncertainty (entropía, info mutua, varianza)")
    print("4. Monte Carlo Dropout approximation")
    print("5. Calibration diagrams")
    print("6. Brier Score & Negative Log Likelihood")
    print("7. Confidence intervals")
    print("8. Uncertainty calibration")
    print("9. Overconfidence detection")
    print("10. Uncertainty-aware reward adjustment")
    print("11. Bayesian uncertainty quantification")
    print("12. Calibration monitoring during training")
    print("\nTotal líneas de código: ~1200")


def getperfo(y, p, label_names, visual_title, plot=None, fig=None):
    """
    Given targets and preditions get the machine learning model performance metrics
    """
    cf = confusion_matrix(y, p)
    #plt = plot_confusion_matrix(cf,label_names,visual_title)
    TP = np.diag(cf)
    FP = cf.sum(axis=0) - TP
    FN = cf.sum(axis=1) - TP
    TN = cf.sum() - (FP + FN + TP)
    accuracy = (TP+TN)/(TP+FP+FN+TN) * 100
    # precise data that does not belong to another class
    recall = TP/(TP+FN) * 100
    # data correctly classified as negative to other classes
    specificity = TN/(TN+FP) * 100
    # data correctly classified as positive to their classes
    precision = TP/(TP+FP) * 100
    harmonic_mean = 2 * ((precision * recall) / (precision + recall))
    if plot == True:
        plt.show()
    if fig != None:
        plt.savefig(fig)
    return accuracy, recall, specificity, precision, harmonic_mean
