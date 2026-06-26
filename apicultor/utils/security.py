class ModelInversionAttack:
    """
    El atacante reconstruye datos de entrenamiento a partir del modelo.
    """
    
    def reconstruct_training_sample(self, target_features):
        # Inicializa ruido
        reconstructed = np.random.randn(*target_features.shape)
        
        # Optimiza para que la predicción del modelo coincida
        for _ in range(1000):
            pred = self.model.predict(reconstructed)
            loss = np.mean((pred - target_features) ** 2)
            
            # Aproximación de gradiente
            grad = (pred - target_features)
            reconstructed = reconstructed - 0.01 * grad
            
            # Si el atacante tiene acceso a memorias intermedias...
            self._extract_memory_patterns(reconstructed)
        
        return reconstructed
    
    # Understand patrones de la memoria del agente
    # Esto podría revelar información sobre datos de entrenamiento
    
        
class DataPoisoningAttack:
    """
    El atacante inyecta datos maliciosos durante el entrenamiento.
    """
    
    def create_backdoor_samples(self, X_clean, Y_clean, trigger_pattern):
        # Añade un patrón "trigger" que el modelo asociará con una salida específica
        X_poisoned = X_clean.copy()
        Y_poisoned = Y_clean.copy()
        
        for i in range(len(X_poisoned)):
            # Inyecta el trigger en una posición específica
            X_poisoned[i][:len(trigger_pattern)] += trigger_pattern
            
            # Cambia la etiqueta a lo que el atacante quiere
            Y_poisoned[i] = self.target_output
        
        return X_poisoned, Y_poisoned
    
    def create_subtle_poisoning(self, X, Y, fraction=0.01):
        # Envenena solo una pequeña fracción de datos (difícil de detectar)
        n_poison = int(len(X) * fraction)
        indices = np.random.choice(len(X), n_poison, replace=False)
        
        for idx in indices:
            # Modificaciones sutiles que sesgan el modelo gradualmente
            X[idx] = X[idx] + np.random.normal(0, 0.01, X[idx].shape)
            Y[idx] = Y[idx] + 0.1  # Sesgo pequeño
            
class AttentionAttack:
    """
    El atacante explota el mecanismo de atención para redirigir el foco.
    """
    
    def attention_redirect_attack(self, x_frame):
        # Identifica qué features tienen más atención
        attention_weights = self.get_attention_weights(x_frame)
        
        # Crea patrones que capturan la atención
        high_attention_features = np.argsort(attention_weights)[-10:]
        
        # Añade ruido específico en esas features
        x_attacked = x_frame.copy()
        for feat in high_attention_features:
            x_attacked[:, feat] += np.random.normal(0, 0.1, x_attacked.shape[0])
        
        # El modelo ahora ignorará features importantes
        return x_attacked
        
class SubstituteModelAttack:
    """
    El atacante entrena un modelo sustituto para atacar el original.
    """
    
    def train_substitute_model(self, query_limit=1000):
        substitute_model = DSVMWrapper()
        
        # Consulta el modelo original para obtener pares (input, output)
        for _ in range(query_limit):
            # Genera inputs aleatorios
            synthetic_input = np.random.randn(1, 1025)
            
            # Consulta el modelo original
            synthetic_output = self.target_model.predict(synthetic_input)
            
            # Entrena el modelo sustituto
            substitute_model.fit_model(synthetic_input, synthetic_output, ...)
        
        # Ahora el atacante tiene un modelo sustituto para generar ataques
        return substitute_model
    
    def generate_transferable_attack(self, x_frame):
        # Usa el modelo sustituto para generar ataques transferibles
        substitute_model = self.train_substitute_model()
        
        # Calcula ataque en el sustituto
        grad = substitute_model.compute_gradient(x_frame)
        
        # El ataque probablemente funcione en el original
        return x_frame + 0.1 * np.sign(grad)
        
class MemoryPoisoningAttack:
    """
    El atacante inyecta experiencias falsas en la memoria del agente.
    """
    
    def inject_false_experiences(self, agent_memory, n_fake=100):
        for _ in range(n_fake):
            # Crea experiencias falsas con alta recompensa
            fake_input = self.generate_fake_input()
            fake_output = self.generate_fake_output()
            
            # Añade a la memoria del agente
            agent_memory.add_experience(
                input_data=fake_input,
                output_data=fake_output,
                reward=0.99,  # Recompensa muy alta
                context={'constitution_score': 0.95}
            )
        
        # El agente ahora tiene recuerdos falsos que sesgan su comportamiento
        
class FeatureImportanceAttack:
    """
    El atacante manipula features importantes identificadas por el modelo.
    """
    
    def identify_important_features(self, X_sample):
        # Calcula importancia de features (como en explain.py)
        importances = []
        for i in range(X_sample.shape[1]):
            X_perturbed = X_sample.copy()
            X_perturbed[:, i] = np.random.permutation(X_perturbed[:, i])
            mse_original = self.compute_mse(X_sample)
            mse_perturbed = self.compute_mse(X_perturbed)
            importance = mse_perturbed - mse_original
            importances.append(importance)
        
        return np.argsort(importances)[-10:]  # Top 10 features más importantes
    
    def attack_important_features(self, x_frame):
        important_features = self.identify_important_features(x_frame)
        
        x_attacked = x_frame.copy()
        for feat in important_features:
            # Perturba fuertemente las features importantes
            x_attacked[:, feat] = 0  # Zero out
            # o
            x_attacked[:, feat] = np.random.randn() * 10
        
        return x_attacked
                                                                     