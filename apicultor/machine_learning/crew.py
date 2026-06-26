class AgentCrewMember:
    """
    Miembro individual del crew (un agente con su modelo y memoria local)
    """
    def __init__(self, agent_id: int, model: DSVMWrapper, memory: AgentMemory, 
                 baseline_dir: str, role: str = "worker"):
        self.agent_id = agent_id
        self.model = model
        self.memory = memory
        self.baseline_dir = baseline_dir
        self.role = role  # "worker", "validator", "teacher"
        self.performance_score = 0.0
        self.consensus_weight = 1.0
        self.hallucination_risk = 0.0
        self.last_prediction = None
        self.prediction_confidence = 0.0
        
    async def predict_with_consensus(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Predice con métrica de confianza para consenso"""
        pred = self.model.predict(X)
        
        # Calcular confianza basada en:
        # 1. Certeza del modelo (si tiene proba)
        if hasattr(self.model, 'proba') and self.model.proba is not None:
            confidence = float(np.mean(self.model.proba))
        else:
            # 2. Consistencia temporal (si hay historial)
            if self.last_prediction is not None:
                similarity = 1.0 / (1.0 + np.linalg.norm(pred - self.last_prediction))
                confidence = 0.5 + 0.5 * similarity
            else:
                confidence = 0.5
        
        self.last_prediction = pred
        self.prediction_confidence = confidence
        return pred, confidence
    
    def update_hallucination_risk(self, reward: float, constitution_score: float):
        """Actualiza el riesgo de hallucinación del agente"""
        # Bajo reward + bajo constitution_score = alta hallucinación
        if reward < 0.5 and constitution_score < 0.6:
            self.hallucination_risk = min(1.0, self.hallucination_risk + 0.1)
        elif reward > 0.8 and constitution_score > 0.8:
            self.hallucination_risk = max(0.0, self.hallucination_risk - 0.05)
        
        # Actualizar peso de consenso (inverso al riesgo)
        self.consensus_weight = 1.0 - self.hallucination_risk


class GlobalMemory:
    """
    Memoria global que unifica todas las memorias de todos los agentes.
    Es el teacher definitivo que consolida conocimiento de todo el crew.
    """
    def __init__(self, max_size: int = 50000, consolidation_threshold: int = 3):
        self.max_size = max_size
        self.consolidation_threshold = consolidation_threshold
        self.global_entries: List[MemoryEntry] = []
        self.agent_memories: Dict[int, AgentMemory] = {}
        self.consensus_patterns = {}  # Patrones acordados por múltiples agentes
        self.contradiction_log = []   # Registrar contradicciones entre agentes
        self.consensus_history = []
        
    def register_agent(self, agent_id: int, agent_memory: AgentMemory):
        """Registra la memoria de un agente en el sistema global"""
        self.agent_memories[agent_id] = agent_memory
        
    def consolidate_from_agents(self):
        """
        Consolida conocimiento de todos los agentes.
        Solo se consolidan patrones que tienen consenso entre múltiples agentes.
        """
        # Recopilar todas las experiencias significativas de todos los agentes
        all_candidates = []
        
        for agent_id, memory in self.agent_memories.items():
            # Tomar episodios significativos de memoria a mediano plazo
            significant = memory.mid_term.get_significant_episodes()
            for episode in significant:
                all_candidates.append({
                    'agent_id': agent_id,
                    'data': episode.data,
                    'metadata': episode.metadata,
                    'importance': episode.importance_score
                })
            
            # También tomar patrones consolidados de largo plazo
            for pattern_hash, pattern_data in memory.long_term.pattern_library.items():
                all_candidates.append({
                    'agent_id': agent_id,
                    'data': pattern_data['pattern'],
                    'metadata': pattern_data['metadata'],
                    'importance': pattern_data['consolidations'] / 10.0,
                    'is_pattern': True,
                    'pattern_hash': pattern_hash
                })
        
        # Agrupar por similitud para detectar consenso
        similar_groups = self._group_similar_experiences(all_candidates)
        
        # Solo consolidar grupos con suficiente consenso (múltiples agentes)
        for group in similar_groups:
            if len(set([c['agent_id'] for c in group])) >= self.consolidation_threshold:
                # Hay consenso entre múltiples agentes
                consensus_pattern = self._create_consensus_pattern(group)
                
                # Calcular hash del patrón consensuado
                pattern_hash = hashlib.md5(consensus_pattern['data'].tobytes()).hexdigest()[:16]
                
                if pattern_hash not in self.consensus_patterns:
                    self.consensus_patterns[pattern_hash] = consensus_pattern
                    self.consensus_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'pattern_hash': pattern_hash,
                        'n_agents': len(set([c['agent_id'] for c in group])),
                        'avg_importance': np.mean([c['importance'] for c in group])
                    })
                    logger.info(f"[GlobalMemory] Nuevo patrón consensuado: {pattern_hash} "
                               f"({len(set([c['agent_id'] for c in group]))} agentes)")
                
        # Limpiar contradicciones (patrones que se contradicen)
        self._resolve_contradictions()
        
    def _group_similar_experiences(self, candidates: List[Dict]) -> List[List[Dict]]:
        """Agrupa experiencias similares usando clustering simple"""
        if not candidates:
            return []
        
        groups = []
        used = set()
        
        for i, candidate in enumerate(candidates):
            if i in used:
                continue
            
            group = [candidate]
            used.add(i)
            
            for j, other in enumerate(candidates):
                if j in used:
                    continue
                
                # Calcular similitud coseno
                sim = self._cosine_similarity(candidate['data'], other['data'])
                
                if sim > 0.85:  # Umbral de similitud para consenso
                    group.append(other)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores"""
        a_flat = a.flatten()
        b_flat = b.flatten()
        min_len = min(len(a_flat), len(b_flat))
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]
        
        norm_product = np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-9
        return float(np.dot(a_flat, b_flat) / norm_product)
    
    def _create_consensus_pattern(self, group: List[Dict]) -> Dict:
        """Crea un patrón de consenso promediando experiencias similares"""
        # Promediar datos
        data_sum = np.zeros_like(group[0]['data'])
        weights = []
        
        for item in group:
            weight = item['importance']
            data_sum += item['data'] * weight
            weights.append(weight)
        
        consensus_data = data_sum / (sum(weights) + 1e-9)
        
        # Combinar metadata
        combined_metadata = {
            'n_agents': len(set([item['agent_id'] for item in group])),
            'avg_reward': np.mean([item['metadata'].get('reward', 0) for item in group]),
            'avg_constitution': np.mean([item['metadata'].get('constitution_score', 0) for item in group]),
            'consensus_timestamp': datetime.now().isoformat(),
            'source_agents': list(set([item['agent_id'] for item in group]))
        }
        
        return {
            'data': consensus_data,
            'metadata': combined_metadata,
            'importance': np.mean([item['importance'] for item in group])
        }
    
    def _resolve_contradictions(self):
        """Resuelve contradicciones entre patrones consensuados"""
        # Detectar patrones que se contradicen (baja similitud pero mismo contexto)
        patterns = list(self.consensus_patterns.values())
        
        for i, p1 in enumerate(patterns):
            for j, p2 in enumerate(patterns):
                if i >= j:
                    continue
                
                sim = self._cosine_similarity(p1['data'], p2['data'])
                
                # Si son muy diferentes pero mismo contexto, hay contradicción
                if sim < 0.3:
                    self.contradiction_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'pattern_1': i,
                        'pattern_2': j,
                        'similarity': sim,
                        'resolution': 'pending'
                    })
                    
                    # Resolver: mantener el de mayor consenso (más agentes)
                    if p1['metadata']['n_agents'] > p2['metadata']['n_agents']:
                        # p1 gana, marcar p2 como obsoleto
                        self.consensus_patterns.pop(list(self.consensus_patterns.keys())[j], None)
                        logger.info(f"[GlobalMemory] Contradicción resuelta: patrón {j} reemplazado por {i}")
                    elif p2['metadata']['n_agents'] > p1['metadata']['n_agents']:
                        self.consensus_patterns.pop(list(self.consensus_patterns.keys())[i], None)
                        logger.info(f"[GlobalMemory] Contradicción resuelta: patrón {i} reemplazado por {j}")
    
    def get_global_context(self, query: np.ndarray, top_k: int = 5) -> np.ndarray:
        """Recupera contexto global basado en patrones consensuados"""
        if not self.consensus_patterns:
            return np.array([])
        
        similarities = []
        query_flat = query.flatten()
        
        for pattern_hash, pattern in self.consensus_patterns.items():
            pattern_flat = pattern['data'].flatten()
            min_len = min(len(query_flat), len(pattern_flat))
            
            sim = self._cosine_similarity(query_flat[:min_len], pattern_flat[:min_len])
            # Ponderar por importancia del consenso
            weighted_sim = sim * pattern['importance']
            similarities.append((weighted_sim, pattern['data']))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        if not similarities:
            return np.array([])
        
        # Promediar top_k patrones más similares
        top_patterns = [p for _, p in similarities[:top_k]]
        if top_patterns:
            # Asegurar misma forma
            target_shape = query.shape
            avg_pattern = np.mean(top_patterns, axis=0)
            if avg_pattern.shape != target_shape:
                avg_pattern = avg_pattern.reshape(target_shape)
            return avg_pattern
        
        return np.array([])
    
    def get_metrics(self) -> Dict:
        """Métricas de la memoria global"""
        return {
            'n_consensus_patterns': len(self.consensus_patterns),
            'n_agents': len(self.agent_memories),
            'n_contradictions': len(self.contradiction_log),
            'avg_consensus_agents': np.mean([p['metadata']['n_agents'] 
                                            for p in self.consensus_patterns.values()]) if self.consensus_patterns else 0,
            'consolidation_threshold': self.consolidation_threshold
        }


class HallucinationGuard:
    """
    Guardia contra hallucinations que valida predicciones antes de aceptarlas.
    """
    def __init__(self, global_memory: GlobalMemory, 
                 similarity_threshold: float = 0.7,
                 max_deviation: float = 0.3):
        self.global_memory = global_memory
        self.similarity_threshold = similarity_threshold
        self.max_deviation = max_deviation
        self.rejected_predictions = []
        
    async def validate(self, prediction: np.ndarray, context: np.ndarray, 
                       input_data: np.ndarray) -> Tuple[bool, float, str]:
        """
        Valida si una predicción es confiable o es una hallucinación.
        Returns: (is_valid, confidence_score, reason)
        """
        # 1. Verificar contra patrones consensuados globales
        global_context = self.global_memory.get_global_context(input_data)
        
        if global_context.size > 0:
            # Calcular similitud con contexto global
            global_sim = self._cosine_similarity(prediction, global_context)
            
            if global_sim < self.similarity_threshold:
                return False, global_sim, f"Deviation from global consensus (sim={global_sim:.3f})"
        
        # 2. Verificar consistencia interna (no varianza colapsada)
        pred_std = np.std(prediction)
        if pred_std < 1e-6:
            return False, 0.0, "Collapsed variance (hallucination detected)"
        
        # 3. Verificar energía anómala
        pred_energy = np.mean(prediction ** 2)
        input_energy = np.mean(input_data ** 2)
        
        if pred_energy > input_energy * 5:  # Amplificación excesiva
            return False, 0.2, f"Excessive amplification ({pred_energy/input_energy:.1f}x)"
        
        if pred_energy < input_energy * 0.01:  # Energía colapsada
            return False, 0.1, "Energy collapse"
        
        # 4. Verificar contra historial de rechazos (no repetir mismos errores)
        for rejected in self.rejected_predictions[-10:]:
            sim = self._cosine_similarity(prediction, rejected['prediction'])
            if sim > 0.9:
                return False, 0.1, "Similar to previously rejected hallucination"
        
        # Pasa todas las validaciones
        confidence = min(1.0, global_sim if global_context.size > 0 else 0.8)
        return True, confidence, "Validation passed"
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similitud coseno"""
        a_flat = a.flatten()
        b_flat = b.flatten()
        min_len = min(len(a_flat), len(b_flat))
        a_flat = a_flat[:min_len]
        b_flat = b_flat[:min_len]
        
        norm_product = np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-9
        return float(np.dot(a_flat, b_flat) / norm_product)
    
    def record_rejection(self, prediction: np.ndarray, reason: str):
        """Registra una predicción rechazada para aprendizaje futuro"""
        self.rejected_predictions.append({
            'prediction': prediction,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
        # Mantener tamaño limitado
        if len(self.rejected_predictions) > 100:
            self.rejected_predictions = self.rejected_predictions[-100:]


class AgentCrew:
    """
    Crew de agentes: orquesta múltiples agentes, gestiona memoria global
    y protege contra hallucinations.
    """
    def __init__(self, output_dir: str, consensus_threshold: int = 3):
        self.output_dir = output_dir
        self.consensus_threshold = consensus_threshold
        self.members: Dict[int, AgentCrewMember] = {}
        self.global_memory = GlobalMemory(consolidation_threshold=consensus_threshold)
        self.hallucination_guard = HallucinationGuard(self.global_memory)
        self.crew_leader = None  # El mejor agente (teacher de todos)
        self.consensus_log = []
        
    def add_member(self, agent_id: int, model: DSVMWrapper, memory: AgentMemory,
                   baseline_dir: str, role: str = "worker"):
        """Agrega un nuevo miembro al crew"""
        member = AgentCrewMember(agent_id, model, memory, baseline_dir, role)
        self.members[agent_id] = member
        self.global_memory.register_agent(agent_id, memory)
        logger.info(f"[AgentCrew] Miembro {agent_id} agregado (role={role})")
        
    def load_baseline_as_member(self, baseline_dir: str, agent_id: int = None):
        """Carga una baseline existente como miembro del crew"""
        # Cargar modelo
        model_path = os.path.join(baseline_dir, "models", "dsvm_main.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            logger.warning(f"[AgentCrew] No se encontró modelo en {baseline_dir}")
            return None
        
        # Cargar memoria
        memory_path = os.path.join(baseline_dir, "memories", "memory_agent_memory.pkl")
        if os.path.exists(memory_path):
            with open(memory_path, 'rb') as f:
                memory_data = pickle.load(f)
                memory = AgentMemory()
                memory.long_term = memory_data['long_term']
                memory.mid_term = memory_data['mid_term']
                memory.short_term = memory_data['short_term']
                memory.memory_stats = memory_data['memory_stats']
        else:
            memory = AgentMemory()
        
        if agent_id is None:
            agent_id = len(self.members)
        
        self.add_member(agent_id, model, memory, baseline_dir, "worker")
        return agent_id
    
    def load_all_baselines(self, baselines_dir: str):
        """Carga todas las baselines de un directorio como miembros del crew"""
        baseline_dirs = [d for d in os.listdir(baselines_dir) 
                        if d.startswith("baseline_") and os.path.isdir(os.path.join(baselines_dir, d))]
        
        for i, bdir in enumerate(baseline_dirs):
            full_path = os.path.join(baselines_dir, bdir)
            self.load_baseline_as_member(full_path, i)
            
        logger.info(f"[AgentCrew] Cargados {len(self.members)} miembros desde {baselines_dir}")
        
    async def consensus_prediction(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Predicción por consenso de todos los miembros del crew.
        """
        predictions = []
        confidences = []
        weights = []
        
        for agent_id, member in self.members.items():
            pred, confidence = await member.predict_with_consensus(X)
            
            # Validar contra hallucination guard
            is_valid, val_confidence, reason = await self.hallucination_guard.validate(
                pred, None, X
            )
            
            if is_valid:
                predictions.append(pred)
                # Ponderar por: confianza del modelo * peso de consenso * validación
                weight = confidence * member.consensus_weight * val_confidence
                weights.append(weight)
                confidences.append(confidence)
            else:
                logger.debug(f"[AgentCrew] Miembro {agent_id} rechazado: {reason}")
                self.hallucination_guard.record_rejection(pred, reason)
        
        if not predictions:
            # Fallback: usar el crew leader si existe
            if self.crew_leader:
                pred = self.crew_leader.model.predict(X)
                return pred, {'fallback': 'crew_leader', 'n_members': 0}
            else:
                raise RuntimeError("No hay predicciones válidas y no hay crew leader")
        
        # Predicción por consenso ponderado
        weights = np.array(weights) / (sum(weights) + 1e-9)
        consensus_pred = np.zeros_like(predictions[0])
        
        for pred, weight in zip(predictions, weights):
            consensus_pred += pred * weight
        
        # Registrar consenso
        self.consensus_log.append({
            'timestamp': datetime.now().isoformat(),
            'n_members': len(predictions),
            'avg_confidence': np.mean(confidences),
            'consensus_weight_std': np.std(weights)
        })
        
        return consensus_pred, {
            'n_members': len(predictions),
            'avg_confidence': float(np.mean(confidences)),
            'weights': weights.tolist(),
            'consensus_reached': len(predictions) >= self.consensus_threshold
        }
    
    async def consolidate_crew_memory(self):
        """Consolida la memoria de todos los miembros en la memoria global"""
        self.global_memory.consolidate_from_agents()
        
        # Actualizar crew leader (el de mayor rendimiento)
        best_member = max(self.members.values(), 
                         key=lambda m: m.performance_score - m.hallucination_risk)
        self.crew_leader = best_member
        
        logger.info(f"[AgentCrew] Crew leader actualizado: agente {best_member.agent_id}, "
                   f"score={best_member.performance_score:.4f}, "
                   f"hallucination_risk={best_member.hallucination_risk:.4f}")
        
        # Distribuir patrones consensuados a todos los miembros
        for member in self.members.values():
            # Los miembros pueden aprender de la memoria global
            member.memory.long_term.pattern_library.update(
                self.global_memory.consensus_patterns
            )
    
    def get_crew_metrics(self) -> Dict:
        """Métricas completas del crew"""
        return {
            'n_members': len(self.members),
            'consensus_threshold': self.consensus_threshold,
            'global_memory': self.global_memory.get_metrics(),
            'hallucination_guard': {
                'rejected_count': len(self.hallucination_guard.rejected_predictions)
            },
            'crew_leader': {
                'agent_id': self.crew_leader.agent_id if self.crew_leader else None,
                'performance': self.crew_leader.performance_score if self.crew_leader else 0,
                'hallucination_risk': self.crew_leader.hallucination_risk if self.crew_leader else 0
            } if self.crew_leader else None,
            'recent_consensus': self.consensus_log[-10:] if self.consensus_log else []
        }