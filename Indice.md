# Inteligencia Artificial

## 1. Fundamentos de la Inteligencia Artificial (IA)
> **Objetivo:** Comprender qué es la IA, su evolución, sus ramas principales y la base teórica necesaria para avanzar.  

- **1.1 Introducción a la IA**  
  - Definiciones y conceptos clave  
  - Diferencia entre IA, ML y DL  
  - Tipos de IA: débil, fuerte y general  

- **1.2 Historia de la IA**  
  - Primeras ideas y pioneros  
  - Invierno(s) de la IA y renacimientos  
  - Avances recientes con Big Data y GPUs  

- **1.3 Áreas de aplicación de la IA**  
  - Visión por computadora  
  - Procesamiento de lenguaje natural (PLN)  
  - Robótica, salud, finanzas, etc.  

- **1.4 Matemáticas y Estadística en la IA**  
  - Álgebra lineal (vectores, matrices, tensores)  
  - Cálculo diferencial e integral (derivadas, gradientes)  
  - Probabilidad y estadística (distribuciones, inferencia)  
  - Optimización numérica  

- **1.5 Aspectos éticos y sociales de la IA**  
  - Sesgos en los datos y decisiones  
  - Transparencia y explicabilidad  
  - Impacto en el trabajo y la sociedad  
  - Regulaciones y políticas emergentes  

---

## 2. Aprendizaje Automático (Machine Learning - ML) Básico
> **Objetivo:** Introducir los métodos más simples de ML y aprender a preparar datos y evaluar modelos.  

- **2.1 Conceptos fundamentales**  
  - Definición de ML y ciclo de vida de un proyecto  
  - Conjuntos de datos: entrenamiento, validación y prueba  

- **2.2 Tipos de aprendizaje**  
  - Supervisado: regresión y clasificación  
  - No supervisado: clustering y reducción de dimensionalidad  
  - Aprendizaje por refuerzo: agentes y recompensas  

- **2.3 Preparación de datos (Data Preprocessing)**  
  - Limpieza de datos y manejo de valores faltantes  
  - Normalización y estandarización  
  - Codificación de variables categóricas  
  - División de datos (train/test split, cross-validation)  

- **2.4 Métricas de evaluación**  
  - Exactitud, precisión, recall, F1-score  
  - Curvas ROC y AUC  
  - Error cuadrático medio (MSE) y R²  

- **2.5 Modelos iniciales**  
  - Regresión lineal y logística  
  - k-Vecinos más cercanos (k-NN)  
  - Naïve Bayes  

---

## 3. Aprendizaje Automático (ML) Intermedio
> **Objetivo:** Dominar modelos más complejos, comprender sus fortalezas y limitaciones, y aprender a combinarlos.  

- **3.1 Modelos de frontera más sofisticados**  
  - Máquinas de Vectores de Soporte (SVM)  
  - Árboles de decisión y sus variantes  

- **3.2 Métodos de ensamble (Ensemble Methods)**  
  - Bagging y Random Forests  
  - Boosting (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost)  
  - Stacking y blending de modelos  

- **3.3 Reducción de dimensionalidad y Feature Engineering**  
  - PCA (Análisis de Componentes Principales)  
  - Selección de características  
  - Extracción de características  

- **3.4 Optimización de modelos**  
  - Validación cruzada avanzada  
  - Grid Search y Random Search  
  - Introducción a Bayesian Optimization  

---

## 4. Aprendizaje Profundo (Deep Learning - DL)
> **Objetivo:** Comprender y aplicar redes neuronales, desde las más simples hasta arquitecturas modernas.  

- **4.1 Redes Neuronales Artificiales (ANNs)**  
  - Perceptrón y perceptrón multicapa (MLP)  
  - Funciones de activación  
  - Backpropagation y descenso del gradiente  

- **4.2 Arquitecturas especializadas**  
  - Redes Convolucionales (CNNs) – visión por computadora  
  - Redes Recurrentes (RNNs, LSTM, GRU) – secuencias y series temporales  
  - Transformers – atención y modelos de lenguaje (BERT, GPT, etc.)  

- **4.3 Entrenamiento avanzado**  
  - Regularización (Dropout, Batch Normalization)  
  - Inicialización de pesos y optimizadores (SGD, Adam, RMSProp)  
  - Técnicas de aumento de datos (Data Augmentation)  

- **4.4 Optimización de hiperparámetros**  
  - Hyperparameter tuning (Optuna, Ray Tune)  
  - Early stopping y learning rate schedules  

- **4.5 Aplicaciones prácticas**  
  - Visión artificial (clasificación, detección, segmentación)  
  - Procesamiento de lenguaje natural (traducción, chatbots, embeddings)  
  - Generación de contenido (GANs, Diffusion Models)  

---

## 5. Generación y Despliegue de Modelos (Práctica Avanzada)
> **Objetivo:** Aprender a implementar modelos reales en entornos productivos, con herramientas modernas y buenas prácticas.  

- **5.1 Herramientas y frameworks**  
  - TensorFlow y Keras  
  - PyTorch  
  - Scikit-learn para prototipado rápido  

- **5.2 Flujo de trabajo de MLOps**  
  - Versionado de datos y modelos (DVC, MLflow)  
  - Pipelines de entrenamiento  
  - Pruebas y validación en ML  

- **5.3 Despliegue de modelos**  
  - Exportación de modelos (ONNX, TorchScript, SavedModel)  
  - APIs para inferencia (FastAPI, Flask)  
  - Contenedores y orquestación (Docker, Kubernetes)  

- **5.4 Fine-tuning y modelos preentrenados**  
  - Transfer learning  
  - Ajuste fino en NLP y visión  
  - Zero-shot y few-shot learning  

- **5.5 Escalabilidad y consideraciones de producción**  
  - Inferencia distribuida  
  - Aceleración con GPUs/TPUs  
  - Monitoreo y mantenimiento en producción  
