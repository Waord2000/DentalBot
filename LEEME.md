# DentalBot - Sistema Híbrido de Clasificación de Anatomía Dental con IA

## Descripción
DentalBot es un sistema inteligente que combina Machine Learning tradicional (Random Forest) con Inteligencia Artificial generativa (Groq API) para clasificar y responder preguntas sobre anatomía dental. El sistema utiliza un enfoque híbrido que integra:

- **Detector de Términos Dentales**: Diccionario especializado con 150+ términos de odontología
- **Clasificador Random Forest**: Modelo ML entrenado con TF-IDF para clasificación binaria
- **Generador de Respuestas IA**: Modelo LLM pre-entrenado (Llama 3.3) vía Groq API

## Características Principales

### Sistema Híbrido de Clasificación
- Detección rápida mediante diccionario de términos especializados
- Clasificación ML con Random Forest como respaldo
- Combinación inteligente de ambos métodos para máxima precisión

### Generación de Respuestas con IA
- Utiliza Groq API con modelo Llama 3.3 (70B parámetros)
- Respuestas contextualizadas y educativas
- No requiere entrenamiento manual ni dataset de respuestas

### Métricas y Evaluación
- Matriz de confusión detallada
- Métricas por clase (Precision, Recall, F1-Score)
- Reporte completo guardado en archivo de texto

### Dependencias
Ver archivo `Requerimientos.txt`

## Instalación y Configuración

### PASO 1: Clonar o Descargar el Proyecto
```bash
git clone [https://github.com/Waord2000/DentalBot]
cd dentalbot
```

### PASO 2: Crear y Activar Entorno Virtual

**En Windows (CMD):**
```bash
python -m venv entorno_virtual
entorno_virtual\Scripts\activate
```

**En Windows (PowerShell):**
```bash
python -m venv entorno_virtual
.\entorno_virtual\Scripts\Activate.ps1
```

**En Linux/Mac:**
```bash
python3 -m venv entorno_virtual
source entorno_virtual/bin/activate
```

### PASO 3: Instalar Dependencias
```bash
pip install -r Requerimientos.txt
```

### PASO 4: Configurar Recursos NLTK
```bash
python configurar_nltk.py
```

### PASO 5: Entrenar el Modelo
```bash
python entrenar_modelo.py
```
Este paso genera:
- Modelo entrenado en `modelos/dentalbot_rf.pkl`
- Reporte de métricas en `resultados/metricas_evaluacion.txt`

### PASO 6: Ejecutar el Sistema
```bash
python ejecutar_dentalbot.py
```

## Uso del Sistema

### Comandos Disponibles

Una vez iniciado el sistema, puedes usar:

- **Hacer preguntas**: Escribe directamente tu pregunta sobre anatomía dental
- **`estadisticas`**: Muestra información detallada del modelo y diccionario
- **`pruebas`**: Ejecuta casos de prueba predefinidos
- **`salir`**: Cierra el sistema

### Ejemplos de Preguntas

**Preguntas sobre Anatomía (SÍ Dental):**
- "¿Qué características tiene el incisivo central superior?"
- "¿Cuántas cúspides tiene el primer molar?"
- "¿Qué es un bracket en ortodoncia?"
- "Morfología del canino inferior"

**Preguntas NO Dentales:**
- "¿Cómo hacer una obturación?" (procedimiento clínico)
- "¿Qué es la diabetes?" (tema médico no dental)
- "¿Capital de Francia?" (tema no relacionado)

## Estructura del Proyecto

```
dentalbot/
│
├── codigo/                          # Código fuente
│   ├── __init__.py
│   ├── clasificador.py             # Random Forest
│   ├── configuracion.py            # Parámetros del sistema
│   ├── dentalbot.py                # Sistema principal
│   ├── detector_terminos_dentales.py  # Detector de términos
│   ├── generador_respuestas_ia.py  # Integración con Groq API
│   └── preprocesador.py            # Preprocesamiento de texto
│
├── datos/                           # Datasets
│   └── preguntas_dentales.csv      # Dataset de entrenamiento
│
├── modelos/                         # Modelos entrenados
│   └── dentalbot_rf.pkl            # Modelo Random Forest
│
├── resultados/                      # Resultados y métricas
│   └── metricas_evaluacion.txt     # Reporte de evaluación
│
├── .gitignore                       # Archivos a ignorar en Git
├── LEEME.md                         # Este archivo
├── Requerimientos.txt               # Dependencias del proyecto
├── configurar_nltk.py               # Configuración de NLTK
├── entrenar_modelo.py               # Script de entrenamiento
└── ejecutar_dentalbot.py            # Script principal de ejecución
```

## Metodología y Arquitectura

### Componente 1: Sistema Híbrido de Clasificación
**Detector de Términos + Random Forest**
- Primero analiza con diccionario especializado (150+ términos)
- Si hay incertidumbre, usa clasificador Random Forest
- Combina resultados para decisión final

**Input**: Pregunta del usuario  
**Output**: [Anatomía Dental: Sí/No] + Nivel de confianza

### Componente 2: Generador de Respuestas con IA
**Modelo LLM Pre-entrenado (Groq API - Llama 3.3)**
- Conocimiento médico integrado del modelo base
- No requiere entrenamiento adicional
- Especialización mediante prompts contextuales

**Ventaja**: Utiliza conocimiento de miles de artículos científicos ya incorporado en el LLM

### Componente 3: Interfaz de Usuario
**CLI (Command Line Interface)**
- Interfaz de línea de comandos interactiva
- Comandos simples e intuitivos
- Resultados detallados con análisis completo

## Métricas del Modelo
El sistema genera métricas completas que incluyen:

- **Accuracy General**: Precisión global del modelo
- **Matriz de Confusión**: Verdaderos/Falsos Positivos y Negativos
- **Métricas por Clase**:
  - Precision: Proporción de predicciones correctas
  - Recall: Proporción de casos reales detectados
  - F1-Score: Media armónica de Precision y Recall
- **Características Importantes**: Top 10 términos más relevantes

## Desactivar Entorno Virtual

Cuando termines de trabajar:
```bash
deactivate
```

## Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje de programación
- **scikit-learn**: Machine Learning (Random Forest, TF-IDF)
- **NLTK**: Procesamiento de lenguaje natural
- **Groq API**: Generación de respuestas con IA
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas

## Limitaciones Conocidas

- Requiere conexión a internet para generar respuestas con IA
- El modelo se especializa en anatomía dental, no en procedimientos clínicos
- La API de Groq tiene límites de uso 

## Licencia

Este proyecto es de código abierto y está disponible bajo la Licencia MIT.

## Autor

**Walter Alexander Ordoñez Ramos**

