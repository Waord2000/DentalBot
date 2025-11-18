# DentalBot - Clasificador de Anatomía Dental

## Descripción
DentalBot es un sistema de inteligencia artificial que clasifica preguntas sobre anatomía dental utilizando un modelo de Random Forest y procesamiento de texto especializado.

## Instalación y Configuración

### PASO 1: ABRIR TERMINAL O CMD

### PASO 2: CREAR Y ACTIVAR ENTORNO VIRTUAL
# Crear entorno virtual (si no existe)
python -m venv entorno_virtual

# Activar entorno virtual en CMD:
entorno_virtual\Scripts\activate

# Activar entorno virtual en PowerShell:
.\entorno_virtual\Scripts\Activate.ps1

### PASO 3: INSTALAR DEPENDENCIAS
pip install -r Requerimientos.txt

### PASO 4: ENTRENAR EL MODELO
python configurar_nltk.py
python entrenar_modelo.py

### PASO 5: EJECUTAR EL BOT
python ejecutar_dentalbot.py

## Cerrar Entorno Virtual
Cuando termines de trabajar con el proyecto, puedes cerrar el entorno virtual ejecutando:
deactivate

## Uso
### Entrenamiento del modelo
Ejecuta `python entrenar_modelo.py` para entrenar el modelo. Los datos de entrenamiento deben estar en `datos/preguntas_dentales.csv`.

### Ejecutar el bot
Ejecuta `python ejecutar_dentalbot.py` para interactuar con el bot.

## Estructura del proyecto
- `codigo/`: Módulo con el código fuente.
- `datos/`: Dataset con preguntas dentales y no dentales.

## Autor
Walter Alexander Ordoñez Ramos