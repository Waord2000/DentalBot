import pandas as pd
import numpy as np
import os

from configuracion import ConfiguracionDentalBot
from preprocesador import ProcesadorTexto
from clasificador import ClasificadorDentalRandomForest
from generador_respuestas_ia import GeneradorRespuestasIA

class DentalBot:
    """Sistema DentalBot con IA para respuestas"""
    
    def __init__(self):
        self.configuracion = ConfiguracionDentalBot()
        self.procesador = ProcesadorTexto()
        self.clasificador = ClasificadorDentalRandomForest(self.configuracion)
        self.generador_respuestas = GeneradorRespuestasIA()  # Ya tiene la API key
        self.entrenado = False
        
    def cargar_datos_ejemplo(self):
        """Datos mínimos de ejemplo"""
        preguntas_dentales = [
            "¿Características del incisivo central superior?",
            "¿Morfología de la corona del incisivo lateral?",
            "¿Diferencias entre incisivo central y lateral?",
            "¿Cúspides del primer premolar superior?",
            "¿Anatomía del canino superior?",
            "¿Cuántas raíces tiene el primer molar superior?",
            "¿Qué es la cúspide de Carabelli?",
            "¿Características del segundo premolar inferior?",
        ]
        
        preguntas_no_dentales = [
            "¿Cómo hacer una obturación?",
            "¿Qué es la diabetes?",
            "¿Capital de Francia?",
            "¿Cómo tratar caries?",
            "¿Síntomas de hipertensión?",
            "¿Qué es un tratamiento de conducto?",
            "¿Cómo blanquear los dientes?",
            "¿Pasos para una extracción dental?",
        ]
        
        preguntas = preguntas_dentales + preguntas_no_dentales
        etiquetas = [1] * len(preguntas_dentales) + [0] * len(preguntas_no_dentales)
        
        return preguntas, etiquetas

    def cargar_datos_desde_csv(self, ruta_csv=None):
        """Carga datos desde CSV (OPCIONAL)"""
        if ruta_csv is None:
            ruta_csv = os.path.join('datos', 'preguntas_dentales.csv')
        
        try:
            if not os.path.exists(ruta_csv):
                print(f"CSV no encontrado: {ruta_csv}")
                print("Usando datos de ejemplo...")
                return self.cargar_datos_ejemplo()
            
            df = pd.read_csv(ruta_csv)
            print(f"Datos cargados desde CSV: {len(df)} preguntas")
            preguntas = df['pregunta'].tolist()
            etiquetas = df['es_anatomia_dental'].tolist()
            
            return preguntas, etiquetas
            
        except Exception as e:
            print(f"Error cargando CSV: {e}")
            print("Usando datos de ejemplo...")
            return self.cargar_datos_ejemplo()
    
    def entrenar_modelo(self, usar_csv=True, ruta_csv=None):
        """Entrena el clasificador"""
        if usar_csv:
            print("Intentando cargar desde CSV...")
            preguntas, etiquetas = self.cargar_datos_desde_csv(ruta_csv)
        else:
            print("Usando datos de ejemplo")
            preguntas, etiquetas = self.cargar_datos_ejemplo()
        
        print(f"Total: {len(preguntas)} preguntas")
        print(f"Distribución: {sum(etiquetas)} dentales, {len(etiquetas)-sum(etiquetas)} no dentales")
        
        print("Preprocesando...")
        preguntas_procesadas = self.procesador.preprocesar_lote(preguntas)
        
        print("Preparando datos...")
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = self.clasificador.preparar_datos(
            preguntas_procesadas, etiquetas
        )
        
        self.clasificador.entrenar(X_entrenamiento, y_entrenamiento)
        self.entrenado = True
        
        print("Evaluando modelo...")
        precision, reporte, predicciones = self.clasificador.evaluar(X_prueba, y_prueba, mostrar_metricas=False)
        
        print(f"Precisión: {precision:.3f}")
        
        return precision, reporte
    
    def clasificar_pregunta(self, pregunta):
        """Clasifica pregunta y genera respuesta con IA"""
        if not self.entrenado:
            return {'error': 'Modelo no entrenado. Ejecuta entrenar_modelo() primero.'}
        
        # Validación de entrada
        if not pregunta or len(pregunta.strip()) < 3:
            return {
                'pregunta_original': pregunta,
                'es_anatomia_dental': False,
                'probabilidad_dental': 0.0,
                'respuesta': 'Por favor, ingresa una pregunta válida sobre anatomía dental.',
                'confianza': 'Baja'
            }
        
        pregunta_procesada = self.procesador.preprocesar(pregunta)
        
        # Si después del preprocesamiento queda vacío, no es dental
        if not pregunta_procesada or len(pregunta_procesada.strip()) < 2:
            return {
                'pregunta_original': pregunta,
                'es_anatomia_dental': False,
                'probabilidad_dental': 0.0,
                'respuesta': 'Lo siento, no pude identificar términos relacionados con anatomía dental en tu pregunta. ¿Podrías reformularla?',
                'confianza': 'Baja'
            }
        
        es_dental = self.clasificador.predecir([pregunta_procesada])[0]
        probabilidad = self.clasificador.predecir_probabilidad([pregunta_procesada])[0]
        
        # UMBRAL DE CONFIANZA: Si la probabilidad es baja, clasificar como NO dental
        UMBRAL_MINIMO = 0.55  # Ajustable
        
        if probabilidad[1] < UMBRAL_MINIMO and es_dental == 1:
            # Forzar a NO dental si la confianza es muy baja
            es_dental = 0
            
        # GENERAR RESPUESTA CON IA
        respuesta = self.generador_respuestas.generar_respuesta(pregunta, bool(es_dental))
        
        return {
            'pregunta_original': pregunta,
            'es_anatomia_dental': bool(es_dental),
            'probabilidad_dental': float(probabilidad[1]),
            'respuesta': respuesta,
            'confianza': 'Alta' if probabilidad[1] > 0.8 else 'Media' if probabilidad[1] > 0.6 else 'Baja'
        }
    
    def obtener_estadisticas(self):
        """Estadísticas del modelo"""
        if not self.entrenado:
            return {'error': 'Modelo no entrenado'}
        
        caracteristicas_importantes = self.clasificador.obtener_caracteristicas_importantes(10)
        
        return {
            'caracteristicas_importantes': caracteristicas_importantes,
            'configuracion': str(self.configuracion),
            'entrenado': self.entrenado
        }
    
    def guardar_modelo(self, ruta='modelos/dentalbot_rf.pkl'):
        """Guarda modelo entrenado"""
        if self.entrenado:
            os.makedirs(os.path.dirname(ruta), exist_ok=True)
            self.clasificador.guardar_modelo(ruta)
        else:
            print("No se puede guardar: modelo no entrenado")
    
    def cargar_modelo(self, ruta='modelos/dentalbot_rf.pkl'):
        """Carga modelo entrenado"""
        if not os.path.exists(ruta):
            print(f"Modelo no encontrado en: {ruta}")
            return False
        
        try:
            self.clasificador = self.clasificador.cargar_modelo(ruta)
            self.entrenado = True
            print("Modelo cargado correctamente")
            return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False