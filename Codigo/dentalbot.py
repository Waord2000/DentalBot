import pandas as pd
import numpy as np
import os

from configuracion import ConfiguracionDentalBot
from preprocesador import ProcesadorTexto
from clasificador import ClasificadorDentalRandomForest
from generador_respuestas_ia import GeneradorRespuestasIA
from detector_terminos_dentales import DetectorTerminosDentales 

class DentalBot:
    """Sistema DentalBot con IA + Detector de Términos"""
    
    def __init__(self):
        self.configuracion = ConfiguracionDentalBot()
        self.procesador = ProcesadorTexto()
        self.clasificador = ClasificadorDentalRandomForest(self.configuracion)
        self.generador_respuestas = GeneradorRespuestasIA()
        self.detector_terminos = DetectorTerminosDentales()  
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
        """
        Clasificación HÍBRIDA: Detector de Términos + Modelo ML
        
        ESTRATEGIA:
        1. Primero usa el detector de términos (rápido, preciso para términos conocidos)
        2. Si el detector no está seguro, usa el modelo ML
        3. Combina ambos resultados para decisión final
        """
        if not self.entrenado:
            return {'error': 'Modelo no entrenado. Ejecuta entrenar_modelo() primero.'}
        
        # Validación de entrada
        if not pregunta or len(pregunta.strip()) < 3:
            return {
                'pregunta_original': pregunta,
                'es_anatomia_dental': False,
                'probabilidad_dental': 0.0,
                'respuesta': 'Por favor, ingresa una pregunta válida sobre anatomía dental.',
                'confianza': 'Baja',
                'metodo': 'Validación'
            }

        deteccion = self.detector_terminos.es_termino_dental(pregunta)
        
        # Si el detector tiene ALTA confianza, confiar en él directamente
        if deteccion['confianza'] == 'Alta':
            es_dental = deteccion['es_dental']
            
            # Generar respuesta con IA
            respuesta = self.generador_respuestas.generar_respuesta(pregunta, es_dental)
            
            return {
                'pregunta_original': pregunta,
                'es_anatomia_dental': es_dental,
                'probabilidad_dental': 0.95 if es_dental else 0.05,
                'respuesta': respuesta,
                'confianza': 'Alta',
                'metodo': 'Detector de Términos',
                'terminos_encontrados': deteccion['terminos_encontrados'][:5],
                'razon': deteccion['razon']
            }

        pregunta_procesada = self.procesador.preprocesar(pregunta)
        
        # Si después del preprocesamiento queda vacío
        if not pregunta_procesada or len(pregunta_procesada.strip()) < 2:
            return {
                'pregunta_original': pregunta,
                'es_anatomia_dental': False,
                'probabilidad_dental': 0.0,
                'respuesta': 'Lo siento, no pude identificar términos relacionados con anatomía dental en tu pregunta. ¿Podrías reformularla?',
                'confianza': 'Baja',
                'metodo': 'Preprocesamiento'
            }
        
        # Clasificar con el modelo ML
        es_dental_ml = self.clasificador.predecir([pregunta_procesada])[0]
        probabilidad = self.clasificador.predecir_probabilidad([pregunta_procesada])[0]      
        # Si el detector encontró términos dentales (confianza media)
        # Y el modelo ML también dice que es dental → ES DENTAL
        if deteccion['es_dental'] and es_dental_ml == 1:
            es_dental_final = True
            confianza_final = 'Alta'
            metodo = 'Híbrido (Detector + ML coinciden)'
            probabilidad_final = max(0.85, probabilidad[1])
        
        # Si el detector dice NO dental pero ML dice dental con baja probabilidad
        elif not deteccion['es_dental'] and es_dental_ml == 1 and probabilidad[1] < 0.7:
            es_dental_final = False
            confianza_final = 'Alta'
            metodo = 'Detector de Términos (override ML)'
            probabilidad_final = probabilidad[1]
        
        # Si el detector dice dental pero ML dice NO dental
        elif deteccion['es_dental'] and es_dental_ml == 0:
            # Darle más peso al detector si encontró términos claros
            if len(deteccion['terminos_encontrados']) >= 2:
                es_dental_final = True
                confianza_final = 'Media-Alta'
                metodo = 'Detector de Términos (override ML)'
                probabilidad_final = 0.75
            else:
                es_dental_final = False
                confianza_final = 'Media'
                metodo = 'Modelo ML (override Detector)'
                probabilidad_final = probabilidad[1]
        
        # Caso por defecto: confiar en el modelo ML
        else:
            UMBRAL_MINIMO = 0.55
            if probabilidad[1] < UMBRAL_MINIMO and es_dental_ml == 1:
                es_dental_final = False
            else:
                es_dental_final = bool(es_dental_ml)
            
            confianza_final = 'Alta' if probabilidad[1] > 0.8 else 'Media' if probabilidad[1] > 0.6 else 'Baja'
            metodo = 'Modelo ML'
            probabilidad_final = probabilidad[1]
        
        # GENERAR RESPUESTA CON IA
        respuesta = self.generador_respuestas.generar_respuesta(pregunta, es_dental_final)
        
        return {
            'pregunta_original': pregunta,
            'es_anatomia_dental': es_dental_final,
            'probabilidad_dental': float(probabilidad_final),
            'respuesta': respuesta,
            'confianza': confianza_final,
            'metodo': metodo,
            'terminos_encontrados': deteccion['terminos_encontrados'][:5] if deteccion['terminos_encontrados'] else [],
            'deteccion_terminos': deteccion['es_dental'],
            'clasificacion_ml': bool(es_dental_ml),
            'probabilidad_ml': float(probabilidad[1])
        }
    
    def obtener_estadisticas(self):
        """Estadísticas del modelo + detector"""
        if not self.entrenado:
            return {'error': 'Modelo no entrenado'}
        
        caracteristicas_importantes = self.clasificador.obtener_caracteristicas_importantes(10)
        stats_detector = self.detector_terminos.contar_terminos()
        
        return {
            'caracteristicas_importantes': caracteristicas_importantes,
            'configuracion': str(self.configuracion),
            'entrenado': self.entrenado,
            'terminos_dentales': stats_detector['terminos_dentales'],
            'terminos_no_dentales': stats_detector['terminos_no_dentales']
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