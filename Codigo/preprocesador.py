import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os

class ProcesadorTexto:
    """Preprocesador de texto personalizado para terminología dental EXPANDIDO"""

    def __init__(self):
        self._descargar_recursos_nltk()
        self.palabras_vacias = self._inicializar_stopwords()
        self.vocabulario_dental = self._inicializar_vocabulario_dental_expandido()  # MEJORADO

    def _descargar_recursos_nltk(self):
        """Descarga todos los recursos necesarios de NLTK una sola vez"""
        recursos_necesarios = ['punkt', 'stopwords', 'punkt_tab']
        
        for recurso in recursos_necesarios:
            try:
                nltk.data.find(f'tokenizers/{recurso}' if recurso.startswith('punkt') else f'corpora/{recurso}')
            except LookupError:
                try:
                    nltk.download(recurso, quiet=True)
                except Exception as e:
                    pass

    def _inicializar_stopwords(self):
        """Inicializa las stopwords en español con manejo de errores"""
        try:
            palabras_vacias = set(stopwords.words('spanish'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            palabras_vacias = set(stopwords.words('spanish'))
        
        # Añadir palabras vacías específicas
        palabras_vacias.update(['qué', 'cómo', 'cuál', 'cuáles', 'dónde', 'cuándo', 'por', 'para'])
        return palabras_vacias

    def _inicializar_vocabulario_dental_expandido(self):
        """
        Vocabulario dental EXPANDIDO - Incluye términos de ortodoncia y más
        Estos términos NUNCA se eliminan como stopwords
        """
        return {
            # ============ ANATOMÍA DENTAL BÁSICA ============
            'incisivo', 'canino', 'premolar', 'molar', 'temporal', 'permanente',
            'superior', 'inferior', 'corona', 'raiz', 'raíz', 'cuspide', 'cúspide',
            'anatomia', 'anatomía', 'morfologia', 'morfología', 'dental', 'diente', 
            'dientes', 'dentición', 'dentadura',
            
            # Plurales y variaciones
            'incisivos', 'caninos', 'premolares', 'molares', 'raices', 'raíces',
            'cuspides', 'cúspides', 'temporales', 'permanentes', 'superiores', 'inferiores',
            
            # ============ ORTODONCIA (NUEVO) ============
            'ortodoncia', 'ortodóncia', 'ortodoncico', 'ortodóncico', 'ortodontico',
            'brackets', 'bracket', 'brakets', 'breket', 'frenillos', 'frenillo',
            'aparato', 'aparatos', 'retenedor', 'retenedores', 'contenedor',
            'alineador', 'alineadores', 'invisalign',
            'arco', 'alambre', 'ligadura', 'ligaduras', 'elástico', 'elásticos',
            'maloclusión', 'maloclusiones', 'maloclusión', 'oclusión', 'oclusal',
            'mordida', 'sobremordida', 'submordida', 'mordida cruzada',
            'apiñamiento', 'diastema', 'espacios',
            'clase', 'skeletal', 'esquelética', 'esquelético',
            
            # ============ NUMERACIÓN Y CLASIFICACIÓN ============
            'fdi', 'oclusal', 'vestibular', 'lingual', 'mesial', 'distal', 
            'cervical', 'incisal', 'nomenclatura', 'numeración', 'clasificación',
            'cuadrante', 'cuadrantes', 'hemiarcada',
            
            # ============ ESTRUCTURAS DENTALES ============
            'esmalte', 'dentina', 'cemento', 'pulpa', 'nervio', 
            'conducto', 'conductos', 'camara', 'cámara', 'pulpar',
            'apex', 'ápice', 'apical', 'foramen',
            
            # ============ TEJIDOS PERIODONTALES ============
            'encía', 'encías', 'gingival', 'periodonto', 'periodontal',
            'ligamento', 'hueso', 'alveolar', 'alvéolo', 'alveolo',
            
            # ============ SUPERFICIES Y CARAS ============
            'palatino', 'palatina', 'proximal', 'interproximal',
            'borde', 'incisal', 'oclusal', 'cara', 'caras',
            'superficie', 'superficies',
            
            # ============ CARACTERÍSTICAS MORFOLÓGICAS ============
            'fosa', 'foseta', 'fosetas', 'surco', 'surcos',
            'cíngulo', 'tubérculo', 'tuberculo', 'cresta', 'crestas',
            'reborde', 'rebordes', 'lóbulo', 'lóbulos', 'lobulo', 'lobulos',
            'carabelli', 'vertiente', 'vertientes', 'cúspide',
            'convexidad', 'concavidad', 'contorno', 'ecuador',
            
            # ============ TIPOS DE DIENTES ============
            'deciduo', 'decidua', 'leche', 'primario', 'primarios',
            'secundario', 'secundarios', 'mixta',
            'muela', 'muelas', 'colmillo', 'colmillos', 'paleta', 'paletas',
            
            # ============ DIMENSIONES Y MEDIDAS ============
            'longitud', 'ancho', 'dimensión', 'dimensiones', 'tamaño',
            'milímetros', 'milimetros', 'mm', 'medida', 'medidas',
            'clínica', 'clinica', 'anatómica', 'anatomica',
            
            # ============ DESARROLLO Y ERUPCIÓN ============
            'erupción', 'erupcionar', 'brote', 'calcificación', 'calcificacion',
            'formación', 'formacion', 'desarrollo', 'crecimiento',
            
            # ============ ANATOMÍA MANDIBULAR/MAXILAR ============
            'maxilar', 'mandíbula', 'mandibula', 'mandibular',
            'anterior', 'posterior', 'arcada', 'arcadas',
            'contacto', 'punto', 'tronera', 'embrasure',
            
            # ============ TÉRMINOS TÉCNICOS ============
            'características', 'característica', 'diferencias', 'diferencia',
            'comparación', 'comparar', 'identificar', 'estructura',
            'forma', 'formas', 'tipo', 'tipos', 'aspecto',
            
            # ============ RADIOLOGÍA DENTAL ============
            'radiografía', 'radiografia', 'radiográfico', 'radiografico',
            'rx', 'periapical', 'panorámica', 'panoramica',
            
            # ============ OTROS TÉRMINOS DENTALES ============
            'tabla', 'oclusal', 'plano', 'línea', 'linea',
            'punto', 'zona', 'área', 'region', 'región',
        }

    def limpiar_texto(self, texto):
        """Limpia el texto manteniendo estructura anatómica"""
        if not isinstance(texto, str) or not texto.strip():
            return ""
        
        # Convertir a minúsculas
        texto = texto.lower()
        
        # Remover puntuación pero conservar significado
        texto = texto.translate(str.maketrans('', '', string.punctuation))
        
        # Remover números
        texto = re.sub(r'\d+', '', texto)
        
        # Remover espacios extra
        texto = ' '.join(texto.split())
        
        return texto

    def tokenizar(self, texto):
        """Tokeniza el texto con manejo de errores robusto"""
        if not texto.strip():
            return []
            
        try:
            return word_tokenize(texto, language='spanish')
        except LookupError:
            self._descargar_recursos_nltk()
            try:
                return word_tokenize(texto, language='spanish')
            except:
                return re.findall(r'\b\w+\b', texto.lower())
        except Exception:
            return re.findall(r'\b\w+\b', texto.lower())

    def remover_palabras_vacias(self, tokens):
        """
        Remueve palabras vacías pero CONSERVA vocabulario dental
        CRÍTICO: Nunca eliminar términos dentales como 'brackets', 'diente', etc.
        """
        if not tokens:
            return []
            
        return [
            token for token in tokens
            if token not in self.palabras_vacias or token in self.vocabulario_dental
        ]

    def preprocesar(self, texto):
        """Pipeline completo de preprocesamiento con manejo robusto de errores"""
        try:
            if not texto or not isinstance(texto, str):
                return ""
                
            texto_limpio = self.limpiar_texto(texto)
            if not texto_limpio:
                return ""
                
            tokens = self.tokenizar(texto_limpio)
            tokens_filtrados = self.remover_palabras_vacias(tokens)
            resultado = ' '.join(tokens_filtrados)
            
            return resultado if resultado else texto_limpio
            
        except Exception as e:
            print(f"Error crítico en preprocesamiento: {e}")
            return self.limpiar_texto(texto) if texto else ""

    def preprocesar_lote(self, textos):
        """Procesa un lote de textos eficientemente"""
        print("Preprocesando lote de textos")
        resultados = []
        for texto in textos:
            resultado = self.preprocesar(texto)
            resultados.append(resultado)
        return resultados