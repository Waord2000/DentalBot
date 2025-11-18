import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os

class ProcesadorTexto:
    """Preprocesador de texto personalizado para terminología dental"""

    def __init__(self):
        self._descargar_recursos_nltk()
        self.palabras_vacias = self._inicializar_stopwords()
        self.vocabulario_dental = self._inicializar_vocabulario_dental()

    def _descargar_recursos_nltk(self):
        """Descarga todos los recursos necesarios de NLTK una sola vez"""
        recursos_necesarios = ['punkt', 'stopwords', 'punkt_tab']
        
        for recurso in recursos_necesarios:
            try:
                nltk.data.find(f'tokenizers/{recurso}' if recurso.startswith('punkt') else f'corpora/{recurso}')
                print(f"Recurso NLTK '{recurso}' ya está disponible")
            except LookupError:
                print(f"Descargando recurso NLTK: {recurso}")
                try:
                    nltk.download(recurso, quiet=True)
                    print(f"Recurso '{recurso}' descargado exitosamente")
                except Exception as e:
                    print(f"Error descargando {recurso}: {e}")

    def _inicializar_stopwords(self):
        """Inicializa las stopwords en español con manejo de errores"""
        try:
            palabras_vacias = set(stopwords.words('spanish'))
        except LookupError:
            print("Descargando stopwords...")
            nltk.download('stopwords', quiet=True)
            palabras_vacias = set(stopwords.words('spanish'))
        
        # Añadir palabras vacías específicas
        palabras_vacias.update(['qué', 'cómo', 'cuál', 'cuáles', 'dónde', 'cuándo', 'por', 'para'])
        return palabras_vacias

    def _inicializar_vocabulario_dental(self):
        """Inicializa el vocabulario dental protegido"""
        return {
            'incisivo', 'canino', 'premolar', 'molar', 'temporal', 'permanente',
            'superior', 'inferior', 'corona', 'raiz', 'cuspide', 'anatomia',
            'morfologia', 'dental', 'diente', 'fdi', 'oclusal', 'vestibular',
            'lingual', 'mesial', 'distal', 'cervical', 'incisal', 'dentición',
            'premolares', 'molares', 'caninos', 'incisivos', 'raíz', 'cúspide',
            'anatómica', 'morfológica', 'oclusal', 'vestibular', 'lingual',
            'mesial', 'distal', 'cervical', 'incisal', 'dentina', 'esmalte',
            'pulpa', 'periodonto', 'encía', 'cemento', 'hueso', 'alveolar'
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
            # Intentar con tokenizador en español
            return word_tokenize(texto, language='spanish')
        except LookupError:
            # Si falla, descargar recursos y reintentar
            self._descargar_recursos_nltk()
            try:
                return word_tokenize(texto, language='spanish')
            except:
                # Fallback: tokenización simple por espacios
                return re.findall(r'\b\w+\b', texto.lower())
        except Exception:
            # Fallback para cualquier otro error
            return re.findall(r'\b\w+\b', texto.lower())

    def remover_palabras_vacias(self, tokens):
        """Remueve palabras vacías pero CONSERVA vocabulario dental"""
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
            # En caso de error crítico, devolver texto limpio básico
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