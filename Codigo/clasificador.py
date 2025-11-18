import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

class ClasificadorDentalRandomForest:
    """Clasificador Random Forest especializado en anatomía dental"""

    def __init__(self, configuracion):
        self.configuracion = configuracion
        self.entrenado = False
        self.importancia_caracteristicas = None
        
        # Vectorizador TF-IDF
        self.vectorizador = TfidfVectorizer(
            max_features=configuracion.max_caracteristicas,
            min_df=configuracion.min_frecuencia_documento,
            max_df=configuracion.max_frecuencia_documento,
            ngram_range=configuracion.rango_ngramas,
            stop_words=None
        )

        # Random Forest Classifier
        self.clasificador = RandomForestClassifier(
            n_estimators=configuracion.numero_arboles,
            max_depth=configuracion.profundidad_maxima,
            min_samples_split=configuracion.muestras_minimas_division,
            min_samples_leaf=configuracion.muestras_minimas_hoja,
            random_state=configuracion.estado_aleatorio,
            n_jobs=-1
        )

        # Pipeline completo
        self.pipeline = Pipeline([
            ('vectorizador', self.vectorizador),
            ('clasificador', self.clasificador)
        ])

    def preparar_datos(self, preguntas, etiquetas):
        """Prepara y divide los datos para entrenamiento"""
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
            preguntas,
            etiquetas,
            test_size=self.configuracion.tamaño_prueba,
            random_state=self.configuracion.estado_aleatorio,
            stratify=etiquetas
        )
        
        print(f"División de datos:")
        print(f"   - Entrenamiento: {len(X_entrenamiento)} ejemplos")
        print(f"   - Prueba: {len(X_prueba)} ejemplos")
        
        return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

    def entrenar(self, X_entrenamiento, y_entrenamiento):
        """Entrena el modelo Random Forest"""
        print("Entrenando modelo Random Forest")
        
        with tqdm(total=100, desc="Progreso") as pbar:
            # Entrenar pipeline
            self.pipeline.fit(X_entrenamiento, y_entrenamiento)
            pbar.update(50)
            
            # Obtener importancia de características
            nombres_caracteristicas = self.vectorizador.get_feature_names_out()
            importancias = self.clasificador.feature_importances_
            self.importancia_caracteristicas = dict(zip(nombres_caracteristicas, importancias))
            pbar.update(50)
        
        self.entrenado = True
        print("Entrenamiento completado")
        return self

    def predecir(self, X_prueba):
        """Realiza predicciones"""
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado primero")
        return self.pipeline.predict(X_prueba)

    def predecir_probabilidad(self, X_prueba):
        """Predice probabilidades"""
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado primero")
        return self.pipeline.predict_proba(X_prueba)

    def evaluar(self, X_prueba, y_prueba, mostrar_metricas=True):
        """Evalúa el modelo con métricas completas en formato texto"""
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado primero")

        predicciones = self.predecir(X_prueba)
        precision = accuracy_score(y_prueba, predicciones)
        reporte = classification_report(
            y_prueba, predicciones,
            target_names=['No Dental', 'Anatomía Dental'],
            output_dict=True
        )

        # NUEVA FUNCIONALIDAD: Mostrar métricas detalladas en texto
        if mostrar_metricas:
            self._mostrar_metricas_detalladas(y_prueba, predicciones, precision, reporte)

        return precision, reporte, predicciones

    def _mostrar_metricas_detalladas(self, y_real, y_pred, precision, reporte):
        """Muestra métricas detalladas en formato texto"""
        
        print("\n" + "="*70)
        print(" "*20 + "REPORTE DE EVALUACIÓN DEL MODELO")
        print("="*70)
        
        # 1. ACCURACY GENERAL
        print(f"\n{'PRECISIÓN GENERAL (ACCURACY)':^70}")
        print("-"*70)
        print(f"   Accuracy: {precision:.4f} ({precision*100:.2f}%)")
        
        # 2. MATRIZ DE CONFUSIÓN
        print(f"\n{'MATRIZ DE CONFUSIÓN':^70}")
        print("-"*70)
        cm = confusion_matrix(y_real, y_pred)
        
        print(f"\n                    Predicho")
        print(f"                 No Dental  |  Anatomía Dental")
        print(f"              " + "-"*35)
        print(f"   Real       |")
        print(f"   No Dental      {cm[0][0]:^6}    |    {cm[0][1]:^6}")
        print(f"   Anatomía Dental {cm[1][0]:^6}    |    {cm[1][1]:^6}")
        
        # Interpretación de la matriz
        print(f"\n   Interpretación:")
        print(f"   - Verdaderos Negativos (VN): {cm[0][0]} - Correctamente clasificados como NO dental")
        print(f"   - Falsos Positivos (FP):     {cm[0][1]} - Incorrectamente clasificados como dental")
        print(f"   - Falsos Negativos (FN):     {cm[1][0]} - Incorrectamente clasificados como NO dental")
        print(f"   - Verdaderos Positivos (VP): {cm[1][1]} - Correctamente clasificados como dental")
        
        # 3. MÉTRICAS POR CLASE
        print(f"\n{'MÉTRICAS DETALLADAS POR CLASE':^70}")
        print("-"*70)
        
        print(f"\n   CLASE: No Dental")
        print(f"   {'Precision:':<20} {reporte['No Dental']['precision']:.4f} ({reporte['No Dental']['precision']*100:.2f}%)")
        print(f"   {'Recall:':<20} {reporte['No Dental']['recall']:.4f} ({reporte['No Dental']['recall']*100:.2f}%)")
        print(f"   {'F1-Score:':<20} {reporte['No Dental']['f1-score']:.4f} ({reporte['No Dental']['f1-score']*100:.2f}%)")
        print(f"   {'Soporte:':<20} {int(reporte['No Dental']['support'])} muestras")
        
        print(f"\n   CLASE: Anatomía Dental")
        print(f"   {'Precision:':<20} {reporte['Anatomía Dental']['precision']:.4f} ({reporte['Anatomía Dental']['precision']*100:.2f}%)")
        print(f"   {'Recall:':<20} {reporte['Anatomía Dental']['recall']:.4f} ({reporte['Anatomía Dental']['recall']*100:.2f}%)")
        print(f"   {'F1-Score:':<20} {reporte['Anatomía Dental']['f1-score']:.4f} ({reporte['Anatomía Dental']['f1-score']*100:.2f}%)")
        print(f"   {'Soporte:':<20} {int(reporte['Anatomía Dental']['support'])} muestras")
        
        # 4. PROMEDIOS
        print(f"\n{'PROMEDIOS':^70}")
        print("-"*70)
        print(f"   {'Macro avg Precision:':<25} {reporte['macro avg']['precision']:.4f}")
        print(f"   {'Macro avg Recall:':<25} {reporte['macro avg']['recall']:.4f}")
        print(f"   {'Macro avg F1-Score:':<25} {reporte['macro avg']['f1-score']:.4f}")
        print(f"\n   {'Weighted avg Precision:':<25} {reporte['weighted avg']['precision']:.4f}")
        print(f"   {'Weighted avg Recall:':<25} {reporte['weighted avg']['recall']:.4f}")
        print(f"   {'Weighted avg F1-Score:':<25} {reporte['weighted avg']['f1-score']:.4f}")
        
        # 5. TOP 10 CARACTERÍSTICAS IMPORTANTES
        if self.importancia_caracteristicas:
            print(f"\n{'TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES':^70}")
            print("-"*70)
            top_features = sorted(self.importancia_caracteristicas.items(), 
                                key=lambda x: x[1], reverse=True)[:10]
            
            for i, (feature, importance) in enumerate(top_features, 1):
                barra = "█" * int(importance * 100)
                print(f"   {i:2}. {feature:<20} {importance:.6f}  {barra}")
        
        # 6. ESTADÍSTICAS ADICIONALES
        print(f"\n{'ESTADÍSTICAS ADICIONALES':^70}")
        print("-"*70)
        
        # Calcular tasa de error
        tasa_error = 1 - precision
        print(f"   {'Tasa de Error:':<30} {tasa_error:.4f} ({tasa_error*100:.2f}%)")
        
        # Distribución de predicciones
        unique, counts = np.unique(y_pred, return_counts=True)
        total = len(y_pred)
        print(f"\n   Distribución de Predicciones:")
        print(f"   - No Dental:        {counts[0]:3} ({counts[0]/total*100:.1f}%)")
        print(f"   - Anatomía Dental:  {counts[1]:3} ({counts[1]/total*100:.1f}%)")
        
        # Distribución real
        unique_real, counts_real = np.unique(y_real, return_counts=True)
        print(f"\n   Distribución Real:")
        print(f"   - No Dental:        {counts_real[0]:3} ({counts_real[0]/total*100:.1f}%)")
        print(f"   - Anatomía Dental:  {counts_real[1]:3} ({counts_real[1]/total*100:.1f}%)")
        
        print("\n" + "="*70)
        
        # GUARDAR MÉTRICAS EN ARCHIVO DE TEXTO
        self._guardar_metricas_txt(precision, reporte, cm, y_real, y_pred)

    def _guardar_metricas_txt(self, precision, reporte, cm, y_real, y_pred):
        """Guarda métricas en archivo de texto"""
        
        # Crear directorio si no existe
        os.makedirs('resultados', exist_ok=True)
        
        ruta_txt = 'resultados/metricas_evaluacion.txt'
        
        with open(ruta_txt, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(" "*20 + "REPORTE DE EVALUACIÓN DEL MODELO\n")
            f.write("="*70 + "\n")
            
            f.write(f"\nPRECISIÓN GENERAL (ACCURACY)\n")
            f.write("-"*70 + "\n")
            f.write(f"   Accuracy: {precision:.4f} ({precision*100:.2f}%)\n")
            
            f.write(f"\nMATRIZ DE CONFUSIÓN\n")
            f.write("-"*70 + "\n")
            f.write(f"\n                    Predicho\n")
            f.write(f"                 No Dental  |  Anatomía Dental\n")
            f.write(f"              " + "-"*35 + "\n")
            f.write(f"   Real       |\n")
            f.write(f"   No Dental      {cm[0][0]:^6}    |    {cm[0][1]:^6}\n")
            f.write(f"   Anatomía Dental {cm[1][0]:^6}    |    {cm[1][1]:^6}\n")
            
            f.write(f"\n   Interpretación:\n")
            f.write(f"   - Verdaderos Negativos (VN): {cm[0][0]}\n")
            f.write(f"   - Falsos Positivos (FP):     {cm[0][1]}\n")
            f.write(f"   - Falsos Negativos (FN):     {cm[1][0]}\n")
            f.write(f"   - Verdaderos Positivos (VP): {cm[1][1]}\n")
            
            f.write(f"\nMÉTRICAS DETALLADAS POR CLASE\n")
            f.write("-"*70 + "\n")
            
            for clase in ['No Dental', 'Anatomía Dental']:
                f.write(f"\n   CLASE: {clase}\n")
                f.write(f"   Precision: {reporte[clase]['precision']:.4f} ({reporte[clase]['precision']*100:.2f}%)\n")
                f.write(f"   Recall:    {reporte[clase]['recall']:.4f} ({reporte[clase]['recall']*100:.2f}%)\n")
                f.write(f"   F1-Score:  {reporte[clase]['f1-score']:.4f} ({reporte[clase]['f1-score']*100:.2f}%)\n")
                f.write(f"   Soporte:   {int(reporte[clase]['support'])} muestras\n")
            
            f.write(f"\nPROMEDIOS\n")
            f.write("-"*70 + "\n")
            f.write(f"   Macro avg Precision:    {reporte['macro avg']['precision']:.4f}\n")
            f.write(f"   Macro avg Recall:       {reporte['macro avg']['recall']:.4f}\n")
            f.write(f"   Macro avg F1-Score:     {reporte['macro avg']['f1-score']:.4f}\n")
            f.write(f"   Weighted avg Precision: {reporte['weighted avg']['precision']:.4f}\n")
            f.write(f"   Weighted avg Recall:    {reporte['weighted avg']['recall']:.4f}\n")
            f.write(f"   Weighted avg F1-Score:  {reporte['weighted avg']['f1-score']:.4f}\n")
            
            # Top características
            if self.importancia_caracteristicas:
                f.write(f"\nTOP 10 CARACTERÍSTICAS MÁS IMPORTANTES\n")
                f.write("-"*70 + "\n")
                top_features = sorted(self.importancia_caracteristicas.items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
                
                for i, (feature, importance) in enumerate(top_features, 1):
                    f.write(f"   {i:2}. {feature:<20} {importance:.6f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"\n   Métricas guardadas en: {ruta_txt}")

    def obtener_caracteristicas_importantes(self, top_n=20):
        """Obtiene las características más importantes"""
        if not self.entrenado or self.importancia_caracteristicas is None:
            return []

        caracteristicas_ordenadas = sorted(
            self.importancia_caracteristicas.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return caracteristicas_ordenadas[:top_n]

    def guardar_modelo(self, ruta):
        """Guarda el modelo entrenado"""
        with open(ruta, 'wb') as f:
            pickle.dump(self, f)
        print(f"Modelo guardado en: {ruta}")

    @classmethod
    def cargar_modelo(cls, ruta):
        """Carga un modelo entrenado"""
        with open(ruta, 'rb') as f:
            modelo = pickle.load(f)
        print(f"Modelo cargado desde: {ruta}")
        return modelo