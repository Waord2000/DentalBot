import os
import sys

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_codigo = os.path.join(ruta_actual, 'codigo')
sys.path.insert(0, ruta_codigo)

try:
    from dentalbot import DentalBot
except ImportError as e:
    print(f"Error de importación: {e}")
    print("Verifica que todos los archivos estén en la carpeta 'codigo/'")
    sys.exit(1)

def main():
    print("="*70)
    print(" "*15 + "DENTALBOT - ENTRENAMIENTO DEL MODELO")
    print("="*70)

    # Crear directorios si no existen
    os.makedirs('modelos', exist_ok=True)
    os.makedirs('datos', exist_ok=True)
    os.makedirs('resultados', exist_ok=True)  # ASEGURAR QUE EXISTE
    
    # Inicializar DentalBot
    print("\nInicializando DentalBot...")
    bot = DentalBot()
    print("Sistema inicializado")
    
    # Ruta del CSV
    ruta_csv = os.path.join('datos', 'preguntas_dentales.csv')
    
    print(f"\nBuscando datos en: {ruta_csv}")
    
    # Verificar si el CSV existe
    if os.path.exists(ruta_csv):
        print("Archivo CSV encontrado")
        usar_csv = True
    else:
        print("Archivo CSV no encontrado")
        print("Se usaran datos de ejemplo integrados")
        usar_csv = False
    
    # Entrenar modelo
    print("\n" + "="*70)
    print("INICIANDO ENTRENAMIENTO")
    print("="*70)
    
    try:
        # AQUÍ SE MOSTRARAN TODAS LAS METRICAS AUTOMÁTICAMENTE
        precision, reporte = bot.entrenar_modelo(usar_csv=usar_csv, ruta_csv=ruta_csv)
    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")
        print("\nReintentando con datos de ejemplo...")
        try:
            precision, reporte = bot.entrenar_modelo(usar_csv=False)
        except Exception as e2:
            print(f"\nError crítico: {e2}")
            print("\nPosibles soluciones:")
            print("   - Verifica que todos los archivos .py estén en codigo/")
            print("   - Verifica las dependencias: pip install -r Requerimientos.txt")
            return
    
    # Mostrar resumen final
    print("\n" + "="*70)
    print("RESUMEN FINAL DEL ENTRENAMIENTO")
    print("="*70)
    print(f"Precision global: {precision:.4f} ({precision*100:.2f}%)")
    
    if reporte and 'Anatomia Dental' in reporte:
        print(f"\nClase 'Anatomia Dental':")
        print(f"   - Precision: {reporte['Anatomia Dental']['precision']:.4f}")
        print(f"   - Recall:    {reporte['Anatomia Dental']['recall']:.4f}")
        print(f"   - F1-Score:  {reporte['Anatomia Dental']['f1-score']:.4f}")
    
    if reporte and 'No Dental' in reporte:
        print(f"\nClase 'No Dental':")
        print(f"   - Precision: {reporte['No Dental']['precision']:.4f}")
        print(f"   - Recall:    {reporte['No Dental']['recall']:.4f}")
        print(f"   - F1-Score:  {reporte['No Dental']['f1-score']:.4f}")
    
    # Guardar modelo
    modelo_ruta = 'modelos/dentalbot_rf.pkl'
    print(f"\nGuardando modelo en: {modelo_ruta}")
    try:
        bot.guardar_modelo(modelo_ruta)
        print("Modelo guardado exitosamente")
    except Exception as e:
        print(f"Error guardando modelo: {e}")
    
    # Mostrar características importantes
    print("\n" + "="*70)
    print("TOP 5 CARACTERISTICAS MAS IMPORTANTES")
    print("="*70)
    try:
        stats = bot.obtener_estadisticas()
        if 'error' not in stats:
            for i, (caract, importancia) in enumerate(stats['caracteristicas_importantes'][:5], 1):
                print(f"   {i}. {caract:<20} -> {importancia:.4f}")
        print("="*70)
    except Exception as e:
        print(f"No se pudieron obtener características: {e}")
    
    # Verificar que el archivo de métricas se generó
    ruta_metricas = 'resultados/metricas_evaluacion.txt'
    if os.path.exists(ruta_metricas):
        print(f"\nMETRICAS GUARDADAS EN: {ruta_metricas}")
    else:
        print(f"\nADVERTENCIA: No se generó el archivo de métricas en {ruta_metricas}")
    
    # Probar con una pregunta de ejemplo
    print("\n" + "="*70)
    print("PROBANDO EL MODELO CON IA")
    print("="*70)
    pregunta_ejemplo = "¿Cuáles son las características del incisivo central superior?"
    print(f"\nPregunta de prueba: '{pregunta_ejemplo}'")
    print("\nGenerando respuesta con IA...")
    
    try:
        resultado = bot.clasificar_pregunta(pregunta_ejemplo)
        
        if 'error' not in resultado:
            print("\n" + "-"*70)
            estado = "SI" if resultado['es_anatomia_dental'] else "NO"
            print(f"Es anatomia dental: {estado}")
            print(f"Probabilidad: {resultado['probabilidad_dental']:.1%}")
            print(f"Confianza: {resultado['confianza']}")
            print("-"*70)
            print("Respuesta generada:")
            print(f"{resultado['respuesta']}")
            print("-"*70)
        else:
            print(f"\nError en prueba: {resultado['error']}")
    except Exception as e:
        print(f"\nError en prueba: {e}")
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print("\nAhora puedes ejecutar: python ejecutar_dentalbot.py")
    print(f"Revisa las métricas en: {ruta_metricas}")

if __name__ == "__main__":
    main()