import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'codigo'))

from dentalbot import DentalBot

def main():
    print("="*70)
    print(" "*10 + "DENTALBOT - SISTEMA CON IA")
    print("="*70)
    
    # Inicializar DentalBot
    print("\nInicializando sistema...")
    bot = DentalBot()
    print("API key configurada")
    print("Detector de términos dentales cargado")
    
    # Mostrar estadísticas del detector
    stats_detector = bot.detector_terminos.contar_terminos()
    print(f"{stats_detector['terminos_dentales']} términos dentales en diccionario")
    print(f"{stats_detector['terminos_no_dentales']} términos clínicos detectados")
    
    # Cargar o entrenar modelo
    modelo_path = 'modelos/dentalbot_rf.pkl'
    if os.path.exists(modelo_path):
        print("\nCargando modelo existente...")
        try:
            bot.cargar_modelo(modelo_path)
            print("Modelo cargado desde archivo")
        except:
            print("Error cargando modelo")
            print("\nEntrenando nuevo modelo...")
            bot.entrenar_modelo()
            bot.guardar_modelo(modelo_path)
    else:
        print("\nEntrenando modelo por primera vez...")
        bot.entrenar_modelo()
        bot.guardar_modelo(modelo_path)
    
    print("\n" + "="*70)
    print(" "*20 + "SISTEMA LISTO")
    print("="*70)
    print("\nComandos disponibles:")
    print("   - Escribe tu pregunta sobre anatomía dental")
    print("   - 'estadisticas' para ver info del modelo")
    print("   - 'pruebas' para ejecutar casos de prueba")
    print("   - 'salir' para terminar")
    print("="*70)
    
    # Bucle interactivo
    while True:
        try:
            pregunta = input("\n➤ Tu pregunta: ").strip()
            
            if pregunta.lower() == 'salir':
                print("\n¡Hasta luego!")
                break
                
            elif pregunta.lower() == 'estadisticas':
                stats = bot.obtener_estadisticas()
                print("\n" + "="*70)
                print("ESTADÍSTICAS DEL SISTEMA")
                print("="*70)
                print(f"Configuración: {stats['configuracion']}")
                print(f"Modelo entrenado: {stats['entrenado']}")
                print(f"Términos dentales: {stats['terminos_dentales']}")
                print(f"Términos no dentales (clínicos): {stats['terminos_no_dentales']}")
                if stats['caracteristicas_importantes']:
                    print("\nTop 5 características importantes (ML):")
                    for i, (caract, imp) in enumerate(stats['caracteristicas_importantes'][:5], 1):
                        print(f"  {i}. {caract}: {imp:.4f}")
                print("="*70)
                continue
                
            elif pregunta.lower() == 'pruebas':
                print("\n" + "="*70)
                print("EJECUTANDO CASOS DE PRUEBA")
                print("="*70)
                
                casos_prueba = [
                    "¿Qué es un diente?",
                    "¿Características de los brackets?",
                    "¿Cómo es la ortodoncia?",
                    "¿Morfología del incisivo central?",
                    "¿Cómo hacer una obturación?",
                    "¿Capital de Francia?",
                ]
                
                for i, caso in enumerate(casos_prueba, 1):
                    print(f"\n[{i}/{len(casos_prueba)}] Probando: '{caso}'")
                    resultado = bot.clasificar_pregunta(caso)
                    
                    if 'error' not in resultado:
                        print(f"   ➜ Resultado: {'✓ DENTAL' if resultado['es_anatomia_dental'] else '✗ NO DENTAL'}")
                        print(f"   ➜ Método: {resultado.get('metodo', 'N/A')}")
                        print(f"   ➜ Probabilidad: {resultado['probabilidad_dental']:.1%}")
                        if resultado.get('terminos_encontrados'):
                            print(f"   ➜ Términos: {', '.join(resultado['terminos_encontrados'][:3])}")
                
                print("\n" + "="*70)
                continue
                
            elif not pregunta:
                continue
            
            print("\nAnalizando por favor esperar...")
            resultado = bot.clasificar_pregunta(pregunta)
            
            if 'error' in resultado:
                print(f"\nError: {resultado['error']}")
                continue
            
            # Mostrar resultados DETALLADOS
            print("\n" + "="*70)
            print("RESULTADO DE CLASIFICACIÓN")
            print("="*70)
            print(f"Es anatomía dental: {'✓ SÍ' if resultado['es_anatomia_dental'] else '✗ NO'}")
            print(f"Probabilidad: {resultado['probabilidad_dental']:.1%}")
            print(f"Confianza: {resultado['confianza']}")
            print(f"Método usado: {resultado.get('metodo', 'N/A')}")
            
            # Detalles del análisis híbrido
            if 'deteccion_terminos' in resultado:
                print("\nANÁLISIS DETALLADO:")
                print(f"   Detector de Términos: {'✓ DENTAL' if resultado['deteccion_terminos'] else '✗ NO DENTAL'}")
                print(f"   Modelo ML:            {'✓ DENTAL' if resultado['clasificacion_ml'] else '✗ NO DENTAL'}")
                print(f"   Probabilidad ML:      {resultado['probabilidad_ml']:.1%}")
                
            if resultado.get('terminos_encontrados'):
                print(f"\nTérminos detectados:")
                for termino in resultado['terminos_encontrados']:
                    print(f"   • {termino}")
            
            if resultado.get('razon'):
                print(f"\nRazón: {resultado['razon']}")
            
            print("-"*70)
            print("RESPUESTA GENERADA POR IA:")
            print("-"*70)
            print(f"{resultado['respuesta']}")
            print("="*70)
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")
            print("Intenta de nuevo o escribe 'salir' para terminar")

if __name__ == "__main__":
    main()