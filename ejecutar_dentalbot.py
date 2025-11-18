import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'codigo'))

from dentalbot import DentalBot

def main():
    print("="*70)
    print(" "*15 + "DENTALBOT - SISTEMA CON IA")
    print("="*70)
    
    # Inicializar DentalBot (API key ya configurada internamente)
    print("\nInicializando sistema...")
    bot = DentalBot()
    print("API key configurada")
    
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
    print("  - Escribe tu pregunta sobre anatomía dental")
    print("  - 'estadisticas' para ver info del modelo")
    print("  - 'salir' para terminar")
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
                print("ESTADÍSTICAS DEL MODELO")
                print("="*70)
                print(f"Configuración: {stats['configuracion']}")
                print(f"Entrenado: {stats['entrenado']}")
                if stats['caracteristicas_importantes']:
                    print("\nTop 5 características importantes:")
                    for i, (caract, imp) in enumerate(stats['caracteristicas_importantes'][:5], 1):
                        print(f"  {i}. {caract}: {imp:.4f}")
                print("="*70)
                continue
                
            elif not pregunta:
                continue
            
            print("\nProcesando con IA...")
            resultado = bot.clasificar_pregunta(pregunta)
            
            if 'error' in resultado:
                print(f"\nError: {resultado['error']}")
                continue
            
            # Mostrar resultados
            print("\n" + "="*70)
            print("RESULTADO DE CLASIFICACIÓN")
            print("="*70)
            print(f"Es anatomía dental: {'✓ SÍ' if resultado['es_anatomia_dental'] else '✗ NO'}")
            print(f"Probabilidad: {resultado['probabilidad_dental']:.1%}")
            print(f"Confianza: {resultado['confianza']}")
            print("-"*70)
            print("RESPUESTA GENERADA POR IA:")
            print("-"*70)
            print(f"{resultado['respuesta']}")
            print("="*70)
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Intenta de nuevo o escribe 'salir' para terminar")

if __name__ == "__main__":
    main()