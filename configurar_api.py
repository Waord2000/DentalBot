import os

def configurar_api_key():
    """Configura la API key de Anthropic"""
    print(" CONFIGURACIÓN DE API KEY DE CLAUDE")
    print("\nPara obtener tu API key:")
    print("1. Ve a: https://console.anthropic.com/")
    print("2. Crea una cuenta o inicia sesión")
    print("3. Ve a 'API Keys' y crea una nueva key")
    print("4. Copia la key (empieza con 'sk-ant-')")
    print("\nIMPORTANTE: La key es secreta, no la compartas")
    print("="*60)
    
    api_key = input("\nPega tu API key aquí: ").strip()
    
    if not api_key:
        print("No se ingresó ninguna key")
        return False
    
    if not api_key.startswith('sk-ant-'):
        print("⚠ Advertencia: La key no tiene el formato esperado")
        continuar = input("¿Continuar de todos modos? (s/n): ").lower()
        if continuar != 's':
            return False
    
    # Guardar en archivo .env
    with open('.env', 'w') as f:
        f.write(f"ANTHROPIC_API_KEY={api_key}\n")
    
    # Establecer variable de entorno
    os.environ['ANTHROPIC_API_KEY'] = api_key
    
    print("\n✓ API key configurada correctamente")
    print("✓ Guardada en archivo .env")
    print("\n⚠ Recuerda: .env está en .gitignore (no se subirá a GitHub)")
    
    return True

if __name__ == "__main__":
    configurar_api_key()