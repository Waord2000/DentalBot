import nltk
import os

def configurar_nltk():
    print("CONFIGURANDO RECURSOS NLTK")
    
    # Recursos necesarios para DentalBot
    recursos = [
        'punkt',           # Tokenizer básico
        'stopwords',       # Palabras vacías
        'punkt_tab',       # Tokenizer para español
    ]
    
    for recurso in recursos:
        try:
            print(f"Verificando: {recurso}")
            if recurso == 'punkt_tab':
                nltk.download('punkt_tab', quiet=False)
            else:
                nltk.download(recurso, quiet=False)
            print(f"{recurso} - LISTO")
        except Exception as e:
            print(f"Error con {recurso}: {e}")
    
    print("\n VERIFICACIÓN FINAL:")
    for recurso in recursos:
        try:
            if recurso == 'punkt_tab':
                nltk.data.find(f'tokenizers/{recurso}')
            else:
                nltk.data.find(f'corpora/{recurso}')
            print(f" {recurso} - CONFIRMADO")
        except LookupError:
            print(f" {recurso} - NO ENCONTRADO")
    
    print("\n Configuración de NLTK completada!")

if __name__ == "__main__":
    configurar_nltk()