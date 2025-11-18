from groq import Groq

class GeneradorRespuestasIA:
    """Generador de respuestas usando IA (Groq API)"""
    
    def __init__(self):
        # API Key de Groq configurada
        self.api_key = "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"
        self.client = Groq(api_key=self.api_key)
        self.modelo = "llama-3.3-70b-versatile"
    
    def generar_respuesta(self, pregunta, es_dental=True):
        """Genera respuesta usando IA"""
        
        if es_dental:
            prompt = f"""Eres un experto en anatomía dental. Responde de forma clara y concisa la siguiente pregunta sobre anatomía dental.
Si la pregunta es sobre procedimientos clínicos o tratamientos, menciona que solo respondes sobre anatomía.

Pregunta: {pregunta}

Responde en máximo 150 palabras, enfocándote solo en aspectos anatómicos."""
        else:
            prompt = f"""La siguiente pregunta NO es sobre anatomía dental: "{pregunta}"

Responde amablemente indicando que eres un asistente especializado en anatomía dental y no puedes responder preguntas sobre otros temas.
Sugiere reformular la pregunta si tiene relación con anatomía dental.

Responde en máximo 50 palabras."""
        
        try:
            respuesta = self.client.chat.completions.create(
                model=self.modelo,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300
            )
            return respuesta.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generando respuesta con IA: {str(e)}"