class DetectorTerminosDentales:
    """Detector basado en diccionarios de términos dentales"""
    
    def __init__(self):
        self.terminos_dentales = self._cargar_terminos_dentales()
        self.terminos_no_dentales = self._cargar_terminos_no_dentales()
    
    def _cargar_terminos_dentales(self):
        """Diccionario completo de términos relacionados con odontología"""
        return {
            # ANATOMÍA DENTAL BÁSICA
            'diente', 'dientes', 'dental', 'dentales', 'dentición', 'dentadura',
            'incisivo', 'incisivos', 'canino', 'caninos', 'premolar', 'premolares',
            'molar', 'molares', 'corona', 'raiz', 'raíz', 'raices', 'raíces',
            
            # TIPOS DE DIENTES
            'temporal', 'temporales', 'permanente', 'permanentes', 'deciduo', 'decidua',
            'leche', 'primario', 'primarios', 'secundario', 'secundarios',
            
            # ESTRUCTURAS DENTALES
            'esmalte', 'dentina', 'cemento', 'pulpa', 'nervio', 'cámara pulpar',
            'conducto', 'conductos', 'apex', 'ápice', 'foramen', 'apical',
            
            # SUPERFICIES Y CARAS
            'oclusal', 'incisal', 'vestibular', 'lingual', 'palatino', 'palatina',
            'mesial', 'distal', 'proximal', 'interproximal', 'cervical',
            
            # ESTRUCTURAS ANATÓMICAS
            'cúspide', 'cúspides', 'fosa', 'foseta', 'surco', 'surcos',
            'cíngulo', 'tubérculo', 'cresta', 'reborde', 'lóbulo',
            'carabelli', 'vertiente', 'vertientes',
            
            # TEJIDOS PERIODONTALES
            'encía', 'encías', 'gingival', 'periodonto', 'periodontal',
            'ligamento', 'hueso alveolar', 'alveolar', 'alvéolo', 'alveolo',
            
            # ORTODONCIA
            'ortodoncia', 'ortodóncia', 'ortodoncico', 'ortodóncico',
            'brackets', 'bracket', 'brakets', 'breket', 'frenillos',
            'aparato', 'aparatos', 'retenedor', 'retenedores',
            'alineador', 'alineadores', 'invisalign',
            'arco', 'alambre', 'ligadura', 'elástico', 'elásticos',
            'maloclusión', 'maloclusiones', 'oclusión', 'mordida',
            'apiñamiento', 'diastema', 'sobremordida', 'submordida',
            'clase skeletal', 'clase molar', 'clase canina',
            
            # CLASIFICACIONES
            'fdi', 'nomenclatura', 'numeración', 'sistema universal',
            'cuadrante', 'cuadrantes', 'hemiarcada',
            
            # MORFOLOGÍA
            'morfología', 'morfológico', 'morfológica', 'anatomía', 'anatómico',
            'estructura', 'característica', 'características', 'forma',
            
            # DIMENSIONES
            'corona clínica', 'corona anatómica', 'longitud', 'ancho',
            'dimensión', 'dimensiones', 'tamaño', 'milímetros',
            
            # DESARROLLO DENTAL
            'erupción', 'erupcionar', 'brote', 'calcificación',
            'formación', 'desarrollo', 'crecimiento',
            
            # RADIOLOGÍA DENTAL
            'radiografía', 'radiográfico', 'rx', 'periapical',
            'panorámica', 'bite-wing', 'oclusal',
            
            # OTROS TÉRMINOS DENTALES
            'maxilar', 'mandíbula', 'mandibular', 'superior', 'inferior',
            'anterior', 'posterior', 'arcada', 'arcadas',
            'contacto', 'punto de contacto', 'tronera', 'embrasure',
            
            # VARIANTES Y SINÓNIMOS
            'muela', 'muelas', 'colmillo', 'colmillos', 'paleta', 'paletas',
        }
    
    def _cargar_terminos_no_dentales(self):
        """Términos que NO son de anatomía dental (procedimientos, enfermedades)"""
        return {
            # PROCEDIMIENTOS CLÍNICOS
            'obturación', 'obturacion', 'amalgama', 'resina', 'composite',
            'extracción', 'extraccion', 'exodoncia', 'sacar',
            'limpieza', 'profilaxis', 'tartrectomía', 'tartrectomia',
            'blanqueamiento', 'blanquear', 'aclarar',
            'implante', 'implantes', 'implantar', 'colocar',
            'endodoncia', 'tratamiento de conducto', 'matar nervio',
            'pulpotomía', 'pulpotomia', 'pulpectomía', 'pulpectomia',
            'corona protésica', 'puente', 'prótesis', 'protesis',
            'cirugía', 'cirugia', 'quirúrgico', 'quirurgico',
            
            # ENFERMEDADES Y PATOLOGÍAS
            'caries', 'cavidad', 'lesión', 'lesion',
            'gingivitis', 'periodontitis', 'piorrea',
            'absceso', 'infección', 'infeccion', 'inflamación', 'inflamacion',
            'pulpitis', 'necrosis', 'granuloma', 'quiste',
            'bruxismo', 'rechinar', 'apretar',
            'halitosis', 'mal aliento',
            'sensibilidad', 'hipersensibilidad',
            
            # MEDICAMENTOS
            'analgésico', 'analgesico', 'antibiótico', 'antibiotico',
            'ibuprofeno', 'paracetamol', 'amoxicilina',
            'anestesia', 'lidocaína', 'lidocaina',
            
            # DIAGNÓSTICO
            'diagnosticar', 'síntoma', 'sintoma', 'signo',
            'dolor', 'molestia', 'hinchazón', 'hinchazon',
        }
    
    def es_termino_dental(self, texto):
        """
        Determina si el texto contiene términos de anatomía dental
        
        Returns:
            dict: {
                'es_dental': bool,
                'terminos_encontrados': list,
                'confianza': str,
                'razon': str
            }
        """
        texto_lower = texto.lower()
        
        # Tokenizar (simple)
        palabras = texto_lower.split()
        
        # Buscar términos dentales
        terminos_encontrados = []
        for termino in self.terminos_dentales:
            if termino in texto_lower:
                terminos_encontrados.append(termino)
        
        # Buscar términos NO dentales (procedimientos/enfermedades)
        terminos_no_dentales_encontrados = []
        for termino in self.terminos_no_dentales:
            if termino in texto_lower:
                terminos_no_dentales_encontrados.append(termino)
        
        # LÓGICA DE DECISIÓN
        
        # Caso 1: Tiene términos NO dentales (procedimientos/patología)
        if terminos_no_dentales_encontrados:
            return {
                'es_dental': False,
                'terminos_encontrados': terminos_no_dentales_encontrados,
                'confianza': 'Alta',
                'razon': 'Pregunta sobre procedimientos clínicos o patología (no anatomía)'
            }
        
        # Caso 2: Tiene términos dentales pero NO términos no dentales
        if terminos_encontrados:
            # Verificar si es pregunta anatómica
            palabras_anatomicas = ['características', 'morfología', 'anatomía', 
                                  'estructura', 'forma', 'cúspide', 'raíz', 
                                  'corona', 'diferencias', 'comparación',
                                  'cuántas', 'cómo es', 'qué es', 'describe']
            
            tiene_contexto_anatomico = any(pal in texto_lower for pal in palabras_anatomicas)
            
            if tiene_contexto_anatomico or len(terminos_encontrados) >= 2:
                return {
                    'es_dental': True,
                    'terminos_encontrados': terminos_encontrados,
                    'confianza': 'Alta',
                    'razon': 'Contiene términos de anatomía dental'
                }
            else:
                return {
                    'es_dental': True,
                    'terminos_encontrados': terminos_encontrados,
                    'confianza': 'Media',
                    'razon': 'Contiene términos dentales (posible anatomía)'
                }
        
        # Caso 3: No tiene términos dentales
        return {
            'es_dental': False,
            'terminos_encontrados': [],
            'confianza': 'Alta',
            'razon': 'No contiene términos dentales reconocibles'
        }
    
    def listar_terminos(self):
        """Lista todos los términos dentales conocidos"""
        return sorted(list(self.terminos_dentales))
    
    def agregar_termino(self, termino):
        """Permite agregar nuevos términos dinámicamente"""
        self.terminos_dentales.add(termino.lower())
        print(f"Término agregado: {termino}")
    
    def contar_terminos(self):
        """Cuenta los términos en el diccionario"""
        return {
            'terminos_dentales': len(self.terminos_dentales),
            'terminos_no_dentales': len(self.terminos_no_dentales),
            'total': len(self.terminos_dentales) + len(self.terminos_no_dentales)
        }
