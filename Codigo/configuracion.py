class ConfiguracionDentalBot:
    """Configuración centralizada del proyecto DentalBot"""

    # Parámetros de Random Forest 
    numero_arboles = 300  # Aumentado para mejor precisión
    profundidad_maxima = 20  # Aumentado para capturar más patrones
    muestras_minimas_division = 3  # Reducido para ser más sensible
    muestras_minimas_hoja = 1  # Reducido para mejor detalle
    estado_aleatorio = 42

    # Parámetros de TF-IDF 
    max_caracteristicas = 8000  # Aumentado para más vocabulario
    min_frecuencia_documento = 1  # Reducido para capturar palabras raras
    max_frecuencia_documento = 0.85  # Ajustado
    rango_ngramas = (1, 2)  # Reducido a bigramas para evitar overfitting

    # Parámetros de validación
    validacion_cruzada = 5
    tamaño_prueba = 0.2

    # Temas específicos de anatomía dental
    temas_dentales = {
        'dientes_permanentes': ['incisivos', 'caninos', 'premolares', 'molares'],
        'dientes_temporales': ['dentición decidua', 'dientes de leche', 'temporales'],
        'clasificacion': ['numeración FDI', 'nomenclatura anatómica', 'clasificación'],
        'caracteristicas': ['morfología dental', 'anatomía dental', 'estructura'],
        'comparaciones': ['superiores vs inferiores', 'derechos vs izquierdos']
    }

    def __str__(self):
        return f"Configuración DentalBot: {self.numero_arboles} árboles, {self.max_caracteristicas} características TF-IDF"