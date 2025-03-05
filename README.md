# Creador de YouTube Shorts con IA

Este script permite crear shorts de YouTube automáticamente utilizando inteligencia artificial para identificar los momentos más interesantes de un video.

## Características

- Descarga videos de YouTube
- Usa GPT-4 Vision para analizar el contenido
- Identifica automáticamente los momentos más interesantes
- Convierte al formato vertical de shorts (9:16)
- Genera múltiples shorts de un solo video

## Requisitos

- Python 3.8 o superior
- Clave de API de OpenAI
- Clave de API de YouTube

## Instalación

1. Clonar el repositorio:
```bash
git clone [url-del-repositorio]
cd [nombre-del-directorio]
```

2. Crear un entorno virtual:
```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

4. Configurar las claves de API:
Crear un archivo `.env` con el siguiente contenido:
```
OPENAI_API_KEY=tu_clave_de_openai
YOUTUBE_API_KEY=tu_clave_de_youtube
```

## Uso

1. Activar el entorno virtual:
```bash
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Ejecutar el script:
```bash
python youtube_shorts_creator.py
```

3. Ingresar la URL del video de YouTube cuando se solicite.

4. Los shorts generados se guardarán en la carpeta `shorts_output/`

## Cómo funciona

1. El script descarga el video de YouTube
2. Utiliza GPT-4 Vision para analizar el contenido y encontrar momentos interesantes
3. Recorta los segmentos seleccionados
4. Convierte cada segmento al formato vertical de shorts
5. Guarda los shorts generados

## Notas

- La duración de los shorts generados está entre 15 y 60 segundos
- El script mantiene la calidad original del video
- Se recomienda usar videos con buena calidad de origen 