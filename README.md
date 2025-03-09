# 🤖 Flujos de Automatización con AI Open Source

## 📝 Descripción
Este repositorio contiene una colección de flujos de automatización implementados con Python y herramientas de AI de código abierto. El objetivo es proporcionar alternativas gratuitas y escalables a las plataformas no-code tradicionales.

## 🎯 Propósito
- Facilitar la automatización de operaciones manuales
- Ofrecer soluciones sin límites de uso
- Proporcionar alternativas gratuitas a herramientas como Zapier, n8n, Make, etc.
- Evitar costos de suscripción insostenibles
- Mejorar la escalabilidad de tus automatizaciones

## 💡 Ventajas
- **Código Abierto**: Todo el código es transparente y modificable
- **Sin Costos Recurrentes**: No hay suscripciones mensuales
- **Escalable**: Puedes adaptar y mejorar los flujos según tus necesidades
- **Documentación Clara**: Cada flujo incluye instrucciones paso a paso
- **Infraestructura Ligera**: No requiere recursos costosos

## 🚀 Cómo Empezar
1. Clona este repositorio
2. Revisa la documentación del flujo que te interese
3. Sigue las instrucciones paso a paso
4. ¡Ejecuta tu automatización!

## 🤝 Ayuda y Soporte
Se recomienda usar [Cursor](https://cursor.com/) o un chat de AI para entender mejor el código. Cada flujo está documentado de manera explícita para facilitar su comprensión.

## 🎯 Público Objetivo
- Emprendedores
- Startups
- Creadores Indie
- Desarrolladores
- Cualquier persona que busque automatizar tareas sin costos recurrentes

## 📢 Nota Importante
Este proyecto nace como respuesta a la problemática común de las herramientas no-code:
- Altos costos de suscripción
- Limitaciones en escalabilidad
- Dependencia de infraestructura externa
- Restricciones en personalización

## 🤝 Contribuciones
¡Las contribuciones son bienvenidas! Si tienes un flujo de automatización que quieras compartir, no dudes en crear un pull request.

## 📝 Licencia
Este proyecto es de código abierto y está disponible bajo la licencia MIT.

# YouTube Shorts Creator

Este script automatiza la creación de shorts de YouTube a partir de videos más largos, incluyendo transcripción automática y subtítulos.

## Características Principales

- Extrae segmentos aleatorios de videos largos
- Convierte videos a formato vertical para shorts
- Transcribe audio usando OpenAI Whisper
- Genera subtítulos sincronizados
- Sube automáticamente a Google Drive
- Actualiza metadata en Google Sheets
- Optimiza títulos y genera hashtags relevantes

## Requisitos

1. Python 3.8 o superior
2. ffmpeg instalado en el sistema
3. Cuenta de Google Cloud con APIs habilitadas:
   - Google Drive API
   - Google Sheets API
4. Clave API de OpenAI

## Instalación

1. Instalar ffmpeg:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install ffmpeg

# Windows
# Descargar de https://ffmpeg.org/download.html
```

2. Instalar dependencias de Python:
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno:
Crear un archivo `.env` con:
```
OPENAI_API_KEY=tu_clave_de_openai
YOUTUBE_API_KEY=tu_clave_de_youtube
```

4. Configurar credenciales de Google:
- Colocar el archivo de credenciales de servicio de Google Cloud en el directorio raíz

5. Crear directorios necesarios:
```bash
mkdir shorts_output temp audio_transcription
```

## Uso

```python
from youtube_shorts_creator import YouTubeShortsCreator

# Configurar parámetros
num_shorts = 15  # Número de shorts a generar
start_time_minutes = 10  # Minuto desde donde empezar

# Crear instancia
creator = YouTubeShortsCreator(num_shorts=num_shorts, start_time_minutes=start_time_minutes)

# Procesar video
url = "URL_DEL_VIDEO_DE_YOUTUBE"
shorts = creator.process_video(url)
```

## Costos Aproximados (basado en los logs)

- Whisper API:
  - ~$0.045 USD por 7.5 minutos de audio
  - Aproximadamente $0.006 USD por minuto

- GPT-3.5:
  - ~$0.0024 USD por tokens de entrada
  - ~$0.0022 USD por tokens de salida
  - Total aproximado por 15 shorts: $0.0496 USD

## Estructura de Directorios

```
├── shorts_output/      # Videos descargados
├── temp/              # Archivos temporales
└── audio_transcription/ # Shorts generados
```

## Limitaciones

- El video debe tener suficiente duración para extraer los segmentos solicitados
- Cada short tiene una duración fija de 30 segundos
- Se requiere conexión a internet para las APIs de OpenAI y Google
