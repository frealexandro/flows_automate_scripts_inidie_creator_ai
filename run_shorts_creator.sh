#!/bin/bash

# Ruta al directorio del proyecto
cd /home/frealexandro/proyectos_personales/automate_scripts

# Activar el entorno virtual
source /home/frealexandro/proyectos_personales/automate_scripts/venv/bin/activate

# Ejecutar el script
python youtube_shorts_creator.py >> /home/frealexandro/proyectos_personales/automate_scripts/shorts_creator.log 2>&1

# Desactivar el entorno virtual
deactivate 