#!/bin/bash

# Ruta al directorio del proyecto
cd /home/frealexandro/proyectos_personales/automate_scripts

# Activar el entorno virtual
source /home/frealexandro/proyectos_personales/automate_scripts/venv/bin/activate

# Ejecutar el script
python publish_shorts.py >> /home/frealexandro/proyectos_personales/automate_scripts/publisher.log 2>&1

# Desactivar el entorno virtual
deactivate 