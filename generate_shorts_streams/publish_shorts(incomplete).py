class ShortsPublisher:
    def __init__(self):
        self.SHEET_ID = "1uLAGRvq0H-2G1RHGdzBJkhMPP6D1iWexp4N8bXDEHgk"
        self.AUDIO_TRANSCRIPTION_DIR = "/home/frealexandro/proyectos_personales/automate_scripts/audio_transcription"
        self.SHORTS_OUTPUT_DIR = "/home/frealexandro/proyectos_personales/automate_scripts/shorts_output"
        
        try:
            # Cargar credenciales de la cuenta de servicio
            self.credentials = service_account.Credentials.from_service_account_file(
                'river-surf-452722-t6-d6bacb04e3e9.json',
                scopes=[
                    'https://www.googleapis.com/auth/drive',
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/youtube.upload'  # Scope específico para subir videos
                ]
            )
            
            # Inicializar servicios
            self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            self.youtube = build('youtube', 'v3', credentials=self.credentials)
            self.social_publisher = SocialMediaPublisher()
            
            logger.info("Servicios inicializados correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar servicios: {str(e)}")
            raise

    def clean_directories(self):
        """Limpia los directorios de trabajo eliminando todos los archivos."""
        try:
            # Limpiar directorio de transcripciones
            if os.path.exists(self.AUDIO_TRANSCRIPTION_DIR):
                for file in os.listdir(self.AUDIO_TRANSCRIPTION_DIR):
                    file_path = os.path.join(self.AUDIO_TRANSCRIPTION_DIR, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            logger.info(f"Archivo eliminado: {file_path}")
                    except Exception as e:
                        logger.error(f"Error al eliminar {file_path}: {str(e)}")

            # Limpiar directorio de shorts
            if os.path.exists(self.SHORTS_OUTPUT_DIR):
                for file in os.listdir(self.SHORTS_OUTPUT_DIR):
                    file_path = os.path.join(self.SHORTS_OUTPUT_DIR, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                            logger.info(f"Archivo eliminado: {file_path}")
                    except Exception as e:
                        logger.error(f"Error al eliminar {file_path}: {str(e)}")

            logger.info("Directorios limpiados exitosamente")
        except Exception as e:
            logger.error(f"Error al limpiar directorios: {str(e)}")

def main():
    try:
        publisher = ShortsPublisher()
        publisher.process_pending_publications()
        
        # Limpiar directorios al finalizar
        publisher.clean_directories()
        logger.info("Proceso completado y directorios limpiados")
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")
        # Intentar limpiar directorios incluso si hubo error
        try:
            publisher.clean_directories()
        except:
            pass 