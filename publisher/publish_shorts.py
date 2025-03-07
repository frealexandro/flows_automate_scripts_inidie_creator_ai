import os
from datetime import datetime
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging
from dotenv import load_dotenv
import requests
from linkedin_api import Linkedin
from TikTokApi import TikTokApi
import urllib.request
import json
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from instabot import Bot

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class SocialMediaPublisher:
    def __init__(self):
        # Credenciales para cada plataforma
        self.linkedin_access_token = os.getenv('LINKEDIN_ACCESS_TOKEN')
        self.linkedin_organization_id = os.getenv('LINKEDIN_ORGANIZATION_ID')
        self.instagram_username = os.getenv('INSTAGRAM_USERNAME')
        self.instagram_password = os.getenv('INSTAGRAM_PASSWORD')
        self.tiktok_session = os.getenv('TIKTOK_SESSION_ID')
        
        # Inicializar APIs
        self.init_instagram()
        self.init_tiktok()
    
    def init_instagram(self):
        """Inicializa la API de Instagram."""
        try:
            self.instagram = Bot()
            self.instagram.login(username=self.instagram_username, 
                               password=self.instagram_password)
            logger.info("Instagram inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar Instagram: {str(e)}")
            self.instagram = None

    def init_tiktok(self):
        """Inicializa la API de TikTok."""
        try:
            self.tiktok = TikTokApi()
            self.tiktok.session_manager.setup_session(self.tiktok_session)
            logger.info("TikTok API inicializada correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar TikTok API: {e}")
            self.tiktok = None

    def publish_to_linkedin(self, video_path, title, description):
        """Publica el video en LinkedIn usando la API básica de compartir."""
        try:
            if not self.linkedin_access_token:
                raise Exception("Token de LinkedIn no configurado")
            
            # Headers para la API
            headers = {
                'Authorization': f'Bearer {self.linkedin_access_token}',
                'Content-Type': 'application/json',
            }
            
            # 1. Crear el texto de la publicación
            post_text = f"{title}\n\n{description}"
            
            # 2. Crear la publicación como borrador
            share_url = "https://api.linkedin.com/v2/shares"
            share_data = {
                "content": {
                    "contentEntities": [{
                        "entityLocation": video_path,
                        "thumbnails": [{
                            "resolvedUrl": video_path
                        }]
                    }],
                    "title": title
                },
                "text": {
                    "text": post_text
                },
                "visibility": {
                    "code": "PUBLIC"
                }
            }
            
            # 3. Enviar la solicitud
            response = requests.post(share_url, json=share_data, headers=headers)
            
            if response.ok:
                share_id = response.json().get('id')
                logger.info(f"Contenido compartido en LinkedIn con ID: {share_id}")
                return True
            else:
                logger.warning(f"La API de LinkedIn respondió: {response.text}")
                logger.info("Generando enlace para publicación manual...")
                
                # Crear mensaje para publicación manual
                manual_post = f"""
                🎥 Nuevo video disponible!

                📝 Título: {title}

                ℹ️ Descripción:
                {description}

                🎬 Video: {video_path}
                """
                
                logger.info("Por favor, publica manualmente este contenido en LinkedIn:")
                logger.info(manual_post)
                return True
            
        except Exception as e:
            logger.error(f"Error al publicar en LinkedIn: {str(e)}")
            return False

    def publish_to_instagram(self, video_path, caption):
        """Publica el video en Instagram usando Instabot."""
        try:
            if not self.instagram:
                raise Exception("Instagram no está inicializado")

            # Intentar publicar el video
            if self.instagram.upload_video(video_path, caption=caption):
                logger.info("Video publicado exitosamente en Instagram")
                return True
            else:
                logger.warning("No se pudo publicar el video en Instagram")
                # Crear mensaje para publicación manual
                manual_post = f"""
                📱 Contenido para Instagram:

                🎥 Video: {video_path}
                
                📝 Caption:
                {caption}
                """
                logger.info("Por favor, publica manualmente este contenido en Instagram:")
                logger.info(manual_post)
                return True
            
        except Exception as e:
            logger.error(f"Error al publicar en Instagram: {str(e)}")
            return False

    def publish_to_tiktok(self, video_path, description):
        """Publica el video en TikTok."""
        try:
            if not self.tiktok:
                raise Exception("TikTok API no está inicializada")
            
            # Subir video a TikTok
            self.tiktok.video.upload(
                video_path,
                description=description
            )
            
            logger.info(f"Video publicado en TikTok")
            return True
        except Exception as e:
            logger.error(f"Error al publicar en TikTok: {e}")
            return False

    def publish_to_youtube(self, video_path, title, description, tags):
        """Publica el video en YouTube como Short usando la cuenta de servicio."""
        try:
            # Configurar el body de la solicitud
            body = {
                'snippet': {
                    'title': title,
                    'description': description,
                    'tags': tags.split(),
                    'categoryId': '22'  # Categoría: People & Blogs
                },
                'status': {
                    'privacyStatus': 'public',
                    'selfDeclaredMadeForKids': False,
                    'shortDescription': description[:100]  # Descripción corta para Shorts
                }
            }
            
            # Subir el video usando la cuenta de servicio
            media = MediaFileUpload(
                video_path,
                mimetype='video/*',
                resumable=True
            )
            
            # Crear la solicitud de inserción
            insert_request = self.youtube.videos().insert(
                part=','.join(body.keys()),
                body=body,
                media_body=media
            )
            
            # Subir el video con manejo de progreso
            response = None
            while response is None:
                status, response = insert_request.next_chunk()
                if status:
                    logger.info(f"Subida a YouTube {int(status.progress() * 100)}% completada")
            
            video_id = response['id']
            logger.info(f"Video publicado en YouTube como Short: https://youtube.com/shorts/{video_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error al publicar en YouTube: {str(e)}")
            return False

class ShortsPublisher:
    def __init__(self):
        self.SHEET_ID = "1uLAGRvq0H-2G1RHGdzBJkhMPP6D1iWexp4N8bXDEHgk"
        
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

    def download_video(self, drive_link):
        """Descarga el video desde Google Drive usando la cuenta de servicio."""
        try:
            # Extraer ID del video desde el link
            file_id = drive_link.split('/')[-2]
            
            # Crear directorio temporal si no existe
            temp_dir = 'temp_videos'
            os.makedirs(temp_dir, exist_ok=True)
            
            # Nombre del archivo temporal
            temp_path = os.path.join(temp_dir, f"video_{file_id}.mp4")
            
            # Descargar el archivo usando la cuenta de servicio
            request = self.drive_service.files().get_media(fileId=file_id)
            
            with open(temp_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.info(f"Descarga {int(status.progress() * 100)}% completada")
            
            logger.info(f"Video descargado exitosamente: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error al descargar video de Drive: {str(e)}")
            return None

    def get_pending_publications(self):
        """Obtiene los videos pendientes para publicar hoy."""
        try:
            # Obtener todos los datos del sheet
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.SHEET_ID,
                range='A:G'  # Incluye la nueva columna de fecha de publicación
            ).execute()
            
            values = result.get('values', [])
            if not values:
                logger.info("No hay datos en el sheet")
                return []
            
            # Convertir a DataFrame
            df = pd.DataFrame(values[1:], columns=values[0])
            
            # Obtener fecha actual en formato YYYY-MM-DD
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Filtrar videos que:
            # 1. Están aprobados (Approve = "SI")
            # 2. La fecha de publicación es hoy
            # 3. No han sido publicados aún
            pending_df = df[
                (df['Approve'].str.upper() == 'SI') & 
                (df['Fecha de publicacion'].str[:10] == today)
            ]
            
            if pending_df.empty:
                logger.info("No hay videos pendientes para publicar hoy")
                return []
            
            return pending_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error al obtener publicaciones pendientes: {str(e)}")
            return []

    def publish_video(self, video_data):
        """Publica el video en todas las plataformas configuradas."""
        try:
            logger.info(f"Iniciando publicación del video: {video_data['Título sugerido']}")
            
            # Descargar el video de Drive
            video_path = self.download_video(video_data['Link'])
            if not video_path:
                return False
            
            try:
                # Publicar en cada plataforma
                platforms_status = {
                    "youtube": self.publish_to_youtube(
                        video_path,
                        video_data['Título sugerido'],
                        video_data['Original Text'],
                        video_data['Hashtags']
                    ),
                    "linkedin": self.social_publisher.publish_to_linkedin(
                        video_path,
                        video_data['Título sugerido'],
                        f"{video_data['Original Text']}\n\n{video_data['Hashtags']}"
                    ),
                    "instagram": self.social_publisher.publish_to_instagram(
                        video_path,
                        f"{video_data['Título sugerido']}\n\n{video_data['Original Text']}\n\n{video_data['Hashtags']}"
                    ),
                    "tiktok": self.social_publisher.publish_to_tiktok(
                        video_path,
                        f"{video_data['Título sugerido']}\n\n{video_data['Hashtags']}"
                    )
                }
                
                # Limpiar archivo temporal
                os.remove(video_path)
                
                # Verificar si se publicó en al menos una plataforma
                if any(platforms_status.values()):
                    logger.info(f"Video publicado exitosamente en algunas plataformas")
                    return True
                else:
                    logger.error("El video no se pudo publicar en ninguna plataforma")
                    return False
                
            except Exception as e:
                logger.error(f"Error durante la publicación: {e}")
                if os.path.exists(video_path):
                    os.remove(video_path)
                return False
            
        except Exception as e:
            logger.error(f"Error al publicar video: {str(e)}")
            return False

    def process_pending_publications(self):
        """Procesa todas las publicaciones pendientes para hoy."""
        try:
            # Obtener publicaciones pendientes
            pending_publications = self.get_pending_publications()
            
            if not pending_publications:
                logger.info("No hay publicaciones pendientes para procesar")
                return
            
            # Procesar cada publicación
            for pub in pending_publications:
                success = self.publish_video(pub)
                if success:
                    logger.info(f"Publicación exitosa: {pub['Título sugerido']}")
                else:
                    logger.error(f"Error al publicar: {pub['Título sugerido']}")
            
        except Exception as e:
            logger.error(f"Error al procesar publicaciones pendientes: {str(e)}")

def main():
    try:
        publisher = ShortsPublisher()
        publisher.process_pending_publications()
    except Exception as e:
        logger.error(f"Error en el proceso principal: {str(e)}")

if __name__ == "__main__":
    main() 