import os
import yt_dlp
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, vfx, ColorClip, TextClip
from openai import OpenAI
from dotenv import load_dotenv
import json
from googleapiclient.discovery import build
import logging
import re
from pydub import AudioSegment
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import moviepy.config
import time
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from datetime import datetime
import random

#! Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#! Cargar variables de entorno
load_dotenv()

class ContentOptimizer:
    def __init__(self, openai_client=None):
        if openai_client is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("No se encontró la clave API de OpenAI")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = openai_client

    def optimize_transcription_for_social(self, transcription):
        """Genera una descripción corta optimizada a partir de la transcripción."""
        try:
            prompt = f"""Instrucciones:
            1. Analiza esta transcripción de un video de 30 segundos sobre tecnología
            2. Genera un título atractivo de MÁXIMO 40 caracteres
            3. El título debe reflejar el tema técnico principal
            4. Debe ser en español
            5. Debe ser llamativo y profesional
            6. NO uses hashtags ni emojis
            7. Mantén términos técnicos en inglés cuando sea apropiado
            8. Enfócate en conceptos de Data Science y programación
            9. NO agregues prefijos como 'Título:', 'Título sugerido:', etc.
            10. Devuelve SOLO el título, sin ningún texto adicional

            Transcripción del video:
            {transcription}"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un experto en marketing de contenido técnico, especializado en transformar transcripciones en títulos atractivos para redes sociales. Devuelve SOLO el título, sin ningún texto adicional."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )

            title = response.choices[0].message.content.strip()
            
            # Eliminar cualquier prefijo común que GPT pueda agregar
            prefixes_to_remove = ["Título:", "Título sugerido:", "Sugerencia:", "Título propuesto:"]
            for prefix in prefixes_to_remove:
                if title.startswith(prefix):
                    title = title.replace(prefix, "", 1).strip()
            
            # Asegurar que no exceda 40 caracteres
            if len(title) > 40:
                last_space = title[:37].rfind(' ')
                if last_space != -1:
                    title = title[:last_space] + "..."
                else:
                    title = title[:37] + "..."

            return title

        except Exception as e:
            logger.error(f"Error al optimizar transcripción: {str(e)}")
            # En caso de error, extraer una parte relevante de la transcripción
            words = transcription.split()[:6]  # Tomar las primeras 6 palabras
            return " ".join(words)[:37] + "..." if len(" ".join(words)) > 40 else " ".join(words)

    def generate_hashtags(self, description):
        """Genera 4 hashtags relevantes basados en la descripción."""
        try:
            prompt = f"""Instrucciones:
            1. Genera EXACTAMENTE 4 hashtags relevantes
            2. Deben estar relacionados con: {description}
            3. Enfócate en términos técnicos de Data Science y programación
            4. Usa una mezcla de español e inglés
            5. NO uses espacios en los hashtags
            6. Cada hashtag debe tener MÁXIMO 9 caracteres incluyendo el '#'
            7. Formato: #hashtag1 #hashtag2 #hashtag3 #hashtag4

            Ejemplos válidos:
            #data #code #dev #ia
            #py #ml #ds #tech

            Descripción: {description}"""

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un experto en SEO y hashtags para contenido técnico de Data Science y programación. Genera hashtags cortos y concisos."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=50
            )

            hashtags = response.choices[0].message.content.strip()
            
            # Procesar y validar cada hashtag
            processed_hashtags = []
            for hashtag in hashtags.split():
                if not hashtag.startswith('#'):
                    hashtag = '#' + hashtag
                # Limitar a 9 caracteres incluyendo el #
                if len(hashtag) > 9:
                    hashtag = hashtag[:9]
                processed_hashtags.append(hashtag)
            
            # Asegurar que tenemos exactamente 4 hashtags
            while len(processed_hashtags) < 4:
                processed_hashtags.append('#tech')
            processed_hashtags = processed_hashtags[:4]
            
            return ' '.join(processed_hashtags)

        except Exception as e:
            logger.error(f"Error al generar hashtags: {str(e)}")
            return "#ds #dev #py #ai"

class DriveUploader:
    def __init__(self):
        # IDs de Drive y Sheet
        self.DRIVE_FOLDER_ID = "1XdlovWoQNRjKN6DpOZVcSL3ThgV-L_XL"
        self.SHEET_ID = "1uLAGRvq0H-2G1RHGdzBJkhMPP6D1iWexp4N8bXDEHgk"
        
        try:
            # Cargar credenciales de la cuenta de servicio
            self.credentials = service_account.Credentials.from_service_account_file(
                'river-surf-452722-t6-24c6cdaf896b.json',
                scopes=[
                    'https://www.googleapis.com/auth/drive',
                    'https://www.googleapis.com/auth/spreadsheets',
                    'https://www.googleapis.com/auth/youtube'
                ]
            )
            
            # Inicializar servicios con la misma cuenta de servicio
            self.drive_service = build('drive', 'v3', credentials=self.credentials)
            self.sheets_service = build('sheets', 'v4', credentials=self.credentials)
            self.youtube = build('youtube', 'v3', credentials=self.credentials)
            
            logger.info("Servicios de Google inicializados correctamente con cuenta de servicio")
        except Exception as e:
            logger.error(f"Error al inicializar servicios de Google: {str(e)}")
            raise

    def upload_video_to_drive(self, video_path, max_retries=3):
        """Sube el video a Google Drive usando la cuenta de servicio."""
        retry_count = 0
        while retry_count < max_retries:
            try:
                file_metadata = {
                    'name': os.path.basename(video_path),
                    'parents': [self.DRIVE_FOLDER_ID],
                    'mimeType': 'video/mp4'
                }
                
                media = MediaFileUpload(
                    video_path, 
                    mimetype='video/mp4',
                    resumable=True
                )
                
                file = self.drive_service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id, webViewLink',
                    supportsAllDrives=True
                ).execute()
                
                # Configurar permisos para que cualquiera con el enlace pueda ver
                try:
                    self.drive_service.permissions().create(
                        fileId=file.get('id'),
                        body={
                            'type': 'anyone',
                            'role': 'reader'
                        },
                        supportsAllDrives=True
                    ).execute()
                except Exception as perm_error:
                    # Si falla al configurar permisos, intentar de nuevo
                    time.sleep(2)  # Esperar 2 segundos antes de reintentar
                    self.drive_service.permissions().create(
                        fileId=file.get('id'),
                        body={
                            'type': 'anyone',
                            'role': 'reader'
                        },
                        supportsAllDrives=True
                    ).execute()
                
                logger.info(f"Video subido exitosamente a Drive: {file.get('webViewLink')}")
                return file.get('webViewLink')
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # Espera exponencial
                    logger.warning(f"Intento {retry_count} fallido. Esperando {wait_time} segundos antes de reintentar...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error al subir video a Drive después de {max_retries} intentos: {str(e)}")
                    raise

    def update_metadata_sheet(self, video_link, optimized_title, hashtags, transcription):
        """Actualiza el Google Sheet usando la cuenta de servicio."""
        try:
            # Preparar los datos para la nueva fila
            row_data = [
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Date
                    video_link,                                     # Link
                    optimized_title,                               # Título sugerido
                    hashtags,                                      # Hashtags
                    transcription[:1000],                          # Original Text
                    "NO"                                           # Approve (por defecto NO)
                ]
            ]
            
            # Obtener el rango actual de datos
            result = self.sheets_service.spreadsheets().values().get(
                spreadsheetId=self.SHEET_ID,
                range='A:F'  # Ahora incluimos la columna F para Approve
            ).execute()
            
            values = result.get('values', [])
            
            if not values:
                # Si el sheet está vacío, agregar los títulos primero
                headers = [["Date", "Link", "Título sugerido", "Hashtags", "Original Text", "Approve"]]
                self.sheets_service.spreadsheets().values().update(
                    spreadsheetId=self.SHEET_ID,
                    range='A1:F1',
                    valueInputOption='USER_ENTERED',
                    body={'values': headers}
                ).execute()
                next_row = 2
            else:
                # Si ya hay datos, agregar después de la última fila
                next_row = len(values) + 1
            
            # Actualizar el sheet
            body = {
                'values': row_data
            }
            
            self.sheets_service.spreadsheets().values().update(
                spreadsheetId=self.SHEET_ID,
                range=f'A{next_row}:F{next_row}',
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()
            
            logger.info(f"Metadata actualizada en el Sheet, fila {next_row}")
            
        except Exception as e:
            logger.error(f"Error al actualizar Sheet: {str(e)}")
            raise

class YouTubeShortsCreator:
    def __init__(self, num_shorts=10, start_time_minutes=5):
        # Configurar OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("No se encontró la clave API de OpenAI")
        self.client = OpenAI(api_key=api_key)
        
        # Configuración de directorios
        self.output_dir = 'shorts_output'
        self.temp_dir = 'temp'
        self.audio_dir = 'audio_transcription'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        
        # Configuración de procesamiento
        self.max_duration = 30  # Duración máxima en segundos para los shorts
        self.num_shorts = num_shorts  # Número de shorts a generar
        self.start_time_seconds = start_time_minutes * 60  # Convertir minutos a segundos
        
        # Configuración de costos de API
        self.whisper_cost_per_minute = 0.006  # Costo de Whisper API por minuto
        self.gpt35_input_cost_per_1k = 0.0005
        self.gpt35_output_cost_per_1k = 0.0015
        self.estimated_tokens_per_segment = 500
        self.usd_to_cop = 4000
        
        # Tracking de costos
        self.real_costs = {
            "whisper_minutes": 0,
            "gpt_input_tokens": 0,
            "gpt_output_tokens": 0
        }
        self.detailed_costs = {
            "whisper_transcriptions": [],
            "gpt_corrections": []
        }
        
        # Configuración de calidad de audio/video
        self.max_workers = multiprocessing.cpu_count()
        self.temp_quality = {
            'audio': {
                'codec': 'mp3'  # Formato para Whisper API
            },
            'video': {
                'fps': None,
                'preset': 'medium',
                'threads': self.max_workers,
                'bitrate': None
            }
        }
        
        # Inicializar servicios
        self.drive_uploader = DriveUploader()
        self.content_optimizer = ContentOptimizer(openai_client=self.client)

    def extract_video_id(self, url):
    #!    """Extrae el ID del video de la URL de YouTube."""
        pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11})(?:&|\/|$)'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None

    
    
    def download_video(self, url):
    #!   """Descarga el video de YouTube."""
        def sanitize_filename(filename):
            # Reemplazar caracteres especiales y espacios
            filename = re.sub(r'[^\w\s-]', '', filename)
            filename = re.sub(r'[-\s]+', '_', filename)
            return filename.strip('-_')

        try:
            with yt_dlp.YoutubeDL({'format': 'best'}) as ydl:
                # Primero obtener la información sin descargar
                info = ydl.extract_info(url, download=False)
                
                # Sanitizar el título para el nombre del archivo
                safe_title = sanitize_filename(info['title'])
                output_path = os.path.join(self.output_dir, f"{safe_title}.mp4")
                
                # Configurar las opciones con el nombre de archivo sanitizado
                ydl_opts = {
                    'format': 'best',
                    'outtmpl': output_path
                }
                
                # Descargar con el nuevo nombre
                with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                    ydl_download.download([url])
                
                return {
                    'path': output_path,
                    'title': info['title'],
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0)
                }
        except Exception as e:
            logger.error(f"Error al descargar el video: {e}")
            raise

    def get_video_transcript(self, video_id):
        #!"""Obtiene la transcripción del video usando la API de YouTube."""
        try:
            captions = self.youtube.captions().list(
                part='snippet',
                videoId=video_id
            ).execute()
            
            if 'items' in captions and len(captions['items']) > 0:
                caption_id = captions['items'][0]['id']
                subtitle = self.youtube.captions().download(
                    id=caption_id,
                    tfmt='srt'
                ).execute()
                return subtitle
            return None
        except Exception as e:
            logger.warning(f"No se pudo obtener la transcripción: {e}")
            return None

    
    
    def analyze_video_content(self, video_info):
        """Extrae segmentos aleatorios del video para crear shorts."""
        video = None
        try:
            logger.info("Analizando el contenido del video...")
            logger.info(f"Intentando abrir el archivo: {video_info['path']}")
            
            # Verificar que el archivo existe
            if not os.path.exists(video_info['path']):
                logger.error(f"El archivo no existe: {video_info['path']}")
                return []
            
            # Cargar el video
            video = VideoFileClip(video_info['path'])
            
            # Extraer el segmento de video que nos interesa analizar
            start_after = self.start_time_seconds
            available_duration = video.duration - start_after - self.max_duration
            
            # Verificar si hay suficiente duración
            if available_duration < self.max_duration:
                logger.warning(f"El video no tiene suficiente duración después del minuto {self.start_time_seconds//60}")
                return []
            
            logger.info(f"Extrayendo segmentos desde el segundo {start_after}...")
            
            # Calcular cuántos segmentos podemos extraer
            max_possible_segments = int(available_duration // self.max_duration)
            
            if max_possible_segments < self.num_shorts:
                logger.warning(f"Solo se pueden extraer {max_possible_segments} segmentos del video")
                num_segments = max_possible_segments
            else:
                num_segments = self.num_shorts
            
            # Generar tiempos de inicio aleatorios
            possible_start_times = []
            current_time = start_after
            
            while current_time + self.max_duration <= video.duration:
                possible_start_times.append(current_time)
                current_time += self.max_duration
            
            # Seleccionar tiempos de inicio aleatorios
            selected_times = random.sample(possible_start_times, min(num_segments, len(possible_start_times)))
            selected_times.sort()  # Ordenar cronológicamente
            
            # Crear los segmentos
            segments = []
            for start_time in selected_times:
                segments.append({
                    "start_time": start_time,
                    "description": f"Segmento desde {start_time} hasta {start_time + self.max_duration}",
                    "duration": self.max_duration
                })
            
            if segments:
                logger.info(f"Se han seleccionado {len(segments)} segmentos aleatorios.")
                return segments
            
            return []
                
        except Exception as e:
            logger.error(f"Error al analizar el contenido: {str(e)}")
            return []
        finally:
            if video is not None:
                try:
                    video.close()
                except:
                    pass

    
    def detect_voice_segments(self, audio_path, min_duration=1.0):
        #!"""Detecta segmentos donde hay voz en el audio."""
        try:
            # Cargar el audio
            y, sr = librosa.load(audio_path)
            
            # Calcular la energía del audio
            energy = librosa.feature.rms(y=y)[0]
            
            # Calcular el umbral de energía (ajustable según necesidad)
            threshold = np.mean(energy) * 1.5
            
            # Encontrar segmentos donde la energía supera el umbral
            voice_segments = []
            is_voice = False
            start_time = 0
            
            frames_to_time = lambda x: float(x) * len(y) / (sr * len(energy))
            
            for i, e in enumerate(energy):
                if not is_voice and e > threshold:
                    start_time = frames_to_time(i)
                    is_voice = True
                elif is_voice and e <= threshold:
                    end_time = frames_to_time(i)
                    if end_time - start_time >= min_duration:
                        voice_segments.append((start_time, end_time))
                    is_voice = False
            
            return voice_segments
        except Exception as e:
            logger.error(f"Error al detectar segmentos de voz: {e}")
            return []

    def adjust_segment_to_voice(self, video_path, start_time, end_time):
        #!"""Ajusta los tiempos del segmento para comenzar con voz."""
        try:
            # Extraer el audio del segmento
            video = VideoFileClip(video_path)
            audio = video.audio
            
            # Guardar el audio temporalmente
            temp_audio_path = os.path.join(self.temp_dir, "temp_audio.wav")
            audio.write_audiofile(temp_audio_path)
            
            # Detectar segmentos de voz
            voice_segments = self.detect_voice_segments(temp_audio_path)
            
            # Encontrar el segmento de voz más cercano al inicio
            if voice_segments:
                for vs_start, vs_end in voice_segments:
                    if vs_start >= start_time and vs_start < end_time:
                        start_time = vs_start
                        break
            
            # Limpiar
            os.remove(temp_audio_path)
            video.close()
            
            return start_time, end_time
        except Exception as e:
            logger.error(f"Error al ajustar segmento a voz: {e}")
            return start_time, end_time

    
    
    def create_vertical_video(self, clip):
        #!"""Crea un video vertical manteniendo la calidad original."""
        try:
            # Dimensiones del video vertical
            target_height = 1920
            target_width = 1080
            
            # Calcular el factor de escala manteniendo la relación de aspecto
            width_scale = target_width / clip.w
            height_scale = target_height / clip.h
            scale_factor = min(width_scale, height_scale)
            
            # Redimensionar el clip manteniendo la calidad original
            scaled_clip = clip.resize(width=int(clip.w * scale_factor))
            
            # Crear fondo negro
            background = ColorClip(
                size=(target_width, target_height),
                color=(0, 0, 0),
                duration=clip.duration
            )
            
            # Posicionar el clip en el centro
            x_center = (target_width - scaled_clip.w) // 2
            y_center = (target_height - scaled_clip.h) // 2
            
            # Combinar clips
            return CompositeVideoClip(
                [background, scaled_clip.set_position((x_center, y_center))],
                size=(target_width, target_height)
            )
            
        except Exception as e:
            logger.error(f"Error al crear video vertical: {str(e)}")
            raise

    def track_whisper_usage(self, audio_duration_seconds):
        """Rastrea el uso real de Whisper."""
        minutes_used = audio_duration_seconds / 60
        self.real_costs["whisper_minutes"] += minutes_used
        cost_usd = minutes_used * self.whisper_cost_per_minute
        cost_cop = cost_usd * self.usd_to_cop
        
        # Guardar detalles de esta transcripción
        self.detailed_costs["whisper_transcriptions"].append({
            "duration_minutes": minutes_used,
            "cost_usd": cost_usd,
            "cost_cop": cost_cop,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        return cost_usd

    def track_gpt_usage(self, response):
        #!"""Rastrea el uso real de tokens de GPT-3.5."""
        if hasattr(response, 'usage'):
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            self.real_costs["gpt_input_tokens"] += input_tokens
            self.real_costs["gpt_output_tokens"] += output_tokens
            
            input_cost_usd = (input_tokens / 1000) * self.gpt35_input_cost_per_1k
            output_cost_usd = (output_tokens / 1000) * self.gpt35_output_cost_per_1k
            total_cost_usd = input_cost_usd + output_cost_usd
            
            input_cost_cop = input_cost_usd * self.usd_to_cop
            output_cost_cop = output_cost_usd * self.usd_to_cop
            total_cost_cop = total_cost_usd * self.usd_to_cop
            
            # Nuevo: Guardar detalles de esta corrección
            self.detailed_costs["gpt_corrections"].append({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost_usd": input_cost_usd,
                "output_cost_usd": output_cost_usd,
                "total_cost_usd": total_cost_usd,
                "total_cost_cop": total_cost_cop,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            logger.info("\nCosto real de GPT-3.5 para esta petición:")
            logger.info(f"  Tokens de entrada: {input_tokens}")
            logger.info(f"  Tokens de salida: {output_tokens}")
            logger.info(f"  Costo entrada USD: ${input_cost_usd:.4f}")
            logger.info(f"  Costo entrada COP: ${input_cost_cop:,.2f}")
            
            return total_cost_usd
        return 0

    def get_total_real_costs(self):
        #!"""Calcula los costos reales totales."""
        whisper_cost_usd = self.real_costs["whisper_minutes"] * self.whisper_cost_per_minute
        gpt_input_cost_usd = (self.real_costs["gpt_input_tokens"] / 1000) * self.gpt35_input_cost_per_1k
        gpt_output_cost_usd = (self.real_costs["gpt_output_tokens"] / 1000) * self.gpt35_output_cost_per_1k
        
        total_cost_usd = whisper_cost_usd + gpt_input_cost_usd + gpt_output_cost_usd
        total_cost_cop = total_cost_usd * self.usd_to_cop
        whisper_cost_cop = whisper_cost_usd * self.usd_to_cop
        gpt_input_cost_cop = gpt_input_cost_usd * self.usd_to_cop
        gpt_output_cost_cop = gpt_output_cost_usd * self.usd_to_cop
        
        return {
            "whisper_cost_usd": round(whisper_cost_usd, 4),
            "whisper_cost_cop": round(whisper_cost_cop, 2),
            "gpt_input_cost_usd": round(gpt_input_cost_usd, 4),
            "gpt_input_cost_cop": round(gpt_input_cost_cop, 2),
            "gpt_output_cost_usd": round(gpt_output_cost_usd, 4),
            "gpt_output_cost_cop": round(gpt_output_cost_cop, 2),
            "total_cost_usd": round(total_cost_usd, 4),
            "total_cost_cop": round(total_cost_cop, 2),
            "total_whisper_minutes": round(self.real_costs["whisper_minutes"], 2),
            "total_gpt_input_tokens": self.real_costs["gpt_input_tokens"],
            "total_gpt_output_tokens": self.real_costs["gpt_output_tokens"]
        }

    def correct_text_with_gpt(self, text):
        """Corrige errores ortográficos en el texto usando GPT-3.5."""
        try:
            # Si el texto es muy largo, dividirlo en fragmentos
            max_chars_per_chunk = 4000  # Aproximadamente 1000 tokens
            if len(text) > max_chars_per_chunk:
                # Dividir el texto en oraciones
                sentences = text.replace('? ', '?|').replace('! ', '!|').replace('. ', '.|').split('|')
                chunks = []
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    if current_length + sentence_length > max_chars_per_chunk:
                        # Guardar el chunk actual y empezar uno nuevo
                        chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = sentence_length
                    else:
                        current_chunk.append(sentence)
                        current_length += sentence_length
                
                # Agregar el último chunk si existe
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Corregir cada chunk por separado
                corrected_chunks = []
                for chunk in chunks:
                    try:
                        corrected_chunk = self._correct_text_chunk(chunk)
                        corrected_chunks.append(corrected_chunk)
                    except Exception as e:
                        logger.error(f"Error al corregir chunk: {str(e)}")
                        corrected_chunks.append(chunk)  # Mantener el texto original si hay error
                
                # Unir los chunks corregidos
                return ' '.join(corrected_chunks)
            else:
                # Si el texto es corto, corregirlo directamente
                return self._correct_text_chunk(text)

        except Exception as e:
            logger.error(f"Error al corregir texto con GPT-3.5: {str(e)}")
            return text

    def _correct_text_chunk(self, text):
        """Corrige un fragmento de texto usando GPT-3.5."""
        prompt = f"""Instrucciones:
        1. Corrige errores ortográficos y de puntuación en español
        2. MANTÉN sin modificar todos los términos técnicos como:
           - Data Science, Data Engineering, Data Warehouse
           - Delta Lake,Business Intelligence , Data Lakehouse , Databricks
           - Machine Learning, Deep Learning, AI
           - Lenguajes: Python, SQL, R, Java, JavaScript
           - Frameworks: TensorFlow, PyTorch, Pandas, NumPy
           - Cloud: AWS, Azure, GCP
           - Big Data: Hadoop, Spark, Kafka
        3. NO agregues ni quites información
        4. NO agregues prefijos o texto adicional
        5. Mantén el mismo tono y significado del texto original

        Texto a corregir: {text}"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un corrector ortográfico experto en español especializado en contenido técnico de Data Science y programación."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000
        )

        # Rastrear el uso de tokens
        self.track_gpt_usage(response)

        # Extraer y limpiar el texto corregido
        corrected_text = response.choices[0].message.content.strip()
        
        # Eliminar cualquier prefijo común que GPT pueda agregar
        prefixes_to_remove = ["Texto corregido:", "Texto:", "Corrección:", "Resultado:"]
        for prefix in prefixes_to_remove:
            if corrected_text.startswith(prefix):
                corrected_text = corrected_text.replace(prefix, "", 1).strip()

        return corrected_text

    def get_audio_transcription(self, clip):
        """Transcribe el audio asegurando que coincida exactamente con el contenido del video."""
        try:
            if not clip.audio:
                logger.error("El clip no tiene audio")
                return []
            
            temp_dir = os.path.join(self.temp_dir, 'audio_transcription')
            os.makedirs(temp_dir, exist_ok=True)
            
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                audio_path = os.path.join(temp_dir, f"audio_{timestamp}.mp3")
                
                # Guardar el audio con alta calidad
                clip.audio.write_audiofile(
                    audio_path,
                    codec='mp3',
                    bitrate='32k',
                    fps=44100,
                    ffmpeg_params=['-ac', '1'],
                    logger=None,
                    verbose=False
                )
                
                def transcribe_with_size(audio_file, max_retries=3):
                    """Intenta transcribir el audio, si falla lo divide en partes más pequeñas."""
                    if max_retries <= 0:
                        return []
                    
                    try:
                        with open(audio_file, 'rb') if isinstance(audio_file, str) else audio_file as f:
                            response = self.client.audio.transcriptions.create(
                                model="whisper-1",
                                file=f,
                                language="es",
                                response_format="json"  # Cambiado a json simple
                            )
                        
                        # Solo registrar el costo una vez que la transcripción es exitosa
                        self.track_whisper_usage(clip.duration)
                        
                        # Procesar la respuesta en formato json simple
                        if hasattr(response, 'text') and response.text.strip():
                            corrected_text = self.correct_text_with_gpt(response.text)
                            return [{
                                "text": corrected_text,
                                "start": 0,
                                "end": clip.duration,
                                "words": []
                            }]
                        return []
                        
                    except Exception as e:
                        logger.error(f"Error en transcripción: {str(e)}")
                        if max_retries > 1:
                            # Dividir el audio en dos partes
                            audio = AudioSegment.from_file(audio_file if isinstance(audio_file, str) else audio_file.name)
                            mid_point = len(audio) // 2
                            
                            # Guardar primera mitad
                            first_half = audio[:mid_point]
                            first_half_path = f"{audio_path}_part1.mp3"
                            first_half.export(first_half_path, format="mp3")
                            
                            # Guardar segunda mitad
                            second_half = audio[mid_point:]
                            second_half_path = f"{audio_path}_part2.mp3"
                            second_half.export(second_half_path, format="mp3")
                            
                            # Transcribir cada mitad recursivamente
                            first_transcription = transcribe_with_size(first_half_path, max_retries - 1)
                            second_transcription = transcribe_with_size(second_half_path, max_retries - 1)
                            
                            # Limpiar archivos temporales
                            try:
                                os.remove(first_half_path)
                                os.remove(second_half_path)
                            except:
                                pass
                            
                            # Combinar resultados
                            return first_transcription + second_transcription
                        return []
                
                # Intentar transcribir el archivo completo
                transcriptions = transcribe_with_size(audio_path)
                if transcriptions:
                    return transcriptions
                
                return []
                
            finally:
                # Limpiar archivos temporales
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except:
                        pass
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error general en la transcripción: {str(e)}")
            return []

    def create_subtitles(self, clip, segments):
        #!"""Crea subtítulos sincronizados con el habla con efectos llamativos."""
        try:
            subtitle_clips = []
            clip_width = clip.w
            clip_height = clip.h
            
            # Configuración de tamaño fijo para todos los subtítulos
            fontsize = min(50, int(clip_height * 3.5))
            max_width = int(clip_width * 0.85)
            # Altura fija para el contenedor de texto
            fixed_height = int(clip_height * 0.15)  # 15% de la altura del video
            
            for segment in segments:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"].strip()
                duration = end_time - start_time  # Mantener la duración original del audio
                
                # Dividir el texto en segmentos más cortos si es muy largo
                words = text.split()
                if len(words) > 7:  # Si hay más de 8 palabras, dividimos en sub-segmentos
                    sub_segments = []
                    current_segment = []
                    for word in words:
                        current_segment.append(word)
                        if len(current_segment) >= 4:
                            sub_segments.append(" ".join(current_segment))
                            current_segment = []
                    if current_segment:  # Agregar las palabras restantes
                        sub_segments.append(" ".join(current_segment))
                    
                    # Mantener la duración proporcional al audio original
                    sub_duration = duration / len(sub_segments)
                    
                    # Crear un clip para cada sub-segmento
                    for i, sub_text in enumerate(sub_segments):
                        sub_start = start_time + (i * sub_duration)
                        
                        try:
                            txt_clip = TextClip(
                                sub_text,
                                fontsize=fontsize,
                                color='white',
                                font='Arial',
                                method='label',
                                size=(max_width, fixed_height),  # Usar altura fija
                                stroke_color='white',
                                stroke_width=2.0,
                                bg_color='black'
                            )
                            
                            if txt_clip is None:
                                continue
                            
                            y_position = int(clip_height * 0.70)
                            fade_duration = min(0.15, sub_duration / 4)
                            txt_comp = (txt_clip
                                      .set_position(('center', y_position))
                                      .set_start(sub_start)
                                      .set_duration(sub_duration)
                                      .crossfadein(fade_duration)
                                      .crossfadeout(fade_duration))
                            
                            subtitle_clips.append(txt_comp)
                            
                        except Exception as e:
                            logger.error(f"Error al procesar sub-segmento: {str(e)}")
                            continue
                else:
                    try:
                        txt_clip = TextClip(
                            text,
                            fontsize=fontsize,
                            color='white',
                            font='Arial',
                            method='label',
                            size=(max_width, fixed_height),  # Usar altura fija
                            stroke_color='white',
                            stroke_width=2.0,
                            bg_color='black'
                        )
                        
                        if txt_clip is None:
                            continue
                        
                        y_position = int(clip_height * 0.70)
                        fade_duration = min(0.15, duration / 4)
                        txt_comp = (txt_clip
                                  .set_position(('center', y_position))
                                  .set_start(start_time)
                                  .set_duration(duration)  # Usar la duración original del audio
                                  .crossfadein(fade_duration)
                                  .crossfadeout(fade_duration))
                        
                        subtitle_clips.append(txt_comp)
                        
                    except Exception as e:
                        logger.error(f"Error al procesar subtítulo individual: {str(e)}")
                        continue
            
            return subtitle_clips
            
        except Exception as e:
            logger.error(f"Error al crear subtítulos: {str(e)}")
            return []

    
    def show_detailed_costs_summary(self):
        #!"""Muestra un resumen detallado de todos los costos."""
        logger.info("\n=== RESUMEN DETALLADO DE COSTOS ===")
        
        # Resumen de Whisper
        total_whisper_minutes = sum(t["duration_minutes"] for t in self.detailed_costs["whisper_transcriptions"])
        total_whisper_usd = sum(t["cost_usd"] for t in self.detailed_costs["whisper_transcriptions"])
        total_whisper_cop = total_whisper_usd * self.usd_to_cop
        
        logger.info("\nTranscripciones de Whisper:")
        logger.info(f"Número total de transcripciones: {len(self.detailed_costs['whisper_transcriptions'])}")
        logger.info(f"Total minutos procesados: {total_whisper_minutes:.2f}")
        logger.info(f"Costo total USD: ${total_whisper_usd:.4f}")
        logger.info(f"Costo total COP: ${total_whisper_cop:,.2f}")
        
        # Resumen de GPT-3.5
        total_gpt_input_tokens = sum(c["input_tokens"] for c in self.detailed_costs["gpt_corrections"])
        total_gpt_output_tokens = sum(c["output_tokens"] for c in self.detailed_costs["gpt_corrections"])
        total_gpt_usd = sum(c["total_cost_usd"] for c in self.detailed_costs["gpt_corrections"])
        total_gpt_cop = total_gpt_usd * self.usd_to_cop
        
        logger.info("\nCorrecciones de GPT-3.5:")
        logger.info(f"Número total de correcciones: {len(self.detailed_costs['gpt_corrections'])}")
        logger.info(f"Total tokens de entrada: {total_gpt_input_tokens}")
        logger.info(f"Total tokens de salida: {total_gpt_output_tokens}")
        logger.info(f"Costo total USD: ${total_gpt_usd:.4f}")
        logger.info(f"Costo total COP: ${total_gpt_cop:,.2f}")
        
        # Total general
        total_usd = total_whisper_usd + total_gpt_usd
        total_cop = total_usd * self.usd_to_cop
        
        logger.info("\n=== TOTAL GENERAL ===")
        logger.info(f"USD: ${total_usd:.4f}")
        logger.info(f"COP: ${total_cop:,.2f}")
        logger.info("============================")

    def process_video(self, url):
        """Procesa un video de YouTube y genera shorts."""
        try:
            logger.info("Iniciando procesamiento del video...")
            video_info = self.download_video(url)
            
            # Calcular y mostrar costos estimados
            video_duration_minutes = video_info['duration'] / 60
            cost_estimate = self.calculate_estimated_cost(video_duration_minutes, self.num_shorts)
            self._show_cost_estimate(cost_estimate)
            
            # Analizar contenido
            interesting_segments = self.analyze_video_content(video_info)
            
            if not interesting_segments:
                logger.warning("No se identificaron segmentos para crear shorts")
                return []
            
            # Procesar cada segmento de manera secuencial
            created_shorts = []
            video = VideoFileClip(video_info['path'])
            
            try:
                for segment in interesting_segments:
                    start_time = int(float(segment["start_time"]))
                    end_time = start_time + self.max_duration
                    
                    if end_time > video.duration:
                        end_time = int(video.duration)
                        start_time = end_time - self.max_duration
                    
                    # Extraer clip
                    clip = video.subclip(start_time, end_time)
                    
                    try:
                        # Crear versión vertical
                        vertical_clip = self.create_vertical_video(clip)
                        
                        # Obtener transcripción y crear subtítulos
                        segments = self.get_audio_transcription(clip)
                        
                        if segments:
                            # Combinar todos los segmentos de texto transcritos para este clip
                            full_transcription = " ".join([seg["text"] for seg in segments])
                            
                            # Crear subtítulos
                            subtitle_clips = self.create_subtitles(vertical_clip, segments)
                            if subtitle_clips:
                                vertical_clip = CompositeVideoClip([vertical_clip] + subtitle_clips)
                        
                            # Guardar el video
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            output_path = f"{self.audio_dir}/short_{start_time}_{end_time}_{timestamp}.mp4"
                            logger.info(f"Guardando video en {output_path}...")
                            
                            vertical_clip.write_videofile(
                                output_path,
                                codec='libx264',
                                audio_codec='aac',
                                preset=self.temp_quality['video']['preset'],
                                threads=self.temp_quality['video']['threads'],
                                ffmpeg_params=['-pix_fmt', 'yuv420p'],
                                logger=None,
                                verbose=False
                            )
                            
                            # Generar título y hashtags
                            optimized_title = self.content_optimizer.optimize_transcription_for_social(full_transcription)
                            hashtags = self.content_optimizer.generate_hashtags(full_transcription)
                            
                            # Subir a Drive y actualizar Sheet
                            video_link = self.drive_uploader.upload_video_to_drive(output_path)
                            self.drive_uploader.update_metadata_sheet(
                                video_link,
                                optimized_title,
                                hashtags,
                                full_transcription
                            )
                            
                            created_shorts.append({
                                "path": output_path,
                                "link": video_link,
                                "title": optimized_title,
                                "hashtags": hashtags,
                                "transcription": full_transcription
                            })
                            
                            logger.info(f"Short {len(created_shorts)} procesado y guardado")
                        
                    except Exception as e:
                        logger.error(f"Error al procesar segmento: {e}")
                    finally:
                        try:
                            clip.close()
                        except:
                            pass
                        try:
                            vertical_clip.close()
                        except:
                            pass
            
            finally:
                video.close()
            
            if len(created_shorts) < self.num_shorts:
                logger.warning(f"Se solicitaron {self.num_shorts} shorts pero solo se pudieron crear {len(created_shorts)}")
            
            # Mostrar costos reales y resumen detallado
            self._show_real_costs()
            self.show_detailed_costs_summary()
            
            logger.info("Proceso completado exitosamente")
            return created_shorts
            
        except Exception as e:
            logger.error(f"Error en el procesamiento del video: {e}")
            raise

    def calculate_estimated_cost(self, video_duration_minutes, num_shorts):
        #!"""Calcula el costo estimado del procesamiento del video."""
        try:
            # Tasa de cambio aproximada (1 USD = 4000 COP)
            usd_to_cop = 4000
            
            # Costo de Whisper (transcripción completa)
            whisper_cost_usd = video_duration_minutes * self.whisper_cost_per_minute
            whisper_cost_cop = whisper_cost_usd * usd_to_cop
            
            # Costo estimado de GPT-3.5 por cada short
            # Estimamos tokens de entrada y salida por cada segmento
            total_input_tokens = num_shorts * self.estimated_tokens_per_segment
            total_output_tokens = num_shorts * 100  # Estimamos 100 tokens por respuesta
            
            gpt_input_cost_usd = (total_input_tokens / 1000) * self.gpt35_input_cost_per_1k
            gpt_output_cost_usd = (total_output_tokens / 1000) * self.gpt35_output_cost_per_1k
            
            total_cost_usd = whisper_cost_usd + gpt_input_cost_usd + gpt_output_cost_usd
            
            # Convertir a COP
            gpt_input_cost_cop = gpt_input_cost_usd * usd_to_cop
            gpt_output_cost_cop = gpt_output_cost_usd * usd_to_cop
            total_cost_cop = total_cost_usd * usd_to_cop
            
            cost_details = {
                "whisper_cost_usd": round(whisper_cost_usd, 4),
                "whisper_cost_cop": round(whisper_cost_cop, 2),
                "gpt_input_cost_usd": round(gpt_input_cost_usd, 4),
                "gpt_input_cost_cop": round(gpt_input_cost_cop, 2),
                "gpt_output_cost_usd": round(gpt_output_cost_usd, 4),
                "gpt_output_cost_cop": round(gpt_output_cost_cop, 2),
                "total_cost_usd": round(total_cost_usd, 4),
                "total_cost_cop": round(total_cost_cop, 2)
            }
            
            return cost_details
            
        except Exception as e:
            logger.error(f"Error al calcular costos: {str(e)}")
            return None

    def _show_cost_estimate(self, cost_estimate):
        #!"""Muestra los costos estimados de manera formateada."""
        if cost_estimate:
            logger.info("\nCosto ESTIMADO del procesamiento:")
            logger.info("----------------------------------------")
            logger.info(f"Whisper (transcripción):")
            logger.info(f"  USD: ${cost_estimate['whisper_cost_usd']}")
            logger.info(f"  COP: ${cost_estimate['whisper_cost_cop']:,.2f}")
            logger.info(f"GPT-3.5 (entrada):")
            logger.info(f"  USD: ${cost_estimate['gpt_input_cost_usd']}")
            logger.info(f"  COP: ${cost_estimate['gpt_input_cost_cop']:,.2f}")
            logger.info(f"GPT-3.5 (salida):")
            logger.info(f"  USD: ${cost_estimate['gpt_output_cost_usd']}")
            logger.info(f"  COP: ${cost_estimate['gpt_output_cost_cop']:,.2f}")
            logger.info("----------------------------------------")
            logger.info(f"Total estimado:")
            logger.info(f"  USD: ${cost_estimate['total_cost_usd']}")
            logger.info(f"  COP: ${cost_estimate['total_cost_cop']:,.2f}")
            logger.info("----------------------------------------")

    def _show_real_costs(self):
        #!"""Muestra los costos reales del procesamiento."""
        real_costs = self.get_total_real_costs()
        logger.info("\nCostos REALES del procesamiento:")
        logger.info("========================================")
        logger.info("Whisper:")
        logger.info(f"  Minutos procesados: {real_costs['total_whisper_minutes']}")
        logger.info(f"  USD: ${real_costs['whisper_cost_usd']}")
        logger.info(f"  COP: ${real_costs['whisper_cost_cop']:,.2f}")
        logger.info("\nGPT-3.5:")
        logger.info(f"  Tokens de entrada: {real_costs['total_gpt_input_tokens']}")
        logger.info(f"  Tokens de salida: {real_costs['total_gpt_output_tokens']}")
        logger.info(f"  Costo entrada USD: ${real_costs['gpt_input_cost_usd']}")
        logger.info(f"  Costo entrada COP: ${real_costs['gpt_input_cost_cop']:,.2f}")
        logger.info(f"  Costo salida USD: ${real_costs['gpt_output_cost_usd']}")
        logger.info(f"  Costo salida COP: ${real_costs['gpt_output_cost_cop']:,.2f}")
        logger.info("----------------------------------------")
        logger.info("TOTAL REAL:")
        logger.info(f"  USD: ${real_costs['total_cost_usd']}")
        logger.info(f"  COP: ${real_costs['total_cost_cop']:,.2f}")
        logger.info("========================================")


#all: inicio del programa
def main():
    # Verificar variables de entorno
    if not os.getenv('OPENAI_API_KEY') or not os.getenv('YOUTUBE_API_KEY'):
        print("Error: Se requieren las claves de API de OpenAI y YouTube.")
        print("Por favor, crea un archivo .env con:")
        print("OPENAI_API_KEY=tu_clave_de_openai")
        print("YOUTUBE_API_KEY=tu_clave_de_youtube")
        return

    #all: Configuración personalizable
    num_shorts = 1  # Número de shorts que quieres generar
    start_time_minutes = 10  # Minuto desde donde empezar a analizar el video
    
    # Crear instancia del creador de shorts
    creator = YouTubeShortsCreator(num_shorts=num_shorts, start_time_minutes=start_time_minutes)
    url = "https://www.youtube.com/watch?v=JzoXW7_aoag&t=489s"
    
    try:
        print("\nConfiguración:")
        print(f"- Número de shorts: {num_shorts}")
        print(f"- Tiempo de inicio: {start_time_minutes} minutos")
        print("\nIniciando procesamiento...")
        
        shorts = creator.process_video(url)
        print("\n¡Proceso completado!")
        print(f"Shorts generados (comenzando desde el minuto {start_time_minutes}):")
        for i, short in enumerate(shorts, 1):
            print(f"\nShort {i} de {num_shorts}:")
            print(f"Ubicación: {short['path']}")
            print(f"Descripción: {short['transcription']}")
    except Exception as e:
        print(f"Error durante el proceso: {e}")

if __name__ == "__main__":
    main() 