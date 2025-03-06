import os
import yt_dlp
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, vfx, ColorClip, TextClip
import openai
from dotenv import load_dotenv
import json
from googleapiclient.discovery import build
import logging
import re
import librosa
import numpy as np
import soundfile as sf
import whisper
from pydub import AudioSegment
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import moviepy.config
import time

#! Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#! Cargar variables de entorno
load_dotenv()

class YouTubeShortsCreator:
    def __init__(self, num_shorts=10, start_time_minutes=5):
        #! Configurar OpenAI
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("No se encontró la clave API de OpenAI")
        self.youtube = build('youtube', 'v3', developerKey=os.getenv('YOUTUBE_API_KEY'))
        self.output_dir = 'shorts_output'
        self.temp_dir = 'temp'
        self.audio_dir = 'audio_transcription'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.audio_dir, exist_ok=True)
        self.max_duration = 30  # Duración máxima en segundos para los shorts
        self.num_shorts = num_shorts  # Número de shorts a generar
        self.start_time_seconds = start_time_minutes * 60  # Convertir minutos a segundos
        self.whisper_cost_per_minute = 0.006
        self.gpt35_input_cost_per_1k = 0.0005
        self.gpt35_output_cost_per_1k = 0.0015
        self.estimated_tokens_per_segment = 500
        self.real_costs = {
            "whisper_minutes": 0,
            "gpt_input_tokens": 0,
            "gpt_output_tokens": 0
        }
        # Nuevo: Rastreo detallado de consumos
        self.detailed_costs = {
            "whisper_transcriptions": [],
            "gpt_corrections": []
        }
        self.usd_to_cop = 4000  # Tasa de cambio aproximada
        self.max_workers = multiprocessing.cpu_count()  # Usar todos los núcleos disponibles
        self.temp_quality = {
            'audio': {
                'fps': 44100,
                'nbytes': 2,
                'codec': 'pcm_s16le'
            },
            'video': {
                'fps': None,  # Mantener fps original
                'preset': 'medium',
                'threads': self.max_workers,
                'bitrate': None  # Mantener bitrate original
            }
        }

    
    
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
        #!""Analiza el contenido del video usando Whisper y GPT-3.5 para identificar momentos interesantes."""
        video = None
        analysis_clip = None
        try:
            logger.info("Analizando el contenido del video con Whisper...")
            logger.info(f"Intentando abrir el archivo: {video_info['path']}")
            
            # Verificar que el archivo existe
            if not os.path.exists(video_info['path']):
                logger.error(f"El archivo no existe: {video_info['path']}")
                return []
            
            # Cargar el video
            video = VideoFileClip(video_info['path'])
            
            # Extraer el segmento de video que nos interesa analizar
            start_after = self.start_time_seconds
            available_duration = video.duration - start_after
            
            # Verificar si hay suficiente duración
            if available_duration < self.max_duration:
                logger.warning(f"El video no tiene suficiente duración después del minuto {self.start_time_seconds//60}")
                return []
            
            logger.info(f"Extrayendo segmento de video desde el segundo {start_after}...")
            # Extraer el segmento de video que queremos analizar
            analysis_clip = video.subclip(start_after, video.duration)
            
            # Cargar el modelo de Whisper si no está cargado
            if not hasattr(self, 'whisper_model'):
                logger.info("Cargando modelo de Whisper...")
                self.whisper_model = whisper.load_model("base")
            
            # Crear un archivo temporal para el audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                logger.info("Extrayendo audio a archivo temporal...")
                analysis_clip.audio.write_audiofile(
                    temp_audio_path,
                    fps=16000,
                    nbytes=2,
                    codec='pcm_s16le',
                    ffmpeg_params=['-ac', '1']  # Forzar mono
                )
                
                # Cargar el audio con librosa
                logger.info("Cargando audio para transcripción...")
                audio_array, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
                
                # Asegurarse de que el audio esté en el formato correcto
                audio_array = audio_array.astype(np.float32)
                audio_array = audio_array / np.max(np.abs(audio_array))
                
                logger.info("Transcribiendo audio con Whisper...")
                result = self.whisper_model.transcribe(audio_array, language="es")
                segments = result.get("segments", [])
                logger.info(f"Transcripción completada. Se encontraron {len(segments)} segmentos.")
                
                # Limpiar archivo temporal
                os.unlink(temp_audio_path)
                
                if not segments:
                    logger.warning("No se encontraron segmentos de audio para transcribir")
                    return []
                
                # Procesar los segmentos y encontrar momentos interesantes
                all_moments = []
                for segment in segments:
                    if segment["end"] - segment["start"] >= 3:
                        all_moments.append({
                            "start_time": start_after + segment["start"],
                            "description": segment["text"]
                        })
                
                # Ordenar por tiempo de inicio
                all_moments.sort(key=lambda x: x["start_time"])
                
                # Seleccionar momentos diferentes asegurando que estén separados por al menos max_duration
                selected_moments = []
                last_time = -float('inf')
                
                for moment in all_moments:
                    if moment["start_time"] - last_time >= self.max_duration:
                        selected_moments.append(moment)
                        last_time = moment["start_time"]
                        if len(selected_moments) >= self.num_shorts:
                            break
                
                if selected_moments:
                    logger.info(f"Análisis completado. Se identificaron {len(selected_moments)} momentos interesantes.")
                    return selected_moments[:self.num_shorts]
                
                return []
                
        except Exception as e:
            logger.error(f"Error al analizar el contenido: {str(e)}")
            return []
        finally:
            # Cerrar los recursos
            if analysis_clip is not None:
                try:
                    analysis_clip.close()
                except:
                    pass
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
        
        # Nuevo: Guardar detalles de esta transcripción
        self.detailed_costs["whisper_transcriptions"].append({
            "duration_minutes": minutes_used,
            "cost_usd": cost_usd,
            "cost_cop": cost_cop,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        logger.info("\nCosto real de Whisper para este segmento:")
        logger.info(f"  Duración: {minutes_used:.2f} minutos")
        logger.info(f"  USD: ${cost_usd:.4f}")
        logger.info(f"  COP: ${cost_cop:,.2f}")
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
        
        return {
            "whisper_cost_usd": round(whisper_cost_usd, 4),
            "whisper_cost_cop": round(whisper_cost_usd * self.usd_to_cop, 2),
            "gpt_input_cost_usd": round(gpt_input_cost_usd, 4),
            "gpt_input_cost_cop": round(gpt_input_cost_usd * self.usd_to_cop, 2),
            "gpt_output_cost_usd": round(gpt_output_cost_usd, 4),
            "gpt_output_cost_cop": round(gpt_output_cost_usd * self.usd_to_cop, 2),
            "total_cost_usd": round(total_cost_usd, 4),
            "total_cost_cop": round(total_cost_cop, 2),
            "total_whisper_minutes": round(self.real_costs["whisper_minutes"], 2),
            "total_gpt_input_tokens": self.real_costs["gpt_input_tokens"],
            "total_gpt_output_tokens": self.real_costs["gpt_output_tokens"]
        }

    def correct_text_with_gpt(self, text):
        #!"""Corrige errores ortográficos en el texto usando GPT-3.5."""
        try:
            prompt = f"""Instrucciones:
            1. Corrige errores ortográficos y de puntuación en español
            2. MANTÉN sin modificar todos los términos técnicos como:
               - Data Science, Data Engineering, Data Warehouse
               - Delta Lake, Data Lake, Business Intelligence
               - Machine Learning, Deep Learning, AI
               - Lenguajes: Python, SQL, R, Java, JavaScript
               - Frameworks: TensorFlow, PyTorch, Pandas, NumPy
               - Cloud: AWS, Azure, GCP
               - Big Data: Hadoop, Spark, Kafka
            3. NO agregues ni quites información
            4. NO agregues prefijos o texto adicional
            5. Mantén el mismo tono y significado del texto original

            Texto a corregir: {text}"""

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un corrector ortográfico experto en español especializado en contenido técnico de Data Science y programación."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
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

            logger.info("Texto corregido con GPT-3.5")
            return corrected_text

        except Exception as e:
            logger.error(f"Error al corregir texto con GPT-3.5: {str(e)}")
            return text

    def get_audio_transcription(self, clip):
        #!"""Transcribe el audio directamente desde el clip de video usando Whisper."""
        temp_audio_path = None
        try:
            # Verificar que el clip tenga audio
            if not clip.audio:
                logger.error("El clip no tiene audio")
                return []
            
            # Cargar el modelo de Whisper si no está cargado
            if not hasattr(self, 'whisper_model'):
                self.whisper_model = whisper.load_model("base")
            
            # Crear archivo de audio en la carpeta audio_transcription
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            audio_filename = f"audio_{timestamp}.wav"
            temp_audio_path = os.path.join(self.audio_dir, audio_filename)
            logger.info(f"Extrayendo audio a archivo: {temp_audio_path}")
            
            # Configurar moviepy para usar ffmpeg directamente
            try:
                clip.audio.write_audiofile(
                    temp_audio_path,
                    fps=16000,
                    codec='pcm_s16le',
                    ffmpeg_params=['-ac', '1'],  # Forzar mono
                    logger=None,
                    verbose=False
                )
            except Exception as e:
                logger.error(f"Error al escribir el archivo de audio: {e}")
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                return []
            
            # Verificar que el archivo se creó correctamente
            if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
                logger.error("El archivo de audio no se creó correctamente")
                return []
            
            # Cargar el audio con librosa
            try:
                audio_array, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
            except Exception as e:
                logger.error(f"Error al cargar el audio con librosa: {e}")
                return []
            
            # Normalizar el audio
            audio_array = audio_array.astype(np.float32)
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Transcribir con Whisper
            result = self.whisper_model.transcribe(audio_array, language="es")
            
            # Rastrear uso de Whisper
            self.track_whisper_usage(clip.duration)
            
            segments = result.get("segments", [])
            logger.info(f"Se encontraron {len(segments)} segmentos en la transcripción")
            
            # Corregir cada segmento con GPT-3.5
            corrected_segments = []
            for segment in segments:
                try:
                    corrected_text = self.correct_text_with_gpt(segment["text"])
                    segment["text"] = corrected_text
                    segment["start"] = float(segment["start"])
                    segment["end"] = float(segment["end"])
                    corrected_segments.append(segment)
                except Exception as e:
                    logger.error(f"Error al corregir segmento: {e}")
                    continue
            
            return corrected_segments
            
        except Exception as e:
            logger.error(f"Error general en la transcripción: {str(e)}")
            return []
        finally:
            # Limpiar archivo temporal si existe
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception as e:
                    logger.error(f"Error al eliminar archivo temporal: {e}")

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
        #!"""Procesa un video de YouTube y genera shorts."""
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
                            subtitle_clips = self.create_subtitles(vertical_clip, segments)
                            if subtitle_clips:
                                vertical_clip = CompositeVideoClip([vertical_clip] + subtitle_clips)
                        
                        # Guardar el video en la carpeta audio_transcription
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
                        
                        created_shorts.append({
                            "path": output_path,
                            "description": segment["description"]
                        })
                        
                        logger.info(f"Short creado exitosamente: {output_path}")
                        
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
            
            # Costo estimado de GPT-3.5 por cada short
            # Estimamos tokens de entrada y salida por cada segmento
            total_input_tokens = num_shorts * self.estimated_tokens_per_segment
            total_output_tokens = num_shorts * 100  # Estimamos 100 tokens por respuesta
            
            gpt_input_cost_usd = (total_input_tokens / 1000) * self.gpt35_input_cost_per_1k
            gpt_output_cost_usd = (total_output_tokens / 1000) * self.gpt35_output_cost_per_1k
            
            total_cost_usd = whisper_cost_usd + gpt_input_cost_usd + gpt_output_cost_usd
            
            # Convertir a COP
            whisper_cost_cop = whisper_cost_usd * usd_to_cop
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
    num_shorts = 10  # Número de shorts que quieres generar
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
            print(f"Descripción: {short['description']}")
    except Exception as e:
        print(f"Error durante el proceso: {e}")

if __name__ == "__main__":
    main() 