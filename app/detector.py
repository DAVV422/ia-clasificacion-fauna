from asyncio import proactor_events
import logging
import uuid
import cv2
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from PIL import Image
from app.config import (
    MEGADETECTOR_PATH,
    CONFIDENCE_THRESHOLD,
    DETECTIONS_DIR,
    DETECTIONS_URL_PREFIX,
    MEGADETECTOR_CLASSES,
    SPECIESNET_MODEL_NAME,
    SPECIESNET_CONFIDENCE_THRESHOLD
)


logger = logging.getLogger("uvicorn.error")

class WildlifeDetector:
    def __init__(self):
        self.model = None
        self.classifier = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device} for inference.")

    def load_model(self):
        """
        Loads the MegaDetector YOLOv5 model using PyTorch Hub and the SpeciesNet classifier.
        It uses the locally downloaded weight file specified in MEGADETECTOR_PATH.
        """
        if self.model is None:
            model_path_str = str(MEGADETECTOR_PATH.resolve())
            logger.info(f"Loading MegaDetector weights from {model_path_str}...")
            
            try:
                # Load custom YOLOv5 model using PyTorch Hub
                self.model = torch.hub.load(
                    "ultralytics/yolov5",
                    "custom",
                    path=model_path_str,
                    device=self.device,
                    force_reload=False,
                    trust_repo=True
                )
                # Set confidence threshold on the model itself
                self.model.conf = CONFIDENCE_THRESHOLD
                logger.info("MegaDetector model loaded successfully!")
            except Exception as e:
                logger.error(f"Error loading model via PyTorch Hub: {e}")
                logger.info("Retrying with force_reload=True...")
                try:
                    self.model = torch.hub.load(
                        "ultralytics/yolov5",
                        "custom",
                        path=model_path_str,
                        device=self.device,
                        force_reload=True,
                        trust_repo=True
                    )
                    self.model.conf = CONFIDENCE_THRESHOLD
                    logger.info("MegaDetector loaded successfully after reload!")
                except Exception as retry_err:
                    logger.error(f"Failed to load model on retry: {retry_err}")
                    raise retry_err

        # Load SpeciesNet classifier
        if self.classifier is None:
            try:
                from speciesnet.classifier import SpeciesNetClassifier
                logger.info(f"Loading SpeciesNet classifier '{SPECIESNET_MODEL_NAME}'...")
                self.classifier = SpeciesNetClassifier(model_name=SPECIESNET_MODEL_NAME, device=self.device)
                logger.info("SpeciesNet classifier loaded successfully!")
            except Exception as e:
                logger.error(f"Error loading SpeciesNet classifier: {e}")

    def map_class_name(self, class_id: int, class_name_str: str) -> str:
        """
        Maps YOLOv5 output class index or name to official MegaDetector labels (animal, person, vehicle).
        """
        # Try index-based lookup
        class_str = str(class_id)
        if class_str in MEGADETECTOR_CLASSES:
            return MEGADETECTOR_CLASSES[class_str]
        
        # Try name-based mapping or lowercase lookup
        clean_name = class_name_str.lower().strip()
        if clean_name in MEGADETECTOR_CLASSES:
            return MEGADETECTOR_CLASSES[clean_name]
        
        # Map indices directly (YOLOv5 default mapping for MegaDetector v5)
        # Class 0: animal, Class 1: person, Class 2: vehicle
        default_map = {0: "animal", 1: "person", 2: "vehicle"}
        return default_map.get(class_id, clean_name)

    def detect_and_crop(self, frame_bgr: cv2.Mat, video_name: str, timestamp: float = 0.0) -> List[Dict[str, Any]]:
        """
        Runs MegaDetector v5a on a single frame, crops detected objects,
        saves the cropped images locally in folders categorized by video instance and category,
        runs SpeciesNet on animals, and returns detection details.
        
        :param frame_bgr: El frame actual en formato BGR.
        :param video_name: Nombre o identificador único de la ejecución del video (ej: "camara1_20260530_112233").
        :param timestamp: Tiempo en segundos dentro del video.
        """
        if self.model is None:
            self.load_model()

        # OpenCV reads in BGR, YOLOv5 expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = frame_bgr.shape

        # Run inference
        results = self.model(frame_rgb)
        
        # Parse results as pandas DataFrame
        # xyxy[0] columns: xmin, ymin, xmax, ymax, confidence, class, name
        detections_df = results.pandas().xyxy[0]

        detections = []

        for _, row in detections_df.iterrows():
        #     conf = float(row["confidence"])
        #     if conf < CONFIDENCE_THRESHOLD:
        #         continue

        #     class_id = int(row["class"])
        #     raw_class_name = str(row["name"])
        #     category = self.map_class_name(class_id, raw_class_name)

        #     # Get bounding boxes in absolute integer pixels
        #     x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            
        #     # Constrain to frame boundaries
        #     x1 = max(0, x1)
        #     y1 = max(0, y1)
        #     x2 = min(width, x2)
        #     y2 = min(height, y2)

        #     # Verify crop coordinates are valid
        #     if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        #         continue

        #     # Normalized coordinates for Pydantic response schema [ymin, xmin, ymax, xmax]
        #     box_normalized = [
        #         round(y1 / height, 4),
        #         round(x1 / width, 4),
        #         round(y2 / height, 4),
        #         round(x2 / width, 4)
        #     ]

        #     # Crop the object
        #     crop_img = frame_bgr[y1:y2, x1:x2]

        #     image_uuid = str(uuid.uuid4())
        #     # 📂 Estructura física en disco organizada
        #     category_dir = DETECTIONS_DIR / video_name / category
        #     category_dir.mkdir(parents=True, exist_ok=True)
            
        #     crop_filename = f"{image_uuid}.jpg"
        #     crop_filepath = category_dir / crop_filename

        #     # Save cropped image locally
        #     success = cv2.imwrite(str(crop_filepath), crop_img)
            
        #     if success:
        #         # 🛠️ CORRECCIÓN: Ajustamos las rutas relativas y URLs para reflejar el orden correcto
        #         relative_path = f"detections/{video_name}/{category}/{crop_filename}"
                
        #         # Usamos los strings limpios en la URL para evitar rutas del sistema (Windows/Linux)
        #         crop_url = f"{DETECTIONS_URL_PREFIX}/{video_name}/{category}/{crop_filename}"
        #     else:
        #         logger.error(f"Failed to save crop at {crop_filepath}")
        #         relative_path = None
        #         crop_url = None

        #     # Run SpeciesNet classification if category is animal
        #     species = None
        #     species_conf = None
        #     if category == "animal" and self.classifier is not None:
        #         try:
        #             # Convert crop to PIL Image for SpeciesNet
        #             crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        #             crop_pil = Image.fromarray(crop_rgb)
                    
        #             preprocessed = self.classifier.preprocess(crop_pil, bboxes=None)
        #             if preprocessed is not None:
        #                 result = self.classifier.predict(filepath=crop_filename, img=preprocessed)
        #                 logger.info(f"SpeciesNet raw result for {crop_filename}: {result}")

        #                 if isinstance(result, dict):
        #                     classifications = result.get("classifications") or {}
        #                     classes_list = classifications.get("classes", [])
        #                     scores_list = classifications.get("scores", [])
                            
        #                     # Creamos una lista para almacenar todas las especies encontradas con sus scores
        #                     all_detected_species = []
                            
        #                     if classes_list and scores_list:
        #                         # Recorremos ambas listas en paralelo usando zip
        #                         for raw_species, score in zip(classes_list, scores_list):
                                    
        #                             # Opcional: Si solo quieres el nombre común (ej: "domestic dog") en vez de toda la taxonomía,
        #                             # puedes usar: species_name = raw_species.split(";")[-1] if ";" in raw_species else raw_species
        #                             parts = raw_species.split(";")
        #                             if len(parts) >= 4:
        #                                 # Caso ideal: Tomamos los últimos 3 elementos y los volvemos a unir
        #                                 species_name = ";".join(parts[-3:])
        #                             elif len(parts) > 1:
        #                                 # Caso con pocos elementos: Quitamos el ID (el primero) y unimos el resto
        #                                 species_name = ";".join(parts[1:])
        #                             else:
        #                                 # Caso extremo: Si solo viene el ID o un texto plano sin ';', devolvemos ese único texto
        #                                 species_name = raw_species
                                    
        #                             rounded_score = round(float(score), 4)
                                    
        #                             # Guardamos cada detección en un diccionario estructurado
        #                             all_detected_species.append({
        #                                 "species": species_name,
        #                                 "confidence": rounded_score
        #                             })
                                    
        #                         logger.info(f"All species detected for {crop_filename}: {all_detected_species}")
                                
        #                         # --- NOTA PARA TU LOGICA POSTERIOR ---
        #                         # Si tu código original dependía de las variables individuales 'species' y 'species_conf',
        #                         # podemos asignarles por defecto el resultado con mayor puntaje (el primero) para que no falle el flujo:
        #                         species = all_detected_species[0]["species"]
        #                         species_conf = all_detected_species[0]["confidence"]
                                
        #                     else:
        #                         # Fallback por si la respuesta viene con el formato antiguo
        #                         species = result.get("classes")
        #                         species_conf = result.get("confidence") or result.get("probability") or result.get("score") or result.get("classes")
        #                         if species_conf is not None:
        #                             species_conf = round(float(species_conf), 4)
                                    
        #                         # Si entra aquí, simulamos la lista con el único resultado encontrado
        #                         if species:
        #                             all_detected_species.append({"species": species, "confidence": species_conf})
        #         except Exception as ex:
        #             logger.error(f"Failed running SpeciesNet inference on crop {crop_filename}: {ex}")

        #     detections.append({
        #         "box": box_normalized,
        #         "class_name": category,
        #         "confidence": round(conf, 4),
        #         "species": all_detected_species,                
        #         "crop_path": relative_path,
        #         "crop_url": crop_url
        #     })

        # return detections
            conf = float(row["confidence"])
            if conf < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(row["class"])
            raw_class_name = str(row["name"])
            category = self.map_class_name(class_id, raw_class_name)

            # Get bounding boxes in absolute integer pixels
            x1, y1, x2, y2 = int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"])
            
            # Constrain to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            # Verify crop coordinates are valid
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            # Normalized coordinates for Pydantic response schema [ymin, xmin, ymax, xmax]
            box_normalized = [
                round(y1 / height, 4),
                round(x1 / width, 4),
                round(y2 / height, 4),
                round(x2 / width, 4)
            ]

            # Crop the object
            crop_img = frame_bgr[y1:y2, x1:x2]
            
            image_uuid = str(uuid.uuid4())
            # 📂 Estructura física en disco organizada por instancia única de video
            category_dir = DETECTIONS_DIR / video_name / category
            category_dir.mkdir(parents=True, exist_ok=True)
            
            crop_filename = f"{image_uuid}.jpg"
            crop_filepath = category_dir / crop_filename

            # Save cropped image locally
            success = cv2.imwrite(str(crop_filepath), crop_img)
            
            if success:
                # 🛠️ Rutas relativas y URLs reflejando el orden e inyectando strings puros
                relative_path = f"detections/{video_name}/{category}/{crop_filename}"
                crop_url = f"{DETECTIONS_URL_PREFIX}/{video_name}/{category}/{crop_filename}"
            else:
                logger.error(f"Failed to save crop at {crop_filepath}")
                relative_path = None
                crop_url = None

            # Inicializamos siempre la lista para que el controlador no reciba un KeyError
            all_detected_species = []

            # Run SpeciesNet classification if category is animal
            if category == "animal" and self.classifier is not None:
                try:
                    # Convert crop to PIL Image for SpeciesNet
                    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    crop_pil = Image.fromarray(crop_rgb)
                    
                    preprocessed = self.classifier.preprocess(crop_pil, bboxes=None)
                    if preprocessed is not None:
                        result = self.classifier.predict(filepath=crop_filename, img=preprocessed)
                        logger.info(f"SpeciesNet raw result for {crop_filename}: {result}")

                        if isinstance(result, dict):
                            classifications = result.get("classifications") or {}
                            classes_list = classifications.get("classes", [])
                            scores_list = classifications.get("scores", [])
                            
                            if classes_list and scores_list:
                                # Recorremos ambas listas en paralelo usando zip
                                for raw_species, score in zip(classes_list, scores_list):
                                    parts = raw_species.split(";")
                                    if len(parts) >= 4:
                                        # Caso ideal: Tomamos los últimos 3 elementos (ej. panthera;onca;jaguar)
                                        species_name = ";".join(parts[-3:])
                                    elif len(parts) > 1:
                                        # Caso corto: Quitamos el ID (índice 0) y unimos el resto
                                        species_name = ";".join(parts[1:])
                                    else:
                                        species_name = raw_species
                                    
                                    rounded_score = round(float(score), 4)
                                    
                                    all_detected_species.append({
                                        "species": species_name,
                                        "confidence": rounded_score
                                    })
                                    
                                logger.info(f"All species detected for {crop_filename}: {all_detected_species}")
                                
                            else:
                                # Fallback por si la respuesta viene con formato plano de un solo elemento
                                species_fallback = result.get("classes")
                                score_fallback = result.get("confidence") or result.get("probability") or result.get("score")
                                if species_fallback:
                                    sf_conf = round(float(score_fallback), 4) if score_fallback else 1.0
                                    all_detected_species.append({"species": species_fallback, "confidence": sf_conf})
                except Exception as ex:
                    logger.error(f"Failed running SpeciesNet inference on crop {crop_filename}: {ex}")

            # Insertamos la estructura final limpia que espera procesar tu endpoint
            detections.append({
                "box": box_normalized,
                "class_name": category,
                "confidence": round(conf, 4),
                "species": all_detected_species, # Enviamos la lista armada de diccionarios               
                "crop_path": relative_path,
                "crop_url": crop_url
            })

        return detections