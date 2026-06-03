===================================================================================
           WWM WORLD WILDLIFE METRICS - DETECTOR & CLASIFICADOR - GUÍA DE USUARIO
===================================================================================

¡Bienvenido al Detector & Clasificador de Especies WWM!
Esta herramienta utiliza Inteligencia Artificial avanzada para analizar
imágenes y videos, detectando automáticamente la presencia de animales 
y clasificándolos por su especie.

--------------------------------------------------------------------
1. ¿CÓMO INICIAR LA APLICACIÓN?
--------------------------------------------------------------------
Para abrir el programa, simplemente haz doble clic en el archivo 
ejecutable (.exe).

⚠️ IMPORTANTE: TIEMPOS DE ESPERA
- Al abrir el programa: Notarás que el programa tarda entre 10 y 20 
  segundos o más en mostrar la interfaz, y durante este tiempo verás una 
  imagen de carga. Esto es completamente NORMAL. Se debe a que el 
  ejecutable contiene motores de IA muy pesados comprimidos en su 
  interior, y Windows necesita extraerlos temporalmente en segundo plano.
  
- La primera vez que lo uses: Una vez abierta la interfaz, la aplicación 
  necesitará descargar los "pesos" o "cerebros" de los modelos de 
  Inteligencia Artificial (aproximadamente 270 MB). Este proceso solo 
  ocurrirá la primera vez. Verás un texto que dice "Cargando modelos..." 
  en la esquina superior derecha. Por favor, espera a que cambie a 
  "Listo" antes de iniciar un análisis. Si el internet falla, el 
  programa intentará reanudar la descarga automáticamente.

--------------------------------------------------------------------
2. CARPETAS GENERADAS AUTOMÁTICAMENTE
--------------------------------------------------------------------
Para funcionar correctamente, el programa creará automáticamente las 
siguientes carpetas junto al archivo .exe:

- carpeta "models/": Aquí es donde se guardan los modelos de Inteligencia 
  Artificial descargados. NO borres esta carpeta, o el programa tendrá 
  que volver a descargar los 270 MB la próxima vez que lo abras.

- carpeta "temp/" (opcional según uso interno): Utilizada por el 
  programa para procesar archivos de video y realizar recortes de 
  imágenes antes de la clasificación.

--------------------------------------------------------------------
3. ¿CÓMO USAR EL CLASIFICADOR?
--------------------------------------------------------------------
1. En "Carpeta de entrada", selecciona la carpeta de tu computadora 
   donde tengas guardadas todas las imágenes y/o videos de cámaras trampa.
2. En "Carpeta de salida", selecciona una carpeta vacía donde quieras 
   que se guarden los resultados.
3. Elige la acción deseada:
   - "Copiar archivos": Hará copias de tus fotos originales a la salida.
   - "Mover archivos": Quitará las fotos de la entrada y las pasará a la salida.
4. Presiona "▶ Iniciar Análisis".

El programa procesará cada archivo y creará subcarpetas automáticas en
tu carpeta de salida con el nombre de cada especie detectada (por ejemplo, 
"jaguar", "puma", etc.), ordenando tus archivos de forma inteligente.

--------------------------------------------------------------------
4. ACERCA DE NOSOTROS
--------------------------------------------------------------------
Esta herramienta ha sido desarrollada orgullosamente por el equipo:
*** CETERIS 404 *** 
Participantes del evento InnovaHack 2026.

Si tienes problemas técnicos, encuentras algún error o necesitas ayuda 
con la aplicación, no dudes en ponerte en contacto con nuestro 
desarrollador principal:

📧 Correo de contacto: diegoalberto42216@gmail.com

¡Gracias por usar WWM World Wildlife Metrics - Detector & Clasificador!
