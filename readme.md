## Stack Tecnologico
Para un entorno profesional como Valeo, basado en Python y con integracion en la nube, el stack debe priorizar el rendimiento de lectura de datos tabulares y el procesamiento geometrico:
1. Lectura de Datos y Tablas: pandas junto con pyarrow o fastparquet para leer los archivos generados y estructurar los eventos temporalmente de forma vectorizada.
2. Integracion Cloud: gcsfs y google-cloud-storage. Dado que los absolute_uri apuntan a gs://, gcsfs permite a pandas y otras librerías leer directamente de la nube sin tener que programar descargas manuales intermedias.
3. Geometria y Vision por Computador: scipy (especificamente scipy.spatial.transform.Rotation para manejar las matrices de rotacion ISO), numpy para las operaciones matriciales de proyeccion camara-lidar, y opencv-python (cv2) para la des-distorsion de imagenes usando los coeficientes del JSON.
4. Nubes de Puntos: open3d para leer, procesar y visualizar los archivos .ply. Es un estandar industrial eficiente en C++ con bindings en Python.
5. Entorno: Docker para asegurar que las dependencias de C++ (OpenCV, Open3D) se ejecuten consistentemente en cualquier servidor de Valeo.

## Flujo de Trabajo del Pipeline
1. Ingesta de Metadatos: El script lee el archivo LBVS730_20251209_151525_recording_descriptor.json y carga en memoria diccionarios con las matrices intrinsecas y extrinsecas de cada sensor indexadas por su nombre (ej. AD_FC, AD_FR).   
2. Lectura Temporal: Se lee el .parquet completo. Se ordena el dataframe por sync_timestamp (o odom_timestamp dependiendo de la estrategia de ego-motion).
3. Sincronizacion: Se define una ventana de tolerancia temporal (ej. $\pm 10$ ms) para agrupar en un solo "Frame" logico el escaneo del LiDAR (GTLDR_TC, GTLDR_RS) y las imagenes de las camaras (AD_FT, AD_FC, SVS_FV, etc.) correspondientes a ese instante.
4. Extraccion I/O (Lazy Loading): En lugar de descargar todo el bucket de GCP, el script itera sobre los "Frames" sincronizados y utiliza las URIs gs:// para descargar en memoria o en el directorio data/processed/ unicamente los archivos .ply y .png emparejados.
5. Proyeccion / Fusion: Se utilizan las matrices leidas en el paso 1 junto con los deltas de odometria del parquet para proyectar los puntos del LiDAR sobre el plano de la imagen de la camara correspondiente, o para transformar todos los sensores al sistema de coordenadas de referencia trasero del vehiculo.


### TO DO
1. Gestión de Datos (Completado): Parser de parquet funcional y agrupación temporal basada en sync_timestamp iterando fotogramas correctamente.   
2. Gestión de Calibración (Completado): Lectura estructurada de JSON para recuperar intrínsecas (incluyendo factores de distorsión) y matrices extrínsecas $[R\vert{}T]$.   
3. Ingeniería Geométrica (90% Completado): Pipeline configurado para proyectar LiDAR crudo sobre imágenes crudas. 
 -> Falta inyectar la carga de archivos .pcd o .bin reales. -> vamos a cargar los scans extraidos ya.

Lo que falta para lograr 3DGS:

1. Formateo de Datos (Exportación COLMAP-Style): 3DGS requiere conocer la pose de cada cámara en un sistema de coordenadas global. Debes crear un script que convierta las transformaciones dinámicas del vehículo (odométrica o GPS/IMU) combinadas con las extrínsecas de cada cámara en un archivo transforms.json o un modelo COLMAP estático.

2. Colorización de la Nube de Puntos (Opcional pero recomendado): Puedes usar el módulo de proyección que acabas de construir para asignar valores RGB a cada punto LiDAR válido proyectado en la cámara. Esto generará un .ply inicial altamente preciso que acelerará la convergencia de los gaussianos en el entrenamiento.

3. Filtrado de Oclusiones (Hidden Point Removal): Si proyectas todo el LiDAR sobre una cámara, los puntos 3D que están detrás de objetos opacos (ej. un peatón tapando un coche) se proyectarán incorrectamente. Necesitarás implementar un z-buffer o filtrado esférico simple para evitar proyectar puntos ocluidos. -> Frustum Culling

4. Integración con el motor 3DGS: Una vez tengas el dataset estructurado (Imágenes RAW + poses en json/colmap + ply inicial), alimentarás el repositorio estándar de 3DGS para comenzar el paso de optimización train.py.