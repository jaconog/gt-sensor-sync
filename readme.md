## EXPLICACION DE LA PIPELINE

Así es como fluye la información paso a paso por cada fotograma:
1. El calib_manager (El Descriptor): Al inicio del programa, este objeto parsea todo el recording_descriptor.json. Almacena en la memoria RAM las matrices intrínsecas ($K$, distorsión) y extrínsecas ($R, T$) de todas las cámaras y LiDARs.   
2. Los matches (El Parquet + Odometría): La función sync_events lee el Parquet y agrupa los datos. Cada match es un diccionario que contiene la clave odom_reference. Ahí viaja la odometría exacta del coche (Translation X/Y/Z y Yaw/Pitch/Roll) en ese preciso instante.   
3. La Inyección (main.py): Cuando el bucle for itera sobre los matches, extrae la odometría (odom_translation, odom_euler_zyx) y le pide al calib_manager la calibración de la cámara (cam_calib).   
4. El Consumo (exporter.add_frame): El main.py le pasa todos estos paquetes ya procesados al exportador mediante la función exporter.add_frame(...). Es decir, el exportador recibe los datos "masticados" y listos para operar.   

# 2. El Proceso Paso a Paso del Exportador 3DGS
Cuando llamas al modo --mode export_3dgs, la clase GaussianSplattingExporter realiza las siguientes operaciones secuenciales por debajo:   
- Paso A: Inicialización y Conversión del PLY
La primera vez que procesa una cámara (ej. AD_FT), crea una carpeta para ella. Como le has pasado la ruta del .pcd global acumulado del SLAM, el exportador usa la librería open3d para leer ese .pcd y lo reescribe como sparse_pc.ply dentro de la nueva carpeta.   
- Paso B: Rectificación de la Imagen (Undistortion)
Los motores como Nerfstudio asumen un mundo ideal donde las cámaras son Pinhole perfectas (líneas rectas). Tus cámaras reales (SVS y AD) usan lentes curvadas (poly4 o fisheye). El método _rectify_image_and_intrinsics lee la imagen RAW, aplica la matemática inversa de la distorsión, recorta los bordes negros generados y calcula una nueva matriz intrínseca K ($f_x, f_y, c_x, c_y$ recalculados para la imagen plana).   
- Paso C: Cálculo de la Pose Global (Camera-to-World)
Nerfstudio no sabe que existe un coche; solo entiende de "Cámaras" flotando en el "Mundo". El método _compute_c2w_matrix hace la transformación:   
    Toma la Odometría (Vehículo $\rightarrow$ Mundo SLAM).   
    Toma el Extrínseco de Valeo (Cámara $\rightarrow$ Vehículo).   
    Las multiplica matricialmente para generar una única matriz 4x4 llamada $M_{c2w}$ (Camera-to-World).   
- Paso D: Guardado Iterativo. Guarda la imagen plana en disco y guarda los parámetros en un diccionario en la memoria RAM. Al terminar todos los frames, llama a finalize_export() y escribe el JSON definitivo.





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



# Google Colab (El Compute Engine):
En Colab no vas a ejecutar tu código de orquestación. Colab se usará exclusivamente como una granja de GPUs.

Subes la carpeta dataset_3dgs generada localmente a Google Drive (o la descargas directamente desde GCP en el notebook).

Abres un Notebook en Colab, montas tu Google Drive.

Instalas Nerfstudio en el entorno de Colab.

Ejecutas el comando de entrenamiento apuntando a tus datos: ns-train splatfacto --data /content/drive/MyDrive/dataset_3dgs/AD_FT.
