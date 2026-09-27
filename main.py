import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from src.io.parquet_reader import read_trace_parquet, read_parquet_by_path
from src.io.io_utils import LocalDataLoader

class CalibrationManager:
    def __init__(self, descriptor_path: str):
        self.calibrations = self._load_calibrations(descriptor_path)

    def _load_calibrations(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            data = json.load(f)
        
        calib_dict = {}
        sensors_data = data.get("sensors", {})
        
        # Iterar explícitamente sobre ambas familias en el JSON
        for family in ["AD", "SVS"]:
            for sensor in sensors_data.get(family, []):
                # Genera nombres como AD_FL, SVS_FV, etc.
                sensor_name = f"{family}_{sensor['position']}" 
                
                # Validar que los atributos existan
                if "attributes" not in sensor or "Calibration" not in sensor["attributes"]:
                    continue
                    
                intrinsics = sensor["attributes"]["Calibration"]["Intrinsics"]
                extrinsics = sensor["attributes"]["Calibration"]["Extrinsics"]
                
                # Construir matriz de rotación 3x3
                R = np.array([
                    [extrinsics["rot_00"], extrinsics["rot_01"], extrinsics["rot_02"]],
                    [extrinsics["rot_10"], extrinsics["rot_11"], extrinsics["rot_12"]],
                    [extrinsics["rot_20"], extrinsics["rot_21"], extrinsics["rot_22"]]
                ])
                
                # Construir vector de traslación
                T = np.array([extrinsics["trans_0_m"], extrinsics["trans_1_m"], extrinsics["trans_2_m"]])
                
                # Matriz intrínseca K
                K = np.array([
                    [intrinsics["fx"], 0, intrinsics["cx"]],
                    [0, intrinsics["fy"], intrinsics["cy"]],
                    [0, 0, 1]
                ])
                
                # Determinamos el modelo de lente asumiendo que "fisheye_k1" solo existe en lentes de ojo de pez
                is_fisheye = "fisheye_k1" in intrinsics
                
                calib_dict[sensor_name] = {
                    "K": K, 
                    "R": R, 
                    "T": T, 
                    "distortion": intrinsics,
                    "is_fisheye": is_fisheye
                }

                # print(f"Calibración cargada para {sensor_name} (Fisheye: {is_fisheye})")
                
        return calib_dict

class SensorSyncPipeline:
    def __init__(self, calib_manager: CalibrationManager):
        self.calib_manager = calib_manager
        
    def sync_events(self, df: pd.DataFrame, tolerance_s: float = 0.05):
        """Agrupa eventos de cámara y LiDAR basándose en la columna 'sync_timestamp'."""
        df = df.dropna(subset=["sync_timestamp"]).sort_values(by="sync_timestamp")
        df["time_group"] = (df["sync_timestamp"] / tolerance_s).round() * tolerance_s
        return df.groupby("time_group")


# Bucle de procesamiento extraído de io_utils.py a la ejecución principal
def process_frames_locally_undistort_image(frames, calib_manager: CalibrationManager, local_base_path: str):
    data_loader = LocalDataLoader(local_base_path=local_base_path)
    
    for timestamp, group in frames:
        print(f"\n--- Frame Sincronizado (ts: {timestamp:.3f}) ---")
        
        cameras = group[group["sensor_type"] == "camera"]
        lidars = group[group["sensor_type"] == "lidar"]
        
        # Procesar primero el LiDAR para tener la nube de puntos del frame (si existe)
        frame_pointclouds = []
        for _, lidar_row in lidars.iterrows():
            lidar_path = data_loader.resolve_local_path(lidar_row["absolute_uri"])
            # pc = load_pointcloud(lidar_path)
            # frame_pointclouds.append(pc)
            
        for _, cam_row in cameras.iterrows():
            cam_name = cam_row["sensor_name"]
            uri = cam_row["absolute_uri"]
            calib = calib_manager.calibrations.get(cam_name)
            
            if calib is None:
                print(f"    [CAM] {cam_name} -> Sin calibración en JSON. Saltando.")
                continue
                
            # Cargar y rectificar la imagen directamente desde el disco
            img = data_loader.load_and_undistort_image(uri, calib)
            
            if img is not None:
                print(f"    [CAM] {cam_name} -> Imagen rectificada (Fisheye: {calib['is_fisheye']})")
                
                # Siguiente fase geométrica:
                # project_lidar_to_camera(frame_pointclouds, calib["R"], calib["T"], calib["K"], img)
            else:
                print(f"    [WARNING] {cam_name} -> No se pudo procesar URI: {uri}")


def process_frames_locally(frames, calib_manager: CalibrationManager, local_base_path: str):
    data_loader = LocalDataLoader(local_base_path=local_base_path)
    
    for timestamp, group in frames:
        print(f"\n--- Frame Sincronizado (ts: {timestamp:.3f}) ---")
        
        cameras = group[group["sensor_type"] == "camera"]
        lidars = group[group["sensor_type"] == "lidar"]
        
        # 1. Cargar y acumular nubes de puntos LiDAR del frame
        frame_pointcloud = []
        for _, lidar_row in lidars.iterrows():
            lidar_path = data_loader.resolve_local_path(lidar_row["absolute_uri"])
            # Asumiendo que tienes una función para leer pcd/bin
            # pc = load_pointcloud_data(lidar_path)  
            # frame_pointcloud.append(pc)
            
        # Simulación temporal si no tienes carga de LiDAR aún:
        # frame_pointcloud = np.random.rand(10000, 3) * 50  # Dummy Nx3
        
        # if len(frame_pointcloud) == 0: continue
        # frame_pointcloud = np.vstack(frame_pointcloud)

        # 2. Procesar proyecciones de cámara
        for _, cam_row in cameras.iterrows():
            cam_name = cam_row["sensor_name"]
            uri = cam_row["absolute_uri"]
            calib = calib_manager.calibrations.get(cam_name)
            
            if calib is None:
                continue
                
            # Cargar imagen RAW (sin rectificar)
            local_img_path = data_loader.resolve_local_path(uri)
            img_raw = cv2.imread(str(local_img_path))
            
            if img_raw is not None:
                # A. Transformar de LiDAR a Cámara
                # points_cam = transform_lidar_to_camera(frame_pointcloud, calib["R"], calib["T"])
                
                # B. Proyectar a píxeles
                # pixels_2d = project_points_to_image(points_cam, calib)
                
                # C. (Opcional) Filtrar píxeles fuera del tamaño de la imagen
                # h, w = img_raw.shape[:2]
                # valid_pixels_mask = (pixels_2d[:, 0] >= 0) & (pixels_2d[:, 0] < w) & \
                #                     (pixels_2d[:, 1] >= 0) & (pixels_2d[:, 1] < h)
                # valid_pixels = pixels_2d[valid_pixels_mask]
                
                print(f"    [CAM] {cam_name} -> Imagen RAW cargada. Lista para proyeccion.")

if __name__ == "__main__":
    # 1. Inicialización del entorno y calibraciones
    descriptor_file = "data/raw/LBVS730_20251209_153903_recording_descriptor.json"
    calib_mgr = CalibrationManager(descriptor_file)
    pipeline = SensorSyncPipeline(calib_mgr)

    # 2. Cargar eventos desde el Parquet
    parquet_path = "data/raw/trace_master_20251209_153903.parquet" 
    df = read_parquet_by_path(parquet_path)

    # 3. Sincronizar fotogramas
    synchronized_frames = pipeline.sync_events(df)
    
    # 4. Procesar y visualizar/proyectar
    # Ajusta la ruta base D:/VALEO/tmp según donde tengas montados tus datos
    process_frames_locally(synchronized_frames, calib_mgr, local_base_path="D:/VALEO/tmp")