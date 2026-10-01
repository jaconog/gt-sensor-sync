import argparse

import json
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any



from src.io.parquet_reader import read_parquet_by_path
from src.io.io_utils import LocalDataLoader
from src.projection.project_pointcloud_auto import transform_slam_to_camera, project_points_to_image

from src.export.exporter_3dgs import GaussianSplattingExporter
# --- CONFIGURACIÓN DE AJUSTE ---
OFFSET_Z = 0.0
MAX_PROJECT_DISTANCE = 40.0

class CalibrationManager:
    def __init__(self, descriptor_path: str):
        self.calibrations = self._load_calibrations(descriptor_path)

    def _load_calibrations(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            data = json.load(f)
        
        calib_dict = {}
        sensors_data = data.get("sensors", {})
        
        for family in ["AD", "SVS", "GTLDR"]:
            for sensor in sensors_data.get(family, []):
                sensor_name = f"{family}_{sensor['position']}" 
                if "attributes" not in sensor or "Calibration" not in sensor["attributes"]: continue
                
                calib_node = sensor["attributes"]["Calibration"]
                extrinsics = calib_node.get("Extrinsics", {})
                if not extrinsics or "rot_00" not in extrinsics: continue
                
                R = np.array([
                    [extrinsics["rot_00"], extrinsics["rot_01"], extrinsics["rot_02"]],
                    [extrinsics["rot_10"], extrinsics["rot_11"], extrinsics["rot_12"]],
                    [extrinsics["rot_20"], extrinsics["rot_21"], extrinsics["rot_22"]]
                ])
                T = np.array([extrinsics["trans_0_m"], extrinsics["trans_1_m"], extrinsics["trans_2_m"]])
                
                intrinsics = calib_node.get("Intrinsics", {})

                # REPARACIÓN: Volver a construir la matriz K (3x3) requerida por el exportador 3DGS
                K = np.array([
                    [intrinsics.get("fx", 0), 0, intrinsics.get("cx", 0)],
                    [0, intrinsics.get("fy", 0), intrinsics.get("cy", 0)],
                    [0, 0, 1]
                ], dtype=np.float64) if intrinsics else None
                
                # REPARACIÓN CRÍTICA DEL MODELO ÓPTICO
                c1 = float(intrinsics.get("dist_c1", 0.0))
                fish_k1 = float(intrinsics.get("fisheye_k1", 0.0))
                
                if abs(c1) > 1e-6:
                    cam_type = "poly4"
                    is_fisheye = False
                elif abs(fish_k1) > 1e-6:
                    cam_type = "fisheye_opencv_px"
                    is_fisheye = True
                else:
                    cam_type = "perspective_opencv_px"
                    is_fisheye = False
                
                calib_dict[sensor_name] = {
                    "K": K,  # <--- MATRIZ REINTEGRADA AQUÍ
                    "R": R, "T": T,
                    "type": cam_type,
                    "is_fisheye": is_fisheye,
                    "fx": intrinsics.get("fx", 0), "fy": intrinsics.get("fy", 0),
                    "cx": intrinsics.get("cx", 0), "cy": intrinsics.get("cy", 0),
                    "k1": intrinsics.get("pinhole_k1", intrinsics.get("fisheye_k1", 0)),
                    "k2": intrinsics.get("pinhole_k2", intrinsics.get("fisheye_k2", 0)),
                    "k3": intrinsics.get("pinhole_k3", intrinsics.get("fisheye_k3", 0)),
                    "k4": intrinsics.get("pinhole_k4", intrinsics.get("fisheye_k4", 0)),
                    "k5": intrinsics.get("pinhole_k5", 0), "k6": intrinsics.get("pinhole_k6", 0),
                    "p1": intrinsics.get("pinhole_p1", 0), "p2": intrinsics.get("pinhole_p2", 0),
                    "c1": c1, "c2": intrinsics.get("dist_c2", 0),
                    "c3": intrinsics.get("dist_c3", 0), "c4": intrinsics.get("dist_c4", 0),
                    "aspect_ratio": intrinsics.get("pixel_aspect_ratio_x_by_y", 1.0),
                    "alpha": intrinsics.get("alpha", 0.0)
                }
        return calib_dict

class SensorSyncPipeline:
    def __init__(self, calib_manager: CalibrationManager):
        self.calib_manager = calib_manager
        
    def sync_events(self, df: pd.DataFrame, cam_threshold_s: float = 0.02, lidar_threshold_s: float = 0.06):
        """ Sincronizador Time-Sync con Cámara Ancla y Nearest Neighbor. """
        df = df.dropna(subset=["sync_timestamp"]).sort_values("sync_timestamp")
        
        cameras = df[df["sensor_type"] == "camera"].copy()
        lidars = df[df["sensor_type"] == "lidar"].copy()
        if cameras.empty: return []
            
        start_times = cameras.groupby("sensor_name")["sync_timestamp"].min()
        ref_cam_name = start_times.idxmax()
        ref_cam_events = cameras[cameras["sensor_name"] == ref_cam_name]
        
        matches = []
        for _, ref_row in ref_cam_events.iterrows():
            ref_ts = ref_row["sync_timestamp"]
            
            cam_diffs = (cameras["sync_timestamp"] - ref_ts).abs()
            valid_cams = cameras[cam_diffs <= cam_threshold_s]
            if valid_cams.empty: continue
            
            closest_cam_indices = valid_cams.groupby("sensor_name")["sync_timestamp"].apply(lambda x: (x - ref_ts).abs().idxmin())
            matched_cams = valid_cams.loc[closest_cam_indices]
            
            lidar_diffs = (lidars["sync_timestamp"] - ref_ts).abs()
            valid_lidars = lidars[lidar_diffs <= lidar_threshold_s]
            matched_lidars = pd.DataFrame()
            
            if not valid_lidars.empty:
                closest_lidar_indices = valid_lidars.groupby("sensor_name")["sync_timestamp"].apply(lambda x: (x - ref_ts).abs().idxmin())
                matched_lidars = valid_lidars.loc[closest_lidar_indices]
            
            matches.append({
                "timestamp": ref_ts,
                "cameras": matched_cams,
                "lidars": matched_lidars,
                "odom_reference": ref_row 
            })
        return matches

def process_frames_locally(matches, calib_manager: CalibrationManager, local_base_path: str, target_lidar: str = None):
    data_loader = LocalDataLoader(local_base_path=local_base_path)
    out_dir = Path(local_base_path) / "validation_projections"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for match in matches:
        timestamp = match["timestamp"]
        odom_row = match["odom_reference"]
        
        if pd.isna(odom_row['Translation X']) or pd.isna(odom_row['Yaw (rotation around Z)']):
            continue
            
        odom_translation = np.array([odom_row['Translation X'], odom_row['Translation Y'], odom_row['Translation Z']])
        
        # REPARACIÓN: Secuencia ZYX de automoción (Yaw, Pitch, Roll)
        odom_euler_zyx = np.array([
            odom_row['Yaw (rotation around Z)'],
            odom_row['Pitch (rotation around Y)'],
            odom_row['Roll (rotation around X)']
        ])

        print(f"\n--- Frame Sincronizado (ts: {timestamp:.3f}) ---")
        
        for _, cam_row in match["cameras"].iterrows():
            cam_name = cam_row["sensor_name"]
            cam_calib = calib_manager.calibrations.get(cam_name)
            if cam_calib is None: continue
                
            local_img_path = data_loader.resolve_local_path(cam_row["absolute_uri"])
            img_raw = cv2.imread(str(local_img_path))
            if img_raw is None: continue
            h, w = img_raw.shape[:2]
            
            # Dinamismo de radio para paliar la escasez del teleobjetivo AD_FC
            point_radius = 4 if cam_name == "AD_FC" else 2

            for _, lidar_row in match["lidars"].iterrows():
                lidar_name = lidar_row["sensor_name"]
                if target_lidar is not None and lidar_name != target_lidar: continue
                
                pc_slam = data_loader.load_pointcloud(lidar_row["absolute_uri"])
                if pc_slam.shape[0] == 0: continue

                points_cam = transform_slam_to_camera(
                    pc_slam, odom_translation, odom_euler_zyx, 
                    cam_calib["R"], cam_calib["T"], offset_z=OFFSET_Z
                )
                if len(points_cam) == 0: continue

                pixels_2d = project_points_to_image(points_cam, cam_calib)
                
                valid_mask = (pixels_2d[:, 0] >= 0) & (pixels_2d[:, 0] < w) & \
                             (pixels_2d[:, 1] >= 0) & (pixels_2d[:, 1] < h)
                
                valid_pixels = pixels_2d[valid_mask]
                valid_depths = points_cam[valid_mask, 2]
                
                if len(valid_pixels) > 0:
                    img_copy = img_raw.copy()
                    norm_depths = np.clip(valid_depths / MAX_PROJECT_DISTANCE, 0, 1)
                    values_array = np.uint8(255 * (1 - norm_depths))
                    colors = cv2.applyColorMap(values_array.reshape(-1, 1), cv2.COLORMAP_JET)

                    for i in range(len(valid_pixels)):
                        u, v = int(valid_pixels[i, 0]), int(valid_pixels[i, 1])
                        cv2.circle(img_copy, (u, v), point_radius, colors[i, 0].tolist(), -1)

                    save_path = out_dir / f"{cam_name}_{lidar_name}_{timestamp:.3f}.png"
                    img_out = cv2.resize(img_copy, (w // 2, h // 2)) if w >= 2880 else img_copy
                    cv2.imwrite(str(save_path), img_out)
                    print(f"    [OK] {cam_name} + {lidar_name} -> {len(valid_pixels)} ptos. Guardado.")

def export_frames_for_3dgs(matches, calib_manager: CalibrationManager, local_base_path: str, target_camera: str = None, global_pcd_path: str = None):
    data_loader = LocalDataLoader(local_base_path=local_base_path)
    
    # --- REPARACIÓN: Cambiar output_base_dir por output_dir ---
    exporter = GaussianSplattingExporter(output_dir=f"{local_base_path}/dataset_3dgs")
    # -----------------------------------------------------------
    
    for match in matches:
        timestamp = match["timestamp"]
        odom_row = match["odom_reference"]
        
        if pd.isna(odom_row['Translation X']) or pd.isna(odom_row['Yaw (rotation around Z)']):
            continue
            
        odom_translation = np.array([odom_row['Translation X'], odom_row['Translation Y'], odom_row['Translation Z']])
        odom_euler_zyx = np.array([
            odom_row['Yaw (rotation around Z)'],
            odom_row['Pitch (rotation around Y)'],
            odom_row['Roll (rotation around X)']
        ])

        for _, cam_row in match["cameras"].iterrows():
            cam_name = cam_row["sensor_name"]
            
            if target_camera and cam_name != target_camera:
                continue
                
            cam_calib = calib_manager.calibrations.get(cam_name)
            if cam_calib is None: continue
                
            local_img_path = data_loader.resolve_local_path(cam_row["absolute_uri"])
            
            exporter.add_frame(
                cam_name=cam_name,
                img_path=str(local_img_path),
                timestamp=timestamp,
                calib=cam_calib,
                odom_trans=odom_translation,
                odom_euler_zyx=odom_euler_zyx,
                offset_z=OFFSET_Z,
                global_pcd_path=global_pcd_path 
            )
            print(f"    [3DGS] Procesado frame {timestamp:.3f} para {cam_name}")
            
    exporter.finalize_export()



if __name__ == "__main__":
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Valeo GT Sensor Sync & 3DGS Exporter")
        parser.add_argument('--mode', type=str, choices=['validation', 'export_3dgs'], default='validation')
        parser.add_argument('--cam', type=str, default=None)
        parser.add_argument('--lidar', type=str, default=None)
        parser.add_argument('--global_pcd', type=str, default=None, help="Ruta de la nube de puntos acumulada en formato .pcd")
        
        args = parser.parse_args()

        descriptor_file = "data/raw/LBVS730_20251209_153903_recording_descriptor.json"
        parquet_path = "data/raw/trace_master_20251209_153903.parquet" 
        
        calib_mgr = CalibrationManager(descriptor_file)
        pipeline = SensorSyncPipeline(calib_mgr)

        df = read_parquet_by_path(parquet_path)
        synchronized_frames = pipeline.sync_events(df)
        
        if args.mode == 'validation':
            print("\n=== INICIANDO PIPELINE DE VALIDACIÓN GT ===")
            process_frames_locally(
                synchronized_frames, 
                calib_mgr, 
                local_base_path="D:/VALEO/tmp",
                target_lidar=args.lidar
            )

            # Ejemplo de ejecución del script
            # python main.py --mode validation --cam AD_FT --lidar GTLD_TC

        elif args.mode == 'export_3dgs':
            print("\n=== INICIANDO EXPORTACIÓN 3D GAUSSIAN SPLATTING ===")
            
            # --- VALIDACIÓN CRÍTICA DE LA NUBE DE PUNTOS ---
            if not args.global_pcd:
                raise ValueError("[ERROR FATAL] Debes proporcionar la ruta a la nube de puntos con --global_pcd para el modo export_3dgs.")
            
            pcd_path = Path(args.global_pcd)
            if not pcd_path.exists():
                raise FileNotFoundError(f"[ERROR FATAL] La nube de puntos no existe en la ruta: {pcd_path}")
            # -----------------------------------------------

            export_frames_for_3dgs(
                synchronized_frames, 
                calib_mgr, 
                local_base_path="D:/VALEO/tmp",
                target_camera=args.cam,
                global_pcd_path=str(pcd_path)
            )

            # Ejemplo de ejecución del script

            # python main.py --mode export_3dgs --cam AD_FT --global_pcd "D:/VALEO/tmp/mapa_acumulado_slam.pcd"

            # Entrenamiento en tu entorno de investigación
            # ns-train splatfacto --data D:/VALEO/tmp/dataset_3dgs/AD_FT

            # python main.py --mode export_3dgs --cam AD_FT --global_pcd "D:\VALEO\tmp\VALEO\data\gt-mapper-data\Answer_03\20251209_153903\trace_20251209_153903_GTLDR_TC.pcd"

