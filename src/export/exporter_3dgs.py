import os
import cv2
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation

"""
1 - Rectifica las imágenes (Undistortion): Elimina la distorsión de la lente usando cv2.fisheye.initUndistortRectifyMap o cv2.initUndistortRectifyMap, 
creando imágenes planas y calculando los nuevos parámetros intrínsecos ($f_x, f_y, c_x, c_y$).   

2 - Calcula la Pose Global ($M_{c2w}$): Multiplica la posición del coche en el mundo (odometría) por la posición de la cámara en el coche (extrínsecos) 
para obtener una única matriz 4x4 de "Cámara a Mundo".   

3 - Genera el transforms.json: Empaqueta toda esta información en el formato exacto que espera Nerfstudio.  

"""

class GaussianSplattingExporter:
    def __init__(self, output_dir: str, voxel_size: float = 0.1, bbox_min=None, bbox_max=None):
        self.output_dir = Path(output_dir)
        self.img_dir = self.output_dir / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames = []
        self.global_ply_path = ""
        self._pcd_converted = False
        
        # Atributos de protección de memoria RAM/VRAM
        self.voxel_size = voxel_size
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

    def convert_pcd_to_ply(self, pcd_path: str):
        """Convierte, recorta espacialmente y filtra la densidad del PCD."""
        if self._pcd_converted:
            return
            
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            if len(pcd.points) == 0:
                print(f"[ERROR CRÍTICO] La nube de puntos {pcd_path} tiene 0 puntos.")
                return
            
            print(f"\n[3DGS EXPORT] Nube original procesada: {len(pcd.points):,} puntos.")
            
            # 1. RECORTAR ESPACIALMENTE (Bounding Box dinámico de la trayectoria)
            if self.bbox_min is not None and self.bbox_max is not None:
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=self.bbox_min, max_bound=self.bbox_max)
                pcd = pcd.crop(bbox)
                print(f"[3DGS EXPORT] Puntos tras recorte de trayectoria (+40m margen): {len(pcd.points):,}")
            
            # 2. ALIGERAR DENSIDAD (Voxel Downsample para salvar VRAM)
            if self.voxel_size > 0.0:
                pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
                print(f"[3DGS EXPORT] Puntos tras Voxel Downsample ({self.voxel_size}m): {len(pcd.points):,}")
            
            sparse_dir = self.output_dir / "sparse" / "0"
            sparse_dir.mkdir(parents=True, exist_ok=True)
            ply_filename = "sparse/0/points3D.ply"
            ply_path = self.output_dir / ply_filename
            
            o3d.io.write_point_cloud(str(ply_path), pcd, write_ascii=False)
            
            self.global_ply_path = ply_filename
            self._pcd_converted = True
            print(f"[3DGS EXPORT] Nube lista para Nerfstudio guardada en: {ply_path}\n")
            
        except Exception as e:
            print(f"[ERROR FATAL] Falla estructural al convertir PCD a PLY: {e}")

            
    def _process_image_and_intrinsics(self, img_raw: np.ndarray, calib: dict) -> tuple:
        h, w = img_raw.shape[:2]
        K = calib["K"]
        model_type = calib.get("type", "perspective_opencv_px")

        # Mantenemos la lógica intacta: RAW para SVS/Fisheye, Rectificadas para AD/Pinhole
        if model_type == "poly4" or model_type == "fisheye_opencv_px" or calib.get("is_fisheye", False):
            k1 = calib.get("c1", calib.get("k1", 0.0))
            k2 = calib.get("c2", calib.get("k2", 0.0))
            k3 = calib.get("c3", calib.get("k3", 0.0))
            k4 = calib.get("c4", calib.get("k4", 0.0))
            camera_model = "OPENCV_FISHEYE"
        else:
            k1 = calib.get("k1", 0.0)
            k2 = calib.get("k2", 0.0)
            p1 = calib.get("p1", 0.0)
            p2 = calib.get("p2", 0.0)
            k3 = calib.get("k3", 0.0)
            camera_model = "OPENCV"

        intrinsics_dict = {
            "camera_model": camera_model,
            "fl_x": K[0, 0],
            "fl_y": K[1, 1],
            "cx": K[0, 2],
            "cy": K[1, 2],
            "k1": k1, "k2": k2, "k3": k3
        }

        if camera_model == "OPENCV_FISHEYE":
            intrinsics_dict["k4"] = k4
        else:
            intrinsics_dict["p1"] = p1
            intrinsics_dict["p2"] = p2

        return img_raw, intrinsics_dict

    def _compute_c2w_matrix(self, odom_trans: np.ndarray, odom_euler_zyx: np.ndarray, 
                            cam_R: np.ndarray, cam_T: np.ndarray, offset_z: float) -> list:
        R_odom = Rotation.from_euler('ZYX', odom_euler_zyx, degrees=False).as_matrix()
        M_v2w = np.eye(4)
        M_v2w[:3, :3] = R_odom
        M_v2w[:3, 3] = odom_trans
        
        M_c2v = np.eye(4)
        M_c2v[:3, :3] = cam_R
        T_adj = cam_T.copy()
        T_adj[2] -= offset_z
        M_c2v[:3, 3] = T_adj
        
        M_c2w = np.dot(M_v2w, M_c2v)
        return M_c2w.tolist()

    def add_frame(self, cam_name: str, img_path: str, timestamp: float, calib: dict, 
                  odom_trans: np.ndarray, odom_euler_zyx: np.ndarray, offset_z: float, global_pcd_path: str = None):
        
        # Procesamos el PCD solo en la primera llamada de la primera cámara
        if global_pcd_path and not self._pcd_converted:
            self.convert_pcd_to_ply(global_pcd_path)
            
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            return
            
        img_processed, intrinsics_dict = self._process_image_and_intrinsics(img_raw, calib)
        
        # Guardamos la imagen incluyendo el cam_name en el archivo para que no se sobreescriban
        img_filename = f"{cam_name}_{timestamp:.3f}.png"
        out_img_path = self.img_dir / img_filename
        cv2.imwrite(str(out_img_path), img_processed)
        
        transform_matrix = self._compute_c2w_matrix(odom_trans, odom_euler_zyx, calib["R"], calib["T"], offset_z)
        h, w = img_processed.shape[:2]
        
        # Añadimos TODO (matriz y cámara) dentro del propio frame
        frame_data = {
            "file_path": f"images/{img_filename}",
            "transform_matrix": transform_matrix,
            "w": w,
            "h": h
        }
        frame_data.update(intrinsics_dict)
        
        self.frames.append(frame_data)

    def finalize_export(self):
        """Escribe el archivo transforms.json global con soporte multi-cámara nativo."""
        if not self.frames:
            print("[ERROR] No hay frames para exportar.")
            return

        # El JSON raíz ya no necesita parámetros de cámara, solo apunta al PLY y a la lista de frames
        transforms_dict = {
            "ply_file_path": self.global_ply_path,
            "frames": self.frames
        }
                    
        json_path = self.output_dir / "transforms.json"
        with open(json_path, 'w') as f:
            json.dump(transforms_dict, f, indent=4)
            
        print(f"\n[3DGS EXPORT] Dataset MULTI-CÁMARA generado exitosamente: {json_path}")
        print(f"[RESUMEN] Total de imágenes procesadas: {len(self.frames)}")