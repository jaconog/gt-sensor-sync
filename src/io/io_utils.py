import cv2
import numpy as np
import open3d as o3d
from pathlib import Path

class LocalDataLoader:
    def __init__(self, local_base_path: str = "D:/VALEO/tmp"):
        self.local_base_path = Path(local_base_path)

    def resolve_local_path(self, gs_uri: str) -> Path:
        """
        Convierte la URI de GCS a la ruta del sistema de archivos local de Windows.
        Ej: gs://VALEO/data/... -> D:/VALEO/tmp/VALEO/data/...
        """
        if not gs_uri.startswith("gs://"):
            return Path(gs_uri)
        
        # Eliminar 'gs://' y construir la ruta absoluta local
        clean_path = gs_uri.replace("gs://", "")
        return self.local_base_path / clean_path

    def load_and_undistort_image(self, gs_uri: str, calib_data: dict) -> np.ndarray:
        local_path = self.resolve_local_path(gs_uri)
        
        if not local_path.exists():
            print(f"[WARNING] Archivo no encontrado localmente: {local_path}")
            return None

        image = cv2.imread(str(local_path))
        if image is None:
            return None

        if not calib_data:
            return image

        # Garantizar que K sea float64 y contigua
        K = np.ascontiguousarray(calib_data["K"], dtype=np.float64)
        h, w = image.shape[:2]

        if calib_data.get("is_fisheye", False):
            # ---------------- FISHEYE MODEL (Cámaras SVS) ----------------
            # Formato estricto requerido por cv2.fisheye: (4, 1) float64
            dist_coeffs = np.array([
                [float(calib_data["distortion"].get("fisheye_k1", 0.0))],
                [float(calib_data["distortion"].get("fisheye_k2", 0.0))],
                [float(calib_data["distortion"].get("fisheye_k3", 0.0))],
                [float(calib_data["distortion"].get("fisheye_k4", 0.0))]
            ], dtype=np.float64)

            try:
                new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, dist_coeffs, (w, h), np.eye(3, dtype=np.float64), balance=1.0
                )
                undistorted_img = cv2.fisheye.undistortImage(
                    image, K, dist_coeffs, Knew=new_K
                )
            except cv2.error as e:
                print(f"[ERROR] Fallo en rectificación Fisheye: {e}")
                return image
            
        else:
            # ---------------- PINHOLE MODEL (Cámaras AD) ----------------
            dist_coeffs = np.array([
                float(calib_data["distortion"].get("pinhole_k1", 0.0)),
                float(calib_data["distortion"].get("pinhole_k2", 0.0)),
                float(calib_data["distortion"].get("pinhole_p1", 0.0)),
                float(calib_data["distortion"].get("pinhole_p2", 0.0)),
                float(calib_data["distortion"].get("pinhole_k3", 0.0))
            ], dtype=np.float64)
            
            new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
            undistorted_img = cv2.undistort(image, K, dist_coeffs, None, new_K)
        
        return undistorted_img

    def load_pointcloud(self, gs_uri: str) -> np.ndarray:
        """Carga una nube de puntos desde disco en formato numpy Nx3."""
        local_path = self.resolve_local_path(gs_uri)
        
        if not local_path.exists():
            print(f"[WARNING] Archivo de nube de puntos no encontrado: {local_path}")
            return np.empty((0, 3))

        try:
            pcd = o3d.io.read_point_cloud(str(local_path))
            return np.asarray(pcd.points)
        except Exception as e:
            print(f"[ERROR] Fallo al leer {local_path}: {str(e)}")
            return np.empty((0, 3))

    