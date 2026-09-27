import numpy as np
import cv2

def transform_lidar_to_camera(lidar_points: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Transforma nube de puntos Nx3 del LiDAR al marco de referencia de la cámara.
    P_cam = (R @ P_lidar.T).T + T
    """
    # Rotación
    points_cam = np.dot(lidar_points, R.T)
    # Traslación
    points_cam += T
    # Filtrar puntos detrás de la cámara (Z <= 0)
    valid_mask = points_cam[:, 2] > 0
    return points_cam[valid_mask]



def project_points_to_image(cartesian_coords: np.ndarray, calib_data: dict) -> np.ndarray:
    """
    Despachador principal que proyecta puntos 3D (N, 3) en coordenadas de cámara 
    a píxeles 2D (N, 2) según el modelo intrínseco.
    """
    model_type = calib_data.get("type", "perspective_opencv_px")
    
    if model_type == "fisheye_opencv_px" or calib_data.get("is_fisheye", False):
        return _project_fisheye_opencv(cartesian_coords, calib_data)
    elif model_type == "poly4":
        return _project_poly4(cartesian_coords, calib_data)
    else:
        return _project_pinhole(cartesian_coords, calib_data)

def _project_fisheye_opencv(coords: np.ndarray, calib_data: dict) -> np.ndarray:
    # Preparar parámetros
    dist = calib_data["distortion"]
    K = calib_data["K"]
    k1, k2 = float(dist.get("fisheye_k1", 0.0)), float(dist.get("fisheye_k2", 0.0))
    k3, k4 = float(dist.get("fisheye_k3", 0.0)), float(dist.get("fisheye_k4", 0.0))
    
    # Coordenadas normalizadas
    x = coords[:, 0] / coords[:, 2]
    y = coords[:, 1] / coords[:, 2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)
    
    # Distorsión
    theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    
    # Manejar división por cero en el centro óptico
    scale = np.ones_like(r)
    non_zero = r > 1e-8
    scale[non_zero] = theta_d[non_zero] / r[non_zero]
    
    x_d = x * scale
    y_d = y * scale
    
    # Proyección a plano de imagen
    u = K[0, 0] * x_d + K[0, 2]
    v = K[1, 1] * y_d + K[1, 2]
    
    return np.column_stack((u, v))

def _project_pinhole(coords: np.ndarray, calib_data: dict) -> np.ndarray:
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    dist = calib_data["distortion"]
    
    dist_coeffs = np.array([
        float(dist.get("pinhole_k1", 0.0)), float(dist.get("pinhole_k2", 0.0)),
        float(dist.get("pinhole_p1", 0.0)), float(dist.get("pinhole_p2", 0.0)),
        float(dist.get("pinhole_k3", 0.0))
    ], dtype=np.float64)
    
    pixels, _ = cv2.projectPoints(coords, rvec, tvec, calib_data["K"], dist_coeffs)
    return pixels[:, 0, :]

#def _project_poly4(coords: np.ndarray, calib_data: dict) -> np.ndarray: