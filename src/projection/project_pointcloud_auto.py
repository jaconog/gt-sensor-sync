import numpy as np
import cv2
from scipy.spatial.transform import Rotation

def _valeo_opencv_pinhole(cartesian_coords, calib):
    x, y, z = cartesian_coords[:, :, 0], cartesian_coords[:, :, 1], cartesian_coords[:, :, 2]
    z_safe = np.where(z == 0, 1e-8, z)
    x_p, y_p = x / z_safe, y / z_safe
    r_squ = x_p ** 2 + y_p ** 2
    
    k1, k2, k3 = calib['k1'], calib['k2'], calib['k3']
    k4, k5, k6 = calib.get('k4', 0), calib.get('k5', 0), calib.get('k6', 0)
    p1, p2 = calib['p1'], calib['p2']
    
    poly = (1.0 + k1 * r_squ + k2 * r_squ**2 + k3 * r_squ**3) / (1.0 + k4 * r_squ + k5 * r_squ**2 + k6 * r_squ**3)
    x_pp = x_p * poly + 2 * p1 * x_p * y_p + p2 * (r_squ + 2 * x_p ** 2)
    y_pp = y_p * poly + p1 * (r_squ + 2 * y_p ** 2) + 2 * p2 * x_p * y_p
    
    u = calib['fx'] * x_pp + calib['cx']
    v = calib['fy'] * y_pp + calib['cy']
    return np.stack((u, v), axis=2)

def _valeo_opencv_fisheye(cartesian_coords, calib):
    x, y, z = cartesian_coords[:, :, 0], cartesian_coords[:, :, 1], cartesian_coords[:, :, 2]
    z_safe = np.where(z == 0, 1e-8, z)
    a, b = x / z_safe, y / z_safe
    r = np.sqrt(a ** 2 + b ** 2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    
    k1, k2, k3, k4 = calib['k1'], calib['k2'], calib['k3'], calib.get('k4', 0)
    theta_d = theta + k1*theta**3 + k2*theta**5 + k3*theta**7 + k4*theta**9
    
    x_p = np.where(r != 0, theta_d / r * a, 0.)
    y_p = np.where(r != 0, theta_d / r * b, 0.)
    
    alpha = calib.get('alpha', 0.0)
    u = calib['fx'] * x_p + calib['fy'] * alpha * y_p + calib['cx']
    v = calib['fy'] * y_p + calib['cy']
    return np.stack((u, v), axis=2)

def _valeo_poly4(cartesian_coords, calib):
    x, y, z = cartesian_coords[:, :, 0], cartesian_coords[:, :, 1], cartesian_coords[:, :, 2]
    chi = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(chi, z)
    
    c1, c2, c3, c4 = calib['c1'], calib['c2'], calib['c3'], calib['c4']
    rho = c1 * theta + c2 * theta**2 + c3 * theta**3 + c4 * theta**4
    
    u_p = np.where(chi != 0., rho * x / chi, 0.)
    v_p = np.where(chi != 0., rho * y / chi, 0.)
    
    aspect_ratio = calib.get('aspect_ratio', 1.0)
    u = u_p + calib['cx']
    v = v_p * aspect_ratio + calib['cy']
    return np.stack((u, v), axis=2)

def transform_lidar_to_vehicle(lidar_points: np.ndarray, R_lidar: np.ndarray, T_lidar: np.ndarray) -> np.ndarray:
    """
    Transforma la nube de puntos del sensor LiDAR al centro del vehículo (ISO Frame).
    """
    # P_veh = P_lidar @ R_lidar^T + T_lidar
    return np.dot(lidar_points, R_lidar.T) + T_lidar

def transform_vehicle_to_camera(vehicle_points: np.ndarray, R_cam: np.ndarray, T_cam: np.ndarray, offset_z: float = 0.0) -> np.ndarray:
    """
    Transforma desde el centro del vehículo a la lente de la cámara.
    La matriz extrínseca de Valeo mapea Cam -> Veh, por lo que P_cam = (P_veh - T_cam) @ R_cam
    """
    T_adj = T_cam.copy()
    T_adj[2] -= offset_z  # Compensación empírica chasis-suelo
    
    # 1. Transformación al marco de la cámara
    points_cam = np.dot(vehicle_points - T_adj, R_cam)
    
    # 2. Z-Culling (Evitar puntos detrás del plano focal)
    mask_z = points_cam[:, 2] > 0.1
    points_cam = points_cam[mask_z]
    
    # 3. Frustum Culling Angular (Evita el efecto Especular / Wrap-around de Fisheye)
    chi = np.sqrt(points_cam[:, 0]**2 + points_cam[:, 1]**2)
    theta = np.arctan2(chi, points_cam[:, 2])
    
    # Límite de seguridad de 105 grados desde el centro óptico para SVS/AD
    mask_fov = theta < np.deg2rad(105) 
    
    return points_cam[mask_fov]

def project_points_to_image(cam_coords: np.ndarray, calib_data: dict) -> np.ndarray:
    """ Despachador matemático exacto de Valeo. """
    model_type = calib_data.get("type", "perspective_opencv_px")
    coords_exp = np.expand_dims(cam_coords, axis=0)
    
    if calib_data.get("is_fisheye", False) or model_type == "fisheye_opencv_px":
        pixels = _valeo_opencv_fisheye(coords_exp, calib_data)
    elif model_type == "poly4" or calib_data.get("c1") != 0.0:
        pixels = _valeo_poly4(coords_exp, calib_data)
    else:
        pixels = _valeo_opencv_pinhole(coords_exp, calib_data)
        
    return pixels[0]

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

def _project_poly4(coords: np.ndarray, calib_data: dict) -> np.ndarray:
    """Proyección para modelo polinómico (Lentes SVS de Valeo)."""
    dist = calib_data["distortion"]
    c1, c2 = float(dist.get("dist_c1", 0.0)), float(dist.get("dist_c2", 0.0))
    c3, c4 = float(dist.get("dist_c3", 0.0)), float(dist.get("dist_c4", 0.0))
    cx, cy = calib_data["K"][0, 2], calib_data["K"][1, 2]
    aspect_ratio = float(dist.get("pixel_aspect_ratio_x_by_y", 1.0))

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    chi = np.sqrt(x**2 + y**2)
    theta = np.arctan2(chi, z)
    rho = c1 * theta + c2 * theta**2 + c3 * theta**3 + c4 * theta**4

    u_p = np.zeros_like(x)
    v_p = np.zeros_like(y)
    
    # Evitar división por cero en el centro óptico
    non_zero = chi > 1e-8
    u_p[non_zero] = rho[non_zero] * x[non_zero] / chi[non_zero]
    v_p[non_zero] = rho[non_zero] * y[non_zero] / chi[non_zero]

    u = u_p + cx
    v = (v_p * aspect_ratio) + cy

    return np.column_stack((u, v))

def transform_slam_to_camera(slam_points: np.ndarray, 
                             odom_translation: np.ndarray, 
                             odom_euler_zyx: np.ndarray, 
                             cam_R: np.ndarray, 
                             cam_T: np.ndarray,
                             offset_z: float = 0.0) -> np.ndarray:
    """
    Transformación de Alta Precisión: Mapa SLAM -> Vehículo -> Cámara.
    """
    # 1. Secuencia Intrínseca Automotriz: Yaw(Z) -> Pitch(Y) -> Roll(X)
    R_odom = Rotation.from_euler('ZYX', odom_euler_zyx, degrees=False).as_matrix()
    
    # P_vehiculo = (P_slam - T_odom) * R_odom
    points_veh = np.dot(slam_points - odom_translation, R_odom)
    
    # 2. Vehículo -> Cámara (Ajuste de offset_z del chasis)
    T_adj = cam_T.copy()
    T_adj[2] -= offset_z  
    
    points_cam = np.dot(points_veh - T_adj, cam_R)
    
    # 3. Z-Culling (Evitar puntos detrás del plano focal)
    mask_z = points_cam[:, 2] > 0.1
    points_cam = points_cam[mask_z]
    
    # 4. Frustum Culling Angular estricto (Previene imágenes especulares)
    chi = np.sqrt(points_cam[:, 0]**2 + points_cam[:, 1]**2)
    theta = np.arctan2(chi, points_cam[:, 2])
    mask_fov = theta < np.deg2rad(105) 
    
    return points_cam[mask_fov]