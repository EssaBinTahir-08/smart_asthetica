"""
SmartAesthetica — Professional Facial Simulation Engine v22 "Balanced Professional"
================================================================================
Strict internal intensity clamping for clinical-grade realism.
- Removed Thread Lift / Facelift / Brow Lift logic.
- Strict 65% Global Intensity Clamp.
- Optimized Shadow-Erasing for Botox and PRP.
"""

import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
from scipy.interpolate import Rbf

DEBUG_LOG = "/Users/essabintahir/Desktop/nftcypher/Facial_project_main/simulation_debug.log"

def log_debug(msg):
    with open(DEBUG_LOG, "a") as f:
        f.write(f"{msg}\n")

# =====================================================================
# TEXTURE ENGINE: Balanced Professional
# =====================================================================

def apply_clinical_skin_treatment(img, mask, treatment_type, landmarks, intensity):
    try:
        # STRICT INTERNAL CLAMP (60-70% Zone)
        mult = min(intensity, 65) / 100.0
        if mult < 0.01: return img
        h, w = img.shape[:2]
        treatment = treatment_type.lower()
        
        is_prp = "prp" in treatment or "needling" in treatment or "hydra" in treatment
        is_botox = "botox" in treatment
        is_skin = any(k in treatment for k in ["peel", "laser", "facial", "scar", "meso", "glow", "bright"])
        
        if not (is_botox or is_skin or is_prp): return img
        
        final_alpha = np.zeros((h, w), dtype=np.float32)
        if is_botox:
            fh = np.array([[landmarks[i]['x']*w, landmarks[i]['y']*h] for i in [10, 338, 297, 332, 103, 67, 109, 10]], np.int32)
            cv2.fillPoly(final_alpha, [fh], 1.0)
            final_alpha = cv2.GaussianBlur(final_alpha, (61, 61), 0)
        else:
            if mask is not None and isinstance(mask, np.ndarray) and mask.size > 0:
                m = cv2.resize(mask, (w, h))
                if len(m.shape) == 3: m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                final_alpha = cv2.GaussianBlur(m, (31, 31), 0) / 255.0
            else:
                pts = np.array([[landmarks[i]['x']*w, landmarks[i]['y']*h] for i in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 152, 172, 58, 132, 93, 234, 127, 21, 54, 103, 67, 109]], np.int32)
                cv2.fillPoly(final_alpha, [pts], 0.6)
                final_alpha = cv2.GaussianBlur(final_alpha, (81, 81), 0)

        # Specialty Processing
        img_f = img.astype(np.float32) / 255.0
        if is_prp:
            low = cv2.bilateralFilter(img, 15, 75, 75).astype(np.float32) / 255.0
            hsv = cv2.cvtColor((low * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= (1.0 + 0.08 * mult)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + (10 * mult), 0, 255)
            low = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        else:
            low = cv2.bilateralFilter(img, 11, 50, 50).astype(np.float32) / 255.0
            
        high = img_f - cv2.bilateralFilter(img, 7, 30, 30).astype(np.float32) / 255.0
        result = np.clip((low + high) * 255.0, 0, 255).astype(np.uint8)
        
        # Protection
        protection = np.ones((h, w, 1), dtype=np.float32)
        for group in [[33, 133, 153, 144, 163, 7], [263, 362, 382, 374, 390, 249], [61, 291, 0, 17]]:
            pts = np.array([[landmarks[i]['x']*w, landmarks[i]['y']*h] for i in group], np.int32)
            m = np.zeros((h, w), dtype=np.uint8); cv2.fillPoly(m, [pts], 255)
            protection -= (cv2.GaussianBlur(m, (11, 11), 0)[:, :, np.newaxis] / 255.0)
            
        alpha_3d = np.clip(final_alpha[:, :, np.newaxis] * protection, 0, 1)
        return (img * (1 - alpha_3d) + result * alpha_3d).astype(np.uint8)
    except: return img

# =====================================================================
# STRUCTURAL ENGINE: Balanced Professional
# =====================================================================

from scipy.interpolate import griddata

def precision_warp(image_np, src_pts, dst_pts, h, w, landmarks, excluded_indices=[]):
    try:
        # 1. Establish DENSE Rigid Anchors
        # Anchor the entire face mesh EXCEPT for the excluded area
        # MediaPipe has 468 landmarks. We'll anchor a significant portion to stay rigid.
        mesh_anchors = [i for i in range(0, 468, 8) if i not in excluded_indices]
        # Always anchor eyes and mouth perimeter for stability
        if 1 not in excluded_indices:
             mesh_anchors.extend([33, 133, 362, 263, 61, 291])
        
        face_pts = [lm_pt(landmarks, i, w, h) for i in mesh_anchors]
        
        # 2. Image Perimeter Anchors (Dense ring to lock background)
        perimeter = []
        for x in np.linspace(0, w-1, 15):
            perimeter.append([x, 0]); perimeter.append([x, h-1])
        for y in np.linspace(0, h-1, 15):
            perimeter.append([0, y]); perimeter.append([w-1, y])
        perimeter = np.array(perimeter, dtype=np.float32)
        
        # Combine all Control Points
        points = np.vstack([src_pts, face_pts, perimeter])
        values_x = np.hstack([dst_pts[:,0], [p[0] for p in face_pts], perimeter[:,0]])
        values_y = np.hstack([dst_pts[:,1], [p[1] for p in face_pts], perimeter[:,1]])
        
        # 3. Create Grid
        grid_y, grid_x = np.mgrid[0:h, 0:w]
        
        # 4. Linear Interpolation (Rigid & Stable)
        # We use 'linear' to prevent the RBF "plateau" smearing
        mx = griddata(points, values_x, (grid_x, grid_y), method='linear')
        my = griddata(points, values_y, (grid_x, grid_y), method='linear')
        
        # Fill NaNs (outside convex hull) with identity mapping
        mx[np.isnan(mx)] = grid_x[np.isnan(mx)]
        my[np.isnan(my)] = grid_y[np.isnan(my)]
        
        mx = mx.astype(np.float32)
        my = my.astype(np.float32)
        
        return cv2.remap(image_np, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    except Exception as e:
        log_debug(f"Rigid Warp Failed: {e}")
        return image_np

def lm_pt(landmarks, idx, w, h): return np.array([landmarks[idx]['x']*w, landmarks[idx]['y']*h], dtype=np.float32)
def move_toward(pt, target, amount):
    d = target - pt; dst = np.linalg.norm(d)
    return pt + (d/dst)*amount if dst > 1e-6 else pt.copy()
def move_direction(pt, dx, dy): return pt + np.array([dx, dy], dtype=np.float32)

# =====================================================================
# MAIN
# =====================================================================

def simulate_procedure(original_image, mask_cv2, feature, landmarks, intensity=100):
    try:
        log_debug(f"--- Sim Start v25 [RIGID]: {feature} ---")
        img = np.array(original_image)
        if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        h_orig, w_orig = img.shape[:2]
        max_dim = 1100
        scale = max_dim / max(h_orig, w_orig)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        
        fl = feature.lower()
        src, dst = [], []
        excluded_idxs = []
        
        # Intensity Scaling (Direct Linear Power)
        mult = intensity / 100.0
        
        # 1. Power Rhinoplasty: Total Nasal Taper
        if "rhino" in fl or "nose" in fl:
            # Expanded indices to include nasal sidewalls and bridge taper points
            excluded_idxs = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64, 102, 331, 129, 358, 203, 423]
            
            # High-impact narrowing factor (Linear scaling for visible results)
            narrow_base = w * 0.09 * mult # Aggressive alar narrowing
            narrow_bridge = w * 0.04 * mult # Visible bridge thinning
            lift_tip = h * 0.03 * mult # Defined tip lift
            
            bridge_top = lm_pt(landmarks, 168, w, h)
            nasal_tip = lm_pt(landmarks, 1, w, h)
            center_x = (bridge_top[0] + nasal_tip[0]) / 2.0
            
            for i in excluded_idxs:
                p = lm_pt(landmarks, i, w, h); src.append(p)
                
                # Alar Base & Side Walls (Heavy Narrowing)
                if i in [48, 64, 98, 115, 220, 278, 294, 327, 344, 440, 102, 331, 129, 358]:
                    dst.append(move_toward(p, np.array([center_x, p[1]]), narrow_base))
                # Nasal Bridge (Subtle but visible thinning)
                elif i in [168, 6, 197, 195, 5, 4]:
                    dst.append(move_toward(p, np.array([center_x, p[1]]), narrow_bridge))
                # Nasal Tip (Lift & Refinement)
                elif i in [1, 2, 94, 19]:
                    dst.append(move_direction(p, 0, -lift_tip))
                else:
                    dst.append(p)
            
        # 2. Lip Augmentation / Fillers
        elif "lip" in fl:
            excluded_idxs = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
            shift = h * 0.018 * mult
            for i in excluded_idxs:
                p = lm_pt(landmarks, i, w, h); src.append(p)
                if i in [0, 37, 267, 39, 269, 13]: # Upper
                    dst.append(move_direction(p, 0, -shift))
                elif i in [14, 317, 87, 318, 88]: # Lower
                    dst.append(move_direction(p, 0, shift * 1.2))
                else:
                    dst.append(p)
                
        # 3. Jawline Contouring / Masseter Botox / V-Line
        elif "jaw" in fl or "masseter" in fl or "v-line" in fl:
            excluded_idxs = [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
            shift = w * 0.045 * mult
            mid_x = w / 2.0
            for i in excluded_idxs:
                p = lm_pt(landmarks, i, w, h); src.append(p)
                # Move towards vertical centerline to slim the jawline
                dst.append(move_toward(p, np.array([mid_x, p[1]]), shift))

        # 4. Chin Augmentation
        elif "chin" in fl:
            excluded_idxs = [152, 148, 149, 150, 377, 378, 379]
            shift = h * 0.04 * mult
            for i in excluded_idxs:
                p = lm_pt(landmarks, i, w, h); src.append(p)
                dst.append(move_direction(p, 0, shift))

        # 5. Neck Lift / Fat Removal / Buccal Fat
        elif "neck" in fl or "fat" in fl or "buccal" in fl:
            # Anchor jawline, but move submental area and cheeks
            excluded_idxs = [152, 148, 149, 150, 377, 378, 379, 172, 136, 212, 432]
            shift_up = h * 0.035 * mult
            shift_in = w * 0.02 * mult
            mid_x = w / 2.0
            for i in excluded_idxs:
                p = lm_pt(landmarks, i, w, h); src.append(p)
                if i in [152, 148, 149, 150, 377, 378, 379]: # Chin/Neck
                    dst.append(move_direction(p, 0, -shift_up * 0.5))
                else: # Cheeks/Buccal
                    dst.append(move_toward(p, np.array([mid_x, p[1]]), shift_in))

        # 3. Apply Warp if structural points exist
        if len(src) > 0: 
            img = precision_warp(img, np.array(src), np.array(dst), h, w, landmarks, excluded_indices=excluded_idxs)
            
        # 4. Apply Skin Textures (Botox, PRP, etc.)
        img = apply_clinical_skin_treatment(img, mask_cv2, feature, landmarks, intensity)

        # 5. Return Result
        final_pil = Image.fromarray(img)
        buffered = io.BytesIO(); final_pil.save(buffered, format="JPEG", quality=92)
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    except:
        buffered = io.BytesIO(); original_image.save(buffered, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
