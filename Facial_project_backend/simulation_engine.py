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

def precision_warp(image_np, src_pts, dst_pts, h, w, landmarks):
    try:
        eye_anchors = [lm_pt(landmarks, i, w, h) for i in [33, 133, 362, 263, 1, 4, 10, 152, 234, 454, 61, 291]]
        anchors = np.array([[0,0],[w-1,0],[0,h-1],[w-1,h-1],[w//2,0],[0,h//2],[w-1,h//2],[w//2,h-1]], dtype=np.float32)
        cs, cd = np.vstack([src_pts, eye_anchors, anchors]), np.vstack([dst_pts, eye_anchors, anchors])
        _, idx = np.unique(np.round(cd, 1), axis=0, return_index=True)
        cs, cd = cs[idx], cd[idx]
        grid_res = 32; gy, gx = np.mgrid[0:h:complex(grid_res), 0:w:complex(grid_res)].astype(np.float32)
        rbf_x = Rbf(cd[:,0], cd[:,1], cs[:,0], function='thin_plate', smooth=0.25); rbf_y = Rbf(cd[:,0], cd[:,1], cs[:,1], function='thin_plate', smooth=0.25)
        mx = cv2.resize(rbf_x(gx, gy), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        my = cv2.resize(rbf_y(gx, gy), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return cv2.remap(image_np, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    except: return image_np

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
        log_debug(f"--- Sim Start v22: {feature} ---")
        img = np.array(original_image)
        if img.shape[2] == 4: img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        h_orig, w_orig = img.shape[:2]
        max_dim = 1100
        scale = max_dim / max(h_orig, w_orig)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        
        fl = feature.lower()
        src, dst = [], []
        
        # STRICT INTERNAL INTENSITY CLAMP (65% Max)
        safe_intensity = min(intensity, 65)
        mult = (np.log1p(safe_intensity / 100.0) / np.log1p(1.0)) * 0.75
        
        if "rhino" in fl or "nose" in fl:
            shift = w * 0.04 * mult
            bt = lm_pt(landmarks, 6, w, h); nt = lm_pt(landmarks, 1, w, h); cx = (bt[0]+nt[0])/2.0
            for i in [168, 6, 197, 195]: src.append(lm_pt(landmarks, i, w, h)); dst.append(lm_pt(landmarks, i, w, h))
            for i in [48, 64, 98, 115, 220, 278, 294, 327, 344, 440]:
                p = lm_pt(landmarks, i, w, h); src.append(p); dst.append(move_toward(p, np.array([cx, p[1]]), shift))
            src.append(nt); dst.append(move_direction(nt, 0, -shift*0.2))
        elif "lip" in fl:
            shift = h * 0.02 * mult
            for i in [0, 37, 267, 39, 269, 13]: p = lm_pt(landmarks, i, w, h); src.append(p); dst.append(move_direction(p, 0, -shift))
            for i in [14, 317, 87, 318, 88]: p = lm_pt(landmarks, i, w, h); src.append(p); dst.append(move_direction(p, 0, shift * 1.1))
        elif "chin" in fl:
            shift = h * 0.038 * mult
            for i in [152, 148, 149, 150, 377, 378, 379]: p = lm_pt(landmarks, i, w, h); src.append(p); dst.append(move_direction(p, 0, shift))

        if len(src) > 0: img = precision_warp(img, np.array(src), np.array(dst), h, w, landmarks)
        img = apply_clinical_skin_treatment(img, mask_cv2, feature, landmarks, intensity)

        final_pil = Image.fromarray(img)
        buffered = io.BytesIO(); final_pil.save(buffered, format="JPEG", quality=92)
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    except:
        buffered = io.BytesIO(); original_image.save(buffered, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
