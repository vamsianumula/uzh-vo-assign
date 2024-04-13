import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import warnings

def getDisparity(left_img, right_img, patch_radius, min_disp, max_disp):
    """
    left_img and right_img are both H x W and you should return a H x W matrix containing the disparity d for
    each pixel of left_img. Set disp_img to 0 for pixels where the SSD and/or d is not defined, and for
    d estimates rejected in Part 2. patch_radius specifies the SSD patch and each valid d should satisfy
    min_disp <= d <= max_disp.
    """
    # TODO: Your code here
    dispMap = np.zeros_like(left_img, dtype=np.float32)
    
    for l_row in range(left_img.shape[0]):
        for l_col in range(left_img.shape[1]):
            # print(f"-----------Index {l_row, l_col}-------------")
            # print(f"Left patch coords {l_row-patch_radius,l_row+patch_radius,l_col-patch_radius,l_col+patch_radius}")
            if l_row-patch_radius<0 or l_row+patch_radius>= left_img.shape[0] or l_col-patch_radius<0 or l_col+patch_radius>=left_img.shape[1]:
                dispMap[l_row, l_col] = 0
                # print("Left patch out of bounds")
                continue
            
            left_patch = left_img[l_row-patch_radius:l_row+patch_radius,l_col-patch_radius:l_col+patch_radius]
            left_patch = np.expand_dims(left_patch.ravel(), 0)
            right_patch_candidates = []
            right_patch_disp = []
            for disp in range(min_disp, max_disp+1):
                # print(f"Right patch - disp {disp}, row_coords: {l_row-patch_radius-disp, l_row+patch_radius-disp}")
                if l_col-patch_radius-disp<0 or l_col+patch_radius-disp>= right_img.shape[1]:
                    # print("Right patch out of bounds")
                    continue
                right_patch = right_img[l_row-patch_radius:l_row+patch_radius, l_col-patch_radius-disp:l_col+patch_radius-disp]
                right_patch_candidates.append(right_patch.ravel())
                right_patch_disp.append(disp)
            right_patch_candidates = np.array(right_patch_candidates)
            if right_patch_candidates.shape[0]==0:
                continue
            # print(right_patch_candidates.shape, left_patch.shape)
            dist = cdist(right_patch_candidates, left_patch, 'sqeuclidean')
            idx = np.argmin(dist)
            optimal_disp = right_patch_disp[idx]
            threshold_score = [1.5*dist[idx]]
            
            if threshold_score[0] < 1.5e-3 or dist[idx][0] < 1e-3:
                continue
            count = np.count_nonzero(dist.ravel()<threshold_score)
            if count>2:
                continue
            # print(dist)
            # print(f"Min disp: {right_patch_disp[idx]},idx: {idx}")
            if optimal_disp==min_disp or optimal_disp==max_disp:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                disp_min=-1
                try:
                    fit = np.polyfit(right_patch_disp[idx-1:idx+2], dist.ravel()[idx-1:idx+2], 2)
                    disp_min = -fit[1]/(2*fit[0])
                    dispMap[l_row, l_col] = disp_min
                except Warning as e:
                    dispMap[l_row, l_col] = optimal_disp
                    # print(optimal_disp, disp_min)
            # print(dispMap[l_row, l_col], type(dispMap[l_row, l_col]))
    return dispMap
                
