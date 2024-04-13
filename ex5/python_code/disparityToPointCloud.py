import numpy as np


def disparityToPointCloud(disp_img, K, baseline, left_img):
    """
    points should be Nx3 and intensities N, where N is the amount of pixels which have a valid disparity.
    I.e., only return points and intensities for pixels of left_img which have a valid disparity estimate!
    The i-th intensity should correspond to the i-th point.
    """
    pass
    # TODO: Your code here
    points = []
    intensities = []
    for l_row in range(left_img.shape[0]):
        for l_col in range(left_img.shape[1]):
            if disp_img[l_row, l_col]==0:
                continue
            disp = disp_img[l_row, l_col]
            Kinv = np.linalg.inv(K)
            p_left = np.array([l_col, l_row, 1])
            p_right = np.array([l_col-disp, l_row, 1])
            # print(Kinv.shape, ().shape)
            pleft = Kinv@ p_left 
            pright = Kinv @ p_right
            A= np.column_stack((pleft,-pright))
            b = np.array([baseline, 0, 0])
            lam = np.linalg.inv((A.T)@A)@(A.T)@b 
            P = lam[0]*Kinv@ np.array([l_col, l_row,1])
            points.append(P)
            intensities.append(left_img[l_row, l_col])
    return np.array(points), np.array(intensities)
            
            
            
