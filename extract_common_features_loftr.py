import torch
import cv2
import numpy as np
from functools import reduce
import pycolmap
from loftr import LoFTR, default_cfg

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")



def extract_features(matcher, image_pair, filter_with_conf=True):
    sz_img = (1280,960)
    img0_raw = cv2.imread(image_pair[0], cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(image_pair[1], cv2.IMREAD_GRAYSCALE)
    img0_raw = cv2.resize(img0_raw, sz_img)
    img1_raw = cv2.resize(img1_raw, sz_img)

    img0 = torch.from_numpy(img0_raw)[None][None].to(device, dtype=torch.float32) / 255.0
    img1 = torch.from_numpy(img1_raw)[None][None].to(device, dtype=torch.float32) / 255.0
    batch = {'image0': img0, 'image1': img1}

    #############################  TODO 4.4 BEGIN  ############################
    # Inference with LoFTR and get prediction
    # The model `matcher` takes a dict with keys `image0` and `image1` as input,
    # and writes fine features back to the same dict.
    # You can get the results with keys:
    #   key         :   value
    #   'mkpts0_f'  :   matching feature coordinates in image0 (N x 2)
    #   'mkpts1_f'  :   matching feature coordinates in image1 (N x 2)
    #   'mconf'     :   confidence of each matching feature    (N x 1)
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        if filter_with_conf:
            mconf = batch['mconf'].cpu().numpy()
            mask = mconf > 0.5  # filter feature with confidence higher than 0.5
            mkpts0 = mkpts0[mask]
            mkpts1 = mkpts1[mask]

    #############################  TODO 4.4 END  ############################
        print("matches:", mkpts0.shape[0])
        # RANSAC options
        ransac_options = pycolmap.RANSACOptions()
        ransac_options.max_error = 2.0  # Max pixel reprojection error
        ransac_options.confidence = 0.99  # Confidence level
        ransac_options.min_num_trials = 100  # Minimum RANSAC iterations
        ransac_options.max_num_trials = 1000  # Maximum RANSAC iterations
        inliers = pycolmap.fundamental_matrix_estimation(mkpts0, mkpts1, ransac_options)['inliers']
        print("inliers:", np.count_nonzero(inliers))

        mkpts0 = mkpts0[inliers]
        mkpts1 = mkpts1[inliers]

        return mkpts0, mkpts1


#############################  TODO 4.5 BEGIN  ############################
from scipy.spatial.distance import cdist


def intersect2D_multiple(arrays, threshold=0):
    
    def pairwise_intersect(a, b):
        distances = cdist(a, b)
        close_pairs = np.argwhere(distances < threshold)
        unique_a_indices = np.unique(close_pairs[:, 0])
        return a[unique_a_indices]

    
    result = reduce(pairwise_intersect, arrays)
    

    unique_rows = []
    for row in result:
        if not any(np.allclose(row, ur, atol=threshold) for ur in unique_rows):
            unique_rows.append(row)
    
    return np.array(unique_rows)


def find_closest_point(point, array):
    
    distances = np.linalg.norm(array - point, axis=1)
    
    
    closest_index = np.argmin(distances)
    
    return closest_index


     
def find_common_features(images, matcher):
    num_frames = len(images)
    all_features_first = []
    other_pair = []
    
    # Extract features for all pairs with the first image
    for i in range(1, num_frames):
        mkpts0, mkpts1 = extract_features(matcher, (images[0], images[i]))
        all_features_first.append(mkpts0)
        other_pair.append(mkpts1)

    
    print(all_features_first[-1].shape)
    intersected = intersect2D_multiple(all_features_first, threshold = 1)
    print(intersected.shape)

    N = intersected.shape[0]
    F = len(images)

    common_features = np.zeros((N,F,2))
    
    for i in range(intersected.shape[0]):
        common_features[i,0,:] = intersected[i]
        for j in range(len(other_pair)):
            q = find_closest_point(intersected[i], all_features_first[j])
            common_features[i,j+1,:] = other_pair[j][q]
           
    common_features *= 3.15
    common_features[:,:,0] -= 2035.7826140150432
    common_features[:,:,1] -= 1500.256751469342
    
    common_features[:,:,0] /= 3108.427510480831
    common_features[:,:,1] /= 3103.95507309346

    
  
    print(common_features.shape)
    print(common_features)
    return common_features
  

##############################  TODO 4.5 END  #############################


def main():
    # Dataset
    image_dir = './data/pennlogo/'
    image_names = ['IMG_8657.jpg', 'IMG_8658.jpg', 'IMG_8659.jpg', 'IMG_8660.jpg', 'IMG_8661.jpg']
    K = np.array([[3108.427510480831, 0.0, 2035.7826140150432], 
                  [0.0, 3103.95507309346, 1500.256751469342], 
                  [0.0, 0.0, 1.0]])

    # LoFTR model
    matcher = LoFTR(config=default_cfg)
    # Load pretrained weights
    checkpoint = torch.load("weights/outdoor_ds.ckpt", map_location=device, weights_only=True)
    matcher.load_state_dict(checkpoint["state_dict"])
    matcher = matcher.eval().to(device)

    #############################  TODO 4.5 BEGIN  ############################
    # Find common features
    # You can add any helper functions you need
    image_paths = [image_dir + name for name in image_names]
    common_features = find_common_features(image_paths, matcher)
    
    ##############################  TODO 4.5 END  #############################

    np.savez("loftr_features.npz", data=common_features, image_names=image_names, intrinsic=K)
    


if __name__ == '__main__':
    main()
