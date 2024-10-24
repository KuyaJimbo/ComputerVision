# James Ocampo Test
import cv2
import numpy as np

def find(fpath):
    # Read images
    reference = cv2.imread('reference.png', cv2.IMREAD_COLOR)
    ref_logo = reference[1855:2065, 745:1295]
    # reduce size to 0.3
    ref_logo = cv2.resize(ref_logo, (0,0), fx=0.3, fy=0.3)

    search = cv2.imread(fpath, cv2.IMREAD_COLOR)
    # reduce size to 0.8
    search = cv2.resize(search, (0,0), fx=0.8, fy=0.8)


    # Initialize SIFT
    sift = cv2.SIFT_create(nfeatures=1500)  # Limit features to save memory
    # sift = cv2.SIFT_create(nfeatures=500)  # Limit to 500 features

    # Detect and compute SIFT features
    keypoints1, descriptors1 = sift.detectAndCompute(ref_logo, None)
    keypoints2, descriptors2 = sift.detectAndCompute(search, None)
    d1 = descriptors1.reshape((descriptors1.shape[0], 1, descriptors1.shape[1]))
    d2 = descriptors2.reshape((1, descriptors2.shape[0], descriptors2.shape[1]))
    dist = np.sum((d1 - d2)**2, axis=2) # Compute the Euclidean distance between the two sets of descriptors

    pt1 = []
    pt2 = []
    good_matches = []

    for i in range(len(dist)):
        a, b = np.argpartition(dist[i], kth=2)[:2]
        A, B = dist[i,a], dist[i,b]
        if B < A:
            A,B = B,A
            a,b=b,a
        r = A / B
        if r < 0.5:
            x1, y1 = keypoints1[i].pt
            x2, y2 = keypoints2[a].pt
            pt1.append((x1,y1))
            pt2.append((x2,y2))
            good_matches.append([pt1[-1], pt2[-1]])

    # Convert good_matches points to numpy arrays
    pt1_np = np.float32(pt1)
    pt2_np = np.float32(pt2)

    # RANSAC if there are at least 4 points
    if len(pt1_np) > 4:
        # Use RANSAC to estimate a homography matrix and remove outliers
        H, mask = cv2.findHomography(pt1_np, pt2_np, cv2.RANSAC)

        # Filter matches using the RANSAC mask
        inliers_pt1 = []    # List to store inliers from ref_logo
        inliers_pt2 = []    # List to store inliers from search
        for i in range(len(mask)):
            if mask[i] and (pt1[i] not in inliers_pt1 and pt2[i] not in inliers_pt2):
                inliers_pt1.append(pt1[i])
                inliers_pt2.append(pt2[i])
    else:
        inliers_pt1 = pt1_np
        inliers_pt2 = pt2_np

    # if there are inliers, calculate the x offset
    if len(inliers_pt1) > 1:
        ref_dx = inliers_pt1[0][0] - inliers_pt1[1][0]
        ref_dy = inliers_pt1[0][1] - inliers_pt1[1][1]
        search_dx = inliers_pt2[0][0] - inliers_pt2[1][0]
        search_dy = inliers_pt2[0][1] - inliers_pt2[1][1]

        ref_distance = np.sqrt(ref_dx**2 + ref_dy**2)
        search_distance = np.sqrt(search_dx**2 + search_dy**2)

        scale_factor = search_distance / ref_distance

        ref_angle = np.arctan2(ref_dy, ref_dx)
        search_angle = np.arctan2(search_dy, search_dx)
        rotation_angle = np.degrees(search_angle - ref_angle)
        rotation_rad = np.radians(rotation_angle)

        # calculate search's logo center based off points
        logo_center_x = 0
        logo_center_y = 0
        for i in range(len(inliers_pt2)):
            logo_center_x += inliers_pt2[i][0]
            logo_center_y += inliers_pt2[i][1]
        logo_center_x /= len(inliers_pt2)
        logo_center_y /= len(inliers_pt2)

        # Calculate real object center with rotation and scale
        ref_offset_x = 2 * 0.3
        ref_offset_y = -190 * 0.3

        scaled_offset_x = ref_offset_x * scale_factor * 1.25
        scaled_offset_y = ref_offset_y * scale_factor * 1.25

        rotated_offset_x = scaled_offset_x * np.cos(rotation_rad) - scaled_offset_y * np.sin(rotation_rad)
        rotated_offset_y = scaled_offset_x * np.sin(rotation_rad) + scaled_offset_y * np.cos(rotation_rad)

        real_center_x = logo_center_x + rotated_offset_x
        real_center_y = logo_center_y + rotated_offset_y

        # calculate the bounding box
        ref_obj_height = 3360 * 0.3

        # scale dimensions
        scaled_obj_height = ref_obj_height * scale_factor

        # print X Y H A (x_position, y_position, height, angle)
        if rotation_angle < 0:
            rotation_angle += 360
        print(int(real_center_x*1.25), int(real_center_y*1.25), int(scaled_obj_height*1.25), int(rotation_angle))
        
    else:
        print("0 0 0 0")

# find for all images (1.png, 2.png, ... 9.png)
# for i in range(1, 15):
#     print(f"Processing {i}.png:", end=" ")
#     find(f"{i}.png")
filename = input()
find(filename)
