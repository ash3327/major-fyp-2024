"""
With parallelization, faster, but may not work on your computer.
Use noparallel one if this script fails.
"""

import os
import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

import concurrent.futures

# --------------------------
# Mediapipe and YOLO Setup
# --------------------------
mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# YOLO models (update paths as needed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# yolo = YOLO("model/yolo11n.pt")
pose_model = YOLO("model/yolo11n-pose.pt")
pose_model.fuse()
pose_model.to(device)
hand_model = YOLO("model/best-3.pt")
hand_model.fuse()
hand_model.to(device)

# --------------------------
# Original Functions
# --------------------------
def get_yolo_pose(image_rgb):
    """
    Original YOLO function (using the pose model).
    """
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = pose_model(image, verbose=False)
    return results

def get_yolo_hand(image_rgb):
    """
    Get hand landmarks
    """
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results = hand_model(image, verbose=False)
    return results

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def extract_from_image(image, mode='RGB'):
    if mode == 'BGR':
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(image_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    image_lab_clahe = cv2.merge((l_clahe, a, b))
    image_rgb = cv2.cvtColor(image_lab_clahe, cv2.COLOR_LAB2RGB)
    return np.ascontiguousarray(image_rgb)

def extract_image(image_path):
    image = cv2.imread(image_path)
    return extract_from_image(image, mode='BGR')

def _extract_features_from_imgpath(image_path, dyn=False, hands=None):
    image_rgb = get_image(image_path)
    features, bbox, num_person = extract_features_from_clipped_region(image_rgb, dyn=dyn, hands=hands)
    return features

def _extract_features_from_img(image_rgb, hands, dyn=False, image_path=None):
    image_rgb2 = extract_from_image(image_rgb)
    
    hand_results = hands.process(image_rgb)
    if not hand_results.multi_hand_landmarks:
        hand_results = hands.process(image_rgb2)
    # pose_results = pose.process(image_rgb)
    # if not pose_results.pose_landmarks:
    #     pose_results = pose.process(image_rgb2)
    
    numpose = 0
    numhands = 0
    landmarks = dict(pose=[], hands=[])
    # if pose_results.pose_landmarks:
    #     for landmark in pose_results.pose_landmarks.landmark:
    #         landmarks['pose'].append([landmark.x, landmark.y, landmark.z])
    #     numpose = 1
    # else:
    #     landmarks['pose'] = [[0, 0, 0]] * 33

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            if numhands == 2:
                break
            hand_list = []
            for landmark in hand_landmarks.landmark:
                hand_list.append([landmark.x, landmark.y, landmark.z])
            landmarks['hands'].append(hand_list)
            numhands += 1
    
    # Ensure we have two hand entries
    if len(landmarks['hands']) < 2:
        landmarks['hands'].extend([[[0, 0, 0]] * 21] * (2 - len(landmarks['hands'])))
    
    landmarks['pose'] = np.array(landmarks['pose'])
    landmarks['hands'] = np.array(landmarks['hands'])
    return np.array((image_path, numpose, numhands, landmarks))

def extract_features_from_image(image_rgb, dyn=False, hands=None):
    if hands is None:
        with mp.solutions.hands.Hands(
            static_image_mode=not dyn,
            max_num_hands=2,
            min_detection_confidence=0.5,
            model_complexity=1
        ) as hands:
            return _extract_features_from_img(image_rgb, hands, dyn=dyn)
    else:
        return _extract_features_from_img(image_rgb, hands, dyn=dyn)

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def extract_features_from_video(video_path, output_dir, dataset, split, skip=False):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, dataset, split, f"{video_name}.npy")
    
    # print(f"{output_path} exists: {os.path.exists(output_path)} | skip={skip}")
    if skip and os.path.exists(output_path):
        print(f"Skipping {video_path}, features already extracted.")
        return
    
    frames = extract_frames_from_video(video_path)
    data = []
    
    def process_frame(idx, frame):
        features, _, _ = extract_features_from_clipped_region(frame, dyn=True)
        features[0] = f"{video_path}${idx}"
        data.append(features)
    
    for idx, frame in enumerate(frames):
        process_frame(idx, frame)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, np.array(data, dtype=object))
    print(f"Saved video features to {output_path}")

def extract_features_from_subfolder(data_dir, dataset, subfolder, output_dir, split, dyn=False, skip=False):
    main_dir = os.path.join(data_dir, dataset, subfolder)
    print('Extracting', main_dir, output_dir)
    
    data = []
    image_paths = []
    video_paths = []
    for root, _, files in os.walk(main_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_paths.append((root, os.path.join(root, file)))
            elif file.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                video_paths.append(os.path.join(root, file))
    num_images = len(image_paths)
    num_videos = len(video_paths)
    print(num_images, num_videos)

    def extract_features(image_path):
        _, num_bodies, num_hands, landmarks = _extract_features_from_imgpath(image_path, dyn=dyn)
        dpath = image_path.split(main_dir)[1].replace("\\", "/")
        dpath = dpath[1:] if dpath and dpath[0] == "/" else dpath
        return (dpath, num_bodies, num_hands, landmarks)
    
    if num_images != 0:
        pbar = tqdm(total=num_images, desc="Processing images")
        num_threads = min(os.cpu_count(), num_images, 16)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(extract_features, image_path) for label, image_path in image_paths]
            
            for future in concurrent.futures.as_completed(futures):
                data.append(future.result())
                pbar.update(1)
                
        pbar.close()
        print("Saving data to npy file...")
        data = np.array(data, dtype=object)
        output_dir = os.path.join(output_dir, dataset)
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"record_{split}.npy"), data)

    if num_videos != 0:
        print("Processing videos...")
        for video_path in tqdm(video_paths, desc="Processing videos"):
            extract_features_from_video(video_path, output_dir, dataset, split, skip=skip)
    
    print("Feature extraction complete!")

def extract_features(data_dir, dataset, subfolders, output_dir, dyn=False, *args, skip=False, **kwargs):
    for split, subfolder in subfolders.items():
        extract_features_from_subfolder(data_dir, dataset, subfolder, output_dir, split, dyn=dyn, skip=skip)

# --------------------------
# New Functions for YOLO Detection & Clipping
# --------------------------

def select_best_detections(yolo_results, image_shape, n=1):
    """
    Selects the best n detections based on bounding box area and confidence.
    Returns a list of tuples (bbox, pose_keypoints) where:
      - bbox is an array [xmin, ymin, xmax, ymax] corresponding to the selected detection,
      - pose_keypoints contains the YOLO-detected pose keypoints for that detection (if available),
        otherwise None.
    """
    detections = []
    conf_bound = 0.2

    for result in yolo_results:
        if result.boxes.xyxy.shape[0] > 0:
            for i in range(result.boxes.xyxy.shape[0]):
                box = result.boxes.xyxy[i]
                conf = result.boxes.conf[i]
                area = (box[2] - box[0]) * (box[3] - box[1])
                if conf >= conf_bound:
                    pose_keypoints = None
                    if hasattr(result, "keypoints") and result.keypoints.xy is not None:
                        try:
                            pose_keypoints = result.keypoints.xy[i].cpu().numpy()
                        except Exception as e:
                            pose_keypoints = None
                    detections.append((box, conf, area, pose_keypoints))

    # Sort detections by area and confidence
    detections.sort(key=lambda x: (x[2], x[1]), reverse=True)

    best_detections = []
    for i in range(min(n, len(detections))):
        box, _, _, pose_keypoints = detections[i]
        box = torch.round(box).int().cpu().numpy()
        # Increase box size by a scaling factor (e.g., 1.3)
        scale = 1.3
        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        size = [box[2] - box[0], box[3] - box[1]]
        xmin = max(center[0] - size[0] * scale / 2, 0)
        ymin = max(center[1] - size[1] * scale / 2, 0)
        xmax = min(center[0] + size[0] * scale / 2, image_shape[1])
        ymax = min(center[1] + size[1] * scale / 2, image_shape[0])
        bbox = np.array([xmin, ymin, xmax, ymax]).astype(int)
        if pose_keypoints is not None:
            pose_keypoints[:, 0] /= image_shape[1]
            pose_keypoints[:, 1] /= image_shape[0]
        best_detections.append((bbox, pose_keypoints))

    return best_detections, len(detections)

def crop_image(image, bbox):
    """
    Crops the image using the given bounding box.
    """
    xmin, ymin, xmax, ymax = bbox
    return image[ymin:ymax, xmin:xmax]

def transform_landmarks(landmarks, bbox, image_shape, normalize=True):
    """
    Transforms landmark coordinates from the clipped image back to the original image space.
    """
    if 'hands' not in landmarks:
        return landmarks
    xmin, ymin, xmax, ymax = bbox
    lmkshape = landmarks['hands'].shape[2]
    scale = [xmax - xmin, ymax - ymin, 1][:lmkshape]
    offset = [xmin, ymin, 0][:lmkshape]
    screen_size = [image_shape[1], image_shape[0], 1][:lmkshape]
    if not normalize:
        screen_size = np.array(screen_size)# * scale
    # if 'pose' in landmarks:
        # landmarks['pose'] = np.where(landmarks['pose'] == 0, 0, (landmarks['pose'] * scale + offset) / screen_size)
    # if 'hands' in landmarks:
    landmarks['hands'] = np.where(landmarks['hands'] == 0, 0, (landmarks['hands'] * scale + offset) / screen_size)
    return landmarks

def postprocess_landmarks(features):
    # if (np.min(landmarks['pose'],axis=0)[:2] == 0).all():
    #     return landmarks
    # if (landmarks['hands'][0]-landmarks['hands'][1])[:2]:
        # landmarks['hands'][1] = np.zeros((3,21))
    # print(landmarks['hands'].shape)
    # print(np.max(landmarks['hands'],axis=(0,1))-np.min(landmarks['hands'],axis=(0,1)))
    # print(np.max(landmarks['pose'],axis=0)-np.min(landmarks['pose'],axis=0))
    body = features[3]['pose']
    if features[2] == 2 and (body[9:11] != 0).all():
        if np.linalg.norm(body[9]-features[3]['hands'][0,0,:2]) > np.linalg.norm(body[9]-features[3]['hands'][0,1,:2]):
            features[3]['hands'][0],features[3]['hands'][1] = np.copy(features[3]['hands'][1]),np.copy(features[3]['hands'][0])
    return features[3]

def extract_features_from_clipped_region(image_rgb, dyn=False, hands=None):
    """
    Extracts features from the region of the image defined by the best YOLO detection.
    1. Runs YOLO to detect regions.
    2. Selects the best detection (largest area & highest confidence).
    3. Crops the image to that region.
    4. Extracts features from the clipped region.
    5. Transforms the landmark coordinates back to the original image space.
    Returns a tuple (features, bbox).
    """
    yolo_results = get_yolo_pose(image_rgb)
    res, num_person = select_best_detections(yolo_results, image_rgb.shape)
    bbox, yolo_pose = res[0] if len(res) > 0 else (None, None)
    # print('*',bbox,res)
    if bbox is not None:
        # print('bbox',bbox)
        clipped_img = crop_image(image_rgb, bbox)
    else:
        clipped_img = image_rgb  # Fallback if no detection is found
    features_prim = extract_features_from_image(image_rgb, dyn=dyn, hands=hands)
    features = extract_features_from_image(clipped_img, dyn=dyn, hands=hands)
    if features_prim[2] > features[2]:
        features = features_prim
        clipped_img = image_rgb
        bbox = None
    if yolo_pose is not None:
        # yolo_pose: [num_points=17, 2]
        features[3]['pose'] = yolo_pose # problems
        features[1] = 1
    else:
        features[3]['pose'] = np.zeros((17,2))
        features[1] = 0
    stor = features[2], np.copy(features[3]['hands'])
    if features[2] < 2:
        yolo_hand_results = get_yolo_hand(clipped_img)
        res, num_hands = select_best_detections(yolo_hand_results, image_rgb.shape, n=2)
        features[2] = 0
        fcount = 0
        for i, (bbox_hand, yolo_hand) in enumerate(res):
            if i == 2:
                break
            hand_img = crop_image(clipped_img, bbox_hand)
            features_hand = extract_features_from_image(hand_img)
            if features_hand[2] >= 1:
                features_hand[3] = transform_landmarks(features_hand[3], bbox_hand, clipped_img.shape)
                features[3]['hands'][i] = features_hand[3]['hands'][0]
                features[2] += 1
                fcount += 1
        if features[2] < stor[0]:
            #features[2]
            features[2], features[3]['hands'] = stor
        elif features[2] == 0:
            for i, (bbox_hand, yolo_hand) in enumerate(res):
                if i == 2:
                    break
                features[3]['hands'] = features[3]['hands'].astype(float)
                # yolo_hand_transformed = dict(hands=np.expand_dims(, axis=0)) # need to scale by size
                # yolo_hand_transformed = transform_landmarks(yolo_hand_transformed, bbox_hand, clipped_img.shape)
                chei, cwid, _ = clipped_img.shape
                ihei, iwid, _ = image_rgb.shape
                features[3]['hands'][i][:,:2] = yolo_hand*[iwid,ihei]/[cwid,chei]
                features[2] += 1
                fcount += 1
                # print('\t',i,yolo_hand.dtype,features[3]['hands'][i].dtype)
                # print(min(yolo_hand[:,0]),max(yolo_hand[:,0]),min(yolo_hand[:,1]),max(yolo_hand[:,1]),cwid,chei,image_rgb.shape)
            # print(features[3]['hands'],res)
            # yolo_hand_results[0].show() ## for testing only
            # yolo = correct-bbox_corner (bbox corner)
            # transform(yolo) = (yolo*bbox_wh+bbox_corner)/image_wh 
            # = ((correct-bbox_corner)*bbox_wh+bbox_corner)/image_wh
            # % (([correct-bbox_corner]/bbox_wh*image_wh)*bbox_wh+bbox_corner)/image_wh
    if bbox is not None:
        features[3] = transform_landmarks(features[3], bbox, image_rgb.shape)
    features[3] = postprocess_landmarks(features)
    features[2] = sum(np.any(hand != 0) for hand in features[3]['hands'])
    return features, bbox, num_person
