# Progress

## Data Preprocessing and Cleaning

1) Previously there is a very high rate of detection failures in most datasets, this update attempts to solve this problem.
     - Examples:
       - Missing hand (enhanced with YOLO clipping)
       - Wrongly detected full-body gesture over single-hand images (enhanced with YOLO-pose detection)

### Methods

1. **Mediapipe for Pose and Hands:**
   - Used Mediapipe's pose and hands detection models on raw images.
   - Output format: `[p0, p1, p2, p3, p4]`
     - `p0`: Total problematic items (any detection issues).
     - `p1`: Only wrong body count.
     - `p2`: Only wrong hand count.
     - `p3`: Both body and hand counts wrong.
     - `p4`: Detections missing (extra detection cases are not counted)

2. **CLAHE for Contrast Improvement:**
   - Applied Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast before detection with Mediapipe.

3. **Mixed Approach:**
   - Attempted detection on raw images with Mediapipe first.
   - If detection failed, applied CLAHE and retried with Mediapipe.

4. **YOLO for Pose and Hands:**
   - Used YOLO-pose for body detection and Mediapipe-hands or YOLO-hand for hand detection.
   - Fine-tuned YOLO for hand landmarks detection (100 epochs, results in `runs\train\hand3\weights\best.pt`).

5. **YOLO-Hierarchical:**
   - Used YOLO-pose to detect and clip the largest human region.
   - Applied Mediapipe-hands on the clipped image.
   - If unsuccessful, used fine-tuned YOLO-hand to detect hand bounding boxes, followed by Mediapipe-hands, selecting the best result.

### Results

The table below summarizes the problematic detection ratios for each method across the datasets:

Note: The notation (Name)<sup>#</sup> indicates that the value has been adjusted as it was previously measured incorrectly by not removing the blank class from the "missing detections" count.

| Dataset    | Split | Raw (%) | CLAHE (%) | Mixed (%) | YOLO (%) | YOLO-Hierarchical (%) | YOLO-H-v2 (%) |
|------------|-------|---------|-----------|-----------|----------|-----------------------|-|
| Lexset     | Train | 10.2<sup>#</sup> | 20.3<sup>#</sup> | 12.1<sup>#</sup> | **5.47**<sup>#</sup>  | *12.4*<sup>#</sup> | **7.74**
| Lexset     | Test  | 14.7<sup>#</sup> | 20.4<sup>#</sup> | 12.6<sup>#</sup> | **4.89**<sup>#</sup> | *12.7*<sup>#</sup> | **7.70**
| Hands      | -     | 35.6    | 35.6      | -         | 9.98     | **8.40** | 9.96 |
| Senz3d     | -     | 2.30    | 2.30      | -         | 0.23     | **0.00** | **0.00** |

**Notes:**
- **Lexset:** Significant improvement with YOLO (9.2% problematic ratio in train split).
- **Hands:** YOLO-Hierarchical achieved the lowest problematic ratio (8.4%) by solving the problem of not able to detect hands with the person only occupying a very small portion of the image, but occlusion is still an issue.
  - Image showcasing the problem: 
    - ![alt text](image-16.png)
    - ![alt text](image-17.png)
- **Senz3d:** YOLO-hierachical solved failures in detection.
  - Multi-detection (more than one hands are detected):
    - ![alt text](<Pasted image 20250313211552.png>)

## Current Progress and Todos

### Todos

1. **Revisit Datasets:**
   - Perform raw detection of body and hands using Mediapipe and visualize results.
   - **Hand Detection:**
     - Address Mediapipe issues by fine-tuning YOLOv11 over hand keypoints (planned for 100 epochs).
   - **Body Detection:**
     - Use YOLOv11 to improve Mediapipe performance.

2. **Revisit Models:**
   - [Further details to be defined as progress continues.]

3. **Occlusion Handling:**
   - Analyze datasets with full upper body versus single-hand scenarios.
   - Explore training with occlusion to enhance model robustness.
   - Goal: Ensure the model adapts to new tasks seamlessly.

### Tests and Commands

Use the following commands to test and validate the pipeline:

- **Test Feature Extractor:**
  ```python
  python test/test_feature_extractor.py -d <dataset_name> -s <split_name>
  ```
  - Omit `-s` to be prompted for the split name (available options provided).

- **Prepare Dataset:**
  ```python
  python prepare_dataset.py <dataset_name>
  ```

- **Verify Failed Detections:**
  - After feature extraction (saved under `data/kpts/`):
    ```python
    python dataset_loader.py -d <dataset_name> -s <split_name>
    ```

## Additional Notes

- **Inference Time:** We are evaluating YOLO's inference time, prioritizing the person with the highest hand detection confidence and largest size (not just size alone).
- **Metrics:** Model performance is assessed using mAP50 and mAP50-90.
- **Fine-Tuning:** YOLO fine-tuning for hand detection completed (100 epochs, 9.916 hours), with weights saved at `runs\train\hand3\weights\best.pt`.
- **Occlusion Challenges:** Notable in Senz3d, where YOLO struggles with overlapping keypoints—further investigation needed.
- **Timeline:**
  - 8/3: Data preprocessing and cleaning completed.
  - 9/3: Contrastive pre-training initiated.
  - 14/4: Report deadline.