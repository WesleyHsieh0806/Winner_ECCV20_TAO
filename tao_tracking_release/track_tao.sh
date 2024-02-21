# modify this part
DETECTION_DIR="/data3/chengyeh/TAO-experiments/AOA/15fps/released_implementation"
ANNOTATION="/compute/trinity-1-38/chengyeh/TAO/annotations/validation_AOA_15fps.json"
IMAGE_DIR="/compute/trinity-1-38/chengyeh/TAO/frames"
OUTPUT_DIR="/data3/chengyeh/TAO-experiments/AOA/15fps/released_implementation"

# avoid modifying this part
FINAL_TRACK_OUTPUT_DIR="${OUTPUT_DIR}/final_track_results"
REID1_OUTPUT_DIR="${OUTPUT_DIR}/ReID1"
REID2_OUTPUT_DIR="${OUTPUT_DIR}/ReID2"
DET_TXT="det.txt"

module load cuda-10.1
module load cudnn-10.1-v7.6.5.32

# Generate detection files from the output of the detection model
python gen_detections.py \
    --detections-dir $DETECTION_DIR \
    --annotations $ANNOTATION \
    --output-dir $OUTPUT_DIR \
    --output-file $DET_TXT \
    --nms-thresh 0.5 \
    --det-num 300 \
    --workers 8

# Extract reid features
python generate_detection_features.py \
    --detections-dir $OUTPUT_DIR \
    --annotations $ANNOTATION \
    --output-dir $REID1_OUTPUT_DIR \
    --image-dir $IMAGE_DIR \
    --detection-file $DET_TXT \
    --model-file reid_pytorch/reid1.onnx \
    --gpus 0 1 2 3 4 5 6 7

python generate_detection_features.py \
    --detections-dir $OUTPUT_DIR \
    --annotations $ANNOTATION \
    --output-dir $REID2_OUTPUT_DIR \
    --image-dir $IMAGE_DIR \
    --detection-file $DET_TXT \
    --model-file reid_pytorch/reid2.onnx \
    --gpus 0 1 2 3 4 5 6 7

# # Tracking with modified DeepSORT
python track_tao.py \
    --detections-dir $REID1_OUTPUT_DIR \
    --annotations $ANNOTATION \
    --output-dir $REID1_OUTPUT_DIR \
    --save-feature \
    --workers 8

python track_tao.py \
    --detections-dir $REID2_OUTPUT_DIR \
    --annotations $ANNOTATION \
    --output-dir $REID2_OUTPUT_DIR \
    --save-feature \
    --workers 8

# get final tracking json
python tao_post_processing/onekey_postprocessing.py \
    --onnx_results1 $REID1_OUTPUT_DIR \
    --onnx_results2 $REID2_OUTPUT_DIR \
    --annotations $ANNOTATION \
    --output-dir $FINAL_TRACK_OUTPUT_DIR \
    --workers 8