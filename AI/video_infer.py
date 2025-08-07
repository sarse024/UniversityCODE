import argparse
import ast
import math
import os
import sys
import time
import cv2
import numpy as np
import yaml
from datetime import datetime
import random
from collections import defaultdict

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.models import create_model
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh
from mindyolo.utils.utils import set_seed


def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="Video Infer", parents=[parents] if parents else [])
    parser.add_argument("--task", type=str, default="detect", choices=["detect"])
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--ms_mode", type=int, default=0, help="train mode, graph/pynative")
    parser.add_argument("--ms_amp_level", type=str, default="O0", help="amp level, O0/O1/O2")
    parser.add_argument(
        "--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="use enable_graph_kernel or not"
    )
    parser.add_argument(
        "--precision_mode", type=str, default=None, help="set accuracy mode of network model"
    )
    parser.add_argument("--weight", type=str, default="yolov7_300.ckpt", help="model.ckpt path(s)")
    parser.add_argument("--img_size", type=int, default=640, help="inference size (pixels)")
    parser.add_argument(
        "--single_cls", type=ast.literal_eval, default=False, help="train multi-class data as single-class"
    )
    parser.add_argument("--exec_nms", type=ast.literal_eval, default=True, help="whether to execute NMS or not")
    parser.add_argument("--nms_time_limit", type=float, default=60.0, help="time limit for NMS")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.65, help="IOU threshold for NMS")
    parser.add_argument(
        "--conf_free", type=ast.literal_eval, default=False, help="Whether the prediction result include conf"
    )
    parser.add_argument("--seed", type=int, default=2, help="set global seed")
    parser.add_argument("--log_level", type=str, default="INFO", help="save dir")
    parser.add_argument("--save_dir", type=str, default="./runs_infer", help="save dir")

    parser.add_argument("--video_path", type=str, required=True, help="path to input video")
    parser.add_argument("--output_path", type=str, default=None, help="path for output video")
    parser.add_argument("--show", type=ast.literal_eval, default=False, help="display video in real-time")
    parser.add_argument("--fps", type=int, default=None, help="output video FPS")
    parser.add_argument("--frame_skip", type=int, default=0, help="process every nth frame (0=all frames)")
    parser.add_argument("--min_confidence", type=float, default=0.5, help="minimum confidence to display results")
    parser.add_argument("--max_trajectory", type=int, default=0,
                        help="maximum points to store in trajectory (0=disable)")
    parser.add_argument("--show_counter", type=ast.literal_eval, default=True,
                        help="show object counter in top-right corner")

    return parser


def set_default_infer(args):
    # Set Context
    ms.set_context(mode=args.ms_mode)
    ms.set_recursion_limit(2000)
    if args.precision_mode is not None:
        ms.device_context.ascend.op_precision.precision_mode(args.precision_mode)
    if args.ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O2"})
    if args.device_target == "Ascend":
        ms.set_device("Ascend", int(os.getenv("DEVICE_ID", 0)))
    args.rank, args.rank_size = 0, 1
    # Set Data
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # number of classes
    args.data.names = ["item"] if args.single_cls and len(args.names) != 1 else args.data.names  # class names
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )
    # Directories and Save run settings
    platform = sys.platform
    if platform == "win32":
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    else:
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.rank % args.rank_size == 0:
        with open(os.path.join(args.save_dir, "cfg.yaml"), "w") as f:
            yaml.dump(vars(args), f, sort_keys=False)
    # Set Logger
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=args.rank, device_per_servers=args.rank_size)
    logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))


def detect(
        network: nn.Cell,
        img: np.ndarray,
        conf_thres: float = 0.25,
        iou_thres: float = 0.65,
        conf_free: bool = False,
        exec_nms: bool = True,
        nms_time_limit: float = 60.0,
        img_size: int = 640,
        stride: int = 32,
        num_class: int = 80,
        is_coco_dataset: bool = True,
):
    # Resize
    h_ori, w_ori = img.shape[:2]  # orig hw
    r = img_size / max(h_ori, w_ori)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        new_h, new_w = math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

    # Transpose Norm
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    imgs_tensor = Tensor(img[None], ms.float32)

    # Run infer
    _t = time.time()
    out, _ = network(imgs_tensor)  # inference and training outputs
    out = out[-1] if isinstance(out, (tuple, list)) else out
    infer_times = time.time() - _t

    # Run NMS
    t = time.time()
    out = out.asnumpy()
    out = non_max_suppression(
        out,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        conf_free=conf_free,
        multi_label=True,
        time_limit=nms_time_limit,
        need_nms=exec_nms,
    )
    nms_times = time.time() - t

    result_dict = {"category_id": [], "bbox": [], "score": []}
    total_category_ids, total_bboxes, total_scores = [], [], []
    for si, pred in enumerate(out):
        if len(pred) == 0:
            continue

        # Predictions
        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))  # native-space pred

        box = xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        category_ids, bboxes, scores = [], [], []
        for p, b in zip(pred.tolist(), box.tolist()):
            category_ids.append(COCO80_TO_COCO91_CLASS[int(p[5])] if is_coco_dataset else int(p[5]))
            bboxes.append([round(x, 3) for x in b])
            scores.append(round(p[4], 5))

        total_category_ids.extend(category_ids)
        total_bboxes.extend(bboxes)
        total_scores.extend(scores)

    result_dict["category_id"].extend(total_category_ids)
    result_dict["bbox"].extend(total_bboxes)
    result_dict["score"].extend(total_scores)

    # Removed detailed logging to reduce clutter
    return result_dict


def process_frame(args, network, frame, is_coco_dataset):
    """Process a single frame with the model"""
    return detect(
        network=network,
        img=frame,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        conf_free=args.conf_free,
        exec_nms=args.exec_nms,
        nms_time_limit=args.nms_time_limit,
        img_size=args.img_size,
        stride=max(max(args.network.stride), 32),
        num_class=args.data.nc,
        is_coco_dataset=is_coco_dataset,
    )


class ObjectTracker:
    def __init__(self, max_trajectory_length=0):
        self.next_id = 1
        self.tracked_objects = {}
        self.colors = {}
        self.trajectories = defaultdict(list)
        self.max_trajectory_length = max_trajectory_length
        self.velocity = defaultdict(float)  # Store velocity for each object
        self.last_position = {}  # Store last position for velocity calculation
        self.last_time = {}  # Store last detection time for each object

    def get_object_id(self, bbox, frame_width, frame_height, timestamp):
        """Assign a unique ID based on object position"""
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2

        # Normalize coordinates relative to frame dimensions
        norm_x = center_x / frame_width
        norm_y = center_y / frame_height

        # Calculate center point for trajectory
        center_point = (int(center_x), int(center_y))

        # Find closest existing object
        object_id = None
        min_distance = float('inf')

        for obj_id, (last_x, last_y, count) in self.tracked_objects.items():
            distance = math.sqrt((norm_x - last_x) ** 2 + (norm_y - last_y) ** 2)

            # If we find a close object (within 5% of frame size)
            if distance < 0.05:
                if distance < min_distance:
                    min_distance = distance
                    object_id = obj_id

                    # Update trajectory if enabled
                    if self.max_trajectory_length > 0:
                        self.trajectories[obj_id].append(center_point)
                        if len(self.trajectories[obj_id]) > self.max_trajectory_length:
                            self.trajectories[obj_id].pop(0)

                        # Calculate velocity (pixels per second)
                        if obj_id in self.last_position and obj_id in self.last_time:
                            prev_x, prev_y = self.last_position[obj_id]
                            time_diff = timestamp - self.last_time[obj_id]

                            if time_diff > 0:
                                distance = math.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                                self.velocity[obj_id] = distance / time_diff

                        # Update last position and time
                        self.last_position[obj_id] = (center_x, center_y)
                        self.last_time[obj_id] = timestamp

        if object_id is None:
            # New object
            object_id = self.next_id
            self.next_id += 1
            self.colors[object_id] = [random.randint(0, 255) for _ in range(3)]

            # Initialize trajectory if enabled
            if self.max_trajectory_length > 0:
                self.trajectories[object_id] = [center_point]
                self.last_position[object_id] = (center_x, center_y)
                self.last_time[object_id] = timestamp
                self.velocity[object_id] = 0.0

        # Update object position
        self.tracked_objects[object_id] = (norm_x, norm_y, 0)

        return object_id, self.colors[object_id], center_point, self.velocity.get(object_id, 0.0)

    def update(self):
        """Update tracked objects state"""
        # Increment counters for all objects
        for obj_id in list(self.tracked_objects.keys()):
            x, y, count = self.tracked_objects[obj_id]
            count += 1
            self.tracked_objects[obj_id] = (x, y, count)

            # Remove objects not seen for more than 5 frames
            if count > 5:
                del self.tracked_objects[obj_id]
                if obj_id in self.colors:
                    del self.colors[obj_id]
                if obj_id in self.trajectories:
                    del self.trajectories[obj_id]
                if obj_id in self.velocity:
                    del self.velocity[obj_id]
                if obj_id in self.last_position:
                    del self.last_position[obj_id]
                if obj_id in self.last_time:
                    del self.last_time[obj_id]


def draw_frame(frame, result_dict, class_names, min_confidence, max_trajectory, show_counter, is_coco_dataset=True):
    """Draw results on frame with unique IDs, consistent colors, trajectories, and counter"""
    im = frame.copy()
    category_id, bbox, score = result_dict["category_id"], result_dict["bbox"], result_dict["score"]

    # Initialize tracker
    if not hasattr(draw_frame, "tracker"):
        draw_frame.tracker = ObjectTracker(max_trajectory_length=max_trajectory)

    tracker = draw_frame.tracker
    frame_height, frame_width = im.shape[:2]
    current_time = time.time()  # Current timestamp for velocity calculation

    # For object counting in current frame
    class_counter = defaultdict(int)

    for i in range(len(bbox)):
        # Skip results below confidence threshold
        if score[i] < min_confidence:
            continue

        x_l, y_t, w, h = bbox[i][:]

        # Get unique ID, color, center, and velocity
        object_id, color, center, velocity = tracker.get_object_id(
            bbox[i], frame_width, frame_height, current_time
        )

        # Get class name
        if is_coco_dataset:
            class_name_index = COCO80_TO_COCO91_CLASS.index(category_id[i])
        else:
            class_name_index = category_id[i]
        class_name = class_names[class_name_index]

        # Update class counter
        class_counter[class_name] += 1

        # Draw center and trajectory if enabled
        if max_trajectory > 0:
            # Draw center point
            cv2.circle(im, center, 4, tuple(color), -1)

            # Draw trajectory
            if object_id in tracker.trajectories:
                trajectory = tracker.trajectories[object_id]
                for j in range(1, len(trajectory)):
                    cv2.line(im, trajectory[j - 1], trajectory[j], tuple(color), 2)

        # Draw bounding box
        x_r, y_b = x_l + w, y_t + h
        x_l, y_t, x_r, y_b = int(x_l), int(y_t), int(x_r), int(y_b)
        cv2.rectangle(im, (x_l, y_t), (x_r, y_b), tuple(color), 2)

        # Draw label with ID
        text = f"{object_id}: {class_name} {score[i]*100:.1f}%"

        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Draw label background
        cv2.rectangle(im, (x_l, y_t - text_h - baseline), (x_l + text_w, y_t), tuple(color), -1)
        cv2.putText(im, text, (x_l, y_t - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Update tracker state
    tracker.update()

    # Add object counter in top-right if enabled
    if show_counter:
        total_objects = sum(class_counter.values())
        display_text = f"Total objects: {total_objects}"

        # Add count per class
        for class_name, count in class_counter.items():
            display_text += f"\n{class_name}: {count}"

        # Position text (top-right)
        text_position = (frame_width - 300, 30)
        line_height = 30

        # Calculate background size
        text_lines = display_text.split('\n')
        max_width = 0
        for line in text_lines:
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            if w > max_width:
                max_width = w

        # Draw text background
        cv2.rectangle(im,
                      (frame_width - max_width - 20, 10),
                      (frame_width - 10, 20 + len(text_lines) * line_height),
                      (0, 0, 0), -1)

        # Draw multi-line text
        for i, line in enumerate(text_lines):
            y_position = text_position[1] + i * line_height
            cv2.putText(im, line, (frame_width - max_width - 10, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return im


def infer(args):
    # Init
    set_seed(args.seed)
    set_default_infer(args)

    # Create Network
    network = create_model(
        model_name=args.network.model_name,
        model_cfg=args.network,
        num_classes=args.data.nc,
        sync_bn=False,
        checkpoint_path=args.weight,
    )
    network.set_train(False)
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)

    # Setup video capture
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {args.video_path}")

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer
    output_fps = args.fps or orig_fps
    output_path = args.output_path or os.path.join(args.save_dir, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))

    is_coco_dataset = "coco" in args.data.dataset_name
    frame_count = 0
    processed_count = 0
    start_time = time.time()

    # Progress bar setup
    progress_interval = max(1, total_frames // 20)  # Update progress 20 times

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if requested
        if args.frame_skip > 0 and frame_count % (args.frame_skip + 1) != 0:
            frame_count += 1
            out.write(frame)  # Write unprocessed frame
            continue

        # Process frame
        result_dict = process_frame(args, network, frame, is_coco_dataset)

        # Draw results directly on the frame
        frame = draw_frame(frame, result_dict, args.data.names, args.min_confidence,
                           args.max_trajectory, args.show_counter, is_coco_dataset)

        # Write frame to output
        out.write(frame)

        # Display if requested
        if args.show:
            cv2.imshow('Output', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        processed_count += 1
        frame_count += 1

        # Show progress
        if processed_count % progress_interval == 0 or processed_count == total_frames:
            elapsed = time.time() - start_time
            fps = processed_count / elapsed if elapsed > 0 else 0
            progress = processed_count / total_frames
            bar_length = 30
            block = int(round(bar_length * progress))
            text = f"\rProgress: [{'=' * block}{' ' * (bar_length - block)}] {progress:.1%} | " \
                   f"FPS: {fps:.1f} | {processed_count}/{total_frames} frames"
            sys.stdout.write(text)
            sys.stdout.flush()

    # Cleanup
    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = processed_count / total_time if total_time > 0 else 0

    # Final message
    print("\n" + "=" * 50)
    print(f"Inference completed")
    print(f"Output saved to: {output_path}")
    print(f"Total frames processed: {processed_count}/{total_frames}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print("=" * 50)


if __name__ == "__main__":
    parser = get_parser_infer()
    args = parse_args(parser)
    infer(args)

## Example video
# python video_infer.py --config ./configs/yolov8/yolov8n.yaml --video_path videoInput/videoCani.mp4 --output_path ./videoOutput/outputCani.mp4 --weight ./yolov8-n_500e_mAP372-cc07f5bd.ckpt --task detect  --device_target CPU
# python video_infer.py --config ./configs/yolov8/yolov8n.yaml --video_path videoInput/videoPeople.mp4 --output_path ./videoOutput/outputPeople.mp4 --weight ./yolov8-n_500e_mAP372-cc07f5bd.ckpt --task detect  --device_target CPU --max_trajectory 100 --show True
# python video_infer.py --config ./configs/yolov8/yolov8n.yaml --video_path videoInput/videoCar.mp4 --output_path ./videoOutput/outputCar.mp4 --weight ./yolov8-n_500e_mAP372-cc07f5bd.ckpt --task detect  --device_target CPU --max_trajectory 100
