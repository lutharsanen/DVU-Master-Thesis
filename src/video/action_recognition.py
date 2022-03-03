import torch
import json

from pytorchvideo.models.hub.vision_transformers import mvit_base_32x3
from pytorchvideo.data.encoded_video import EncodedVideo
from typing import Dict
from video.transformer import transformator


def get_action_from_video(model,video_path, num_frames, mean, std, side_size, crop_size, alpha, sampling_rate, frames_per_second, data_loc):
    # Device on which to run the model
    # Set to cuda to load on GPU
    device = "cuda"

    #model = mvit_base_32x3(pretrained = True)

    # Set to eval mode and move to desired device
    model = model.to(device)
    model = model.eval()

    loc = f"{data_loc}/video/"
    with open(f"{loc}/kinetics_classnames.json", "r") as f:
        kinetics_classnames = json.load(f)

    # Create an id to label name mapping
    kinetics_id_to_classname = {}
    for k, v in kinetics_classnames.items():
        kinetics_id_to_classname[v] = str(k).replace('"', "")

    transform = transformator(num_frames, mean, std, side_size, crop_size, alpha)

    # The duration of the input clip is also specific to the model.
    clip_duration = (num_frames * sampling_rate)/frames_per_second

    # Select the duration of the clip to load by specifying the start and end duration
    # The start_sec should correspond to where the action occurs in the video
    start_sec = 0
    end_sec = start_sec + clip_duration

    # Initialize an EncodedVideo helper class
    video = EncodedVideo.from_path(video_path)

    # Load the desired clip
    video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

    # Apply a transform to normalize the video input
    video_data = transform(video_data)

    # Move the inputs to the desired device
    inputs = video_data["video"]
    inputs = [i.to(device)[None, ...] for i in inputs]

    # Pass the input clip through the model
    preds = model(inputs)

    # Get the predicted classes
    post_act = torch.nn.Softmax(dim=1)
    preds = post_act(preds)
    pred_classes = preds.topk(k=5).indices

    # Map the predicted classes to the label names
    pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
    return pred_class_names