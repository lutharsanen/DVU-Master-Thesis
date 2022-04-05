from __future__ import print_function
from vision.visual_preprocessing.utilis import mkdir_ifmiss
from vision.visual_preprocessing.utilis.package import *
import os
import cv2

from vision.visual_preprocessing.shotdetect.video_manager import VideoManager
from vision.visual_preprocessing.shotdetect.shot_manager import ShotManager
# For caching detection metrics and saving/loading to a stats file
from vision.visual_preprocessing.shotdetect.stats_manager import StatsManager

# For content-aware shot detection:
from vision.visual_preprocessing.shotdetect.detectors.content_detector_hsv_luv import ContentDetectorHSVLUV

from vision.visual_preprocessing.shotdetect.video_splitter import is_ffmpeg_available,split_video_ffmpeg
from vision.visual_preprocessing.shotdetect.keyf_img_saver import generate_images,generate_images_txt

def create_keyframes(video_path, data_root, video_prefix, split_video = True):
    
    #print(video_prefix)
    stats_file_folder_path = osp.join(data_root, "shot_stats")
    mkdir_ifmiss(stats_file_folder_path)

    stats_file_path = osp.join(stats_file_folder_path, '{}.csv'.format(video_prefix))
    video_manager = VideoManager([video_path])
    stats_manager = StatsManager()
    # Construct our shotManager and pass it our StatsManager.
    shot_manager = ShotManager(stats_manager)

    # Add ContentDetector algorithm (each detector's constructor
    # takes detector options, e.g. threshold).
    shot_manager.add_detector(ContentDetectorHSVLUV())
    base_timecode = video_manager.get_base_timecode()

    shot_list = []


    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    try:
        # If stats file exists, load it.
        if osp.exists(stats_file_path):
            # Read stats from CSV file opened in read mode:
            with open(stats_file_path, 'r') as stats_file:
                stats_manager.load_from_csv(stats_file, base_timecode)

        # Set begin and end time
        #if args.begin_time is not None:
        start_time = base_timecode + 0
        end_time = base_timecode + duration
        #end_time = base_timecode + 20
        video_manager.set_duration(start_time=start_time, end_time=end_time)
        video_manager.set_downscale_factor(1)

        # Start video_manager.
        video_manager.start()

        # Perform shot detection on video_manager.
        shot_manager.detect_shots(frame_source=video_manager)

        # Obtain list of detected shots.
        shot_list = shot_manager.get_shot_list(base_timecode)
        output_dir = osp.join(data_root, "shot_keyf", video_prefix)
        generate_images(video_manager, shot_list, output_dir)
        
        # Save keyf txt of frame ind
        #if args.save_keyf_txt:
        output_dir = osp.join(data_root, "shot_txt", "{}.txt".format(video_prefix))
        mkdir_ifmiss(osp.join(data_root, "shot_txt"))
        generate_images_txt(shot_list, output_dir)

        # Split video into shot video
        if split_video:
            output_dir = osp.join(data_root, "shot_split_video", video_prefix)
            split_video_ffmpeg([video_path], shot_list, output_dir, suppress_output=True)

        # We only write to the stats file if a save is required:
        if stats_manager.is_save_required():
            with open(stats_file_path, 'w') as stats_file:
                stats_manager.save_to_csv(stats_file, base_timecode)
    finally:
        video_manager.release()

