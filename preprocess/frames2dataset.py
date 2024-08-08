# MIT License

# Copyright (c) 2023 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os 
import glob 
import tqdm 
import shutil
import sys 
import argparse
sys.path.append(".")
sys.path.append("Marigold")

#sys.path.append("deep-image-matching")

from pre_colmap import * 
from process_frame import getColmapFrame, undistortFrame
import ffmpeg
import imutils
import cv2
import torch
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def extractframes(videopath, startframe=0, endframe=60, w=-1, output_path="dataset"):
    video = ffmpeg.input(videopath, ss="00:40", t="10")

    if w == -1:
        resize = video
    else:
        resize = video.filter('scale', w, -1)
    
    outpath = os.path.join(output_path, os.path.basename(videopath).replace(".MOV", ""))
    os.makedirs(outpath, exist_ok=True)

    resize.output(os.path.join(outpath,"%d.png")).run()

def estimate_depth(offset, output_path="dataset", pipe=None):

    savedir = os.path.join(output_path, "colmap_" + str(offset))
    depth_savedir = os.path.join(savedir, "depth")
    os.makedirs(depth_savedir, exist_ok=True)

    # predict depth
    images = os.listdir(os.path.join(output_path, "colmap_" + str(offset), "images"))

    for cam in images:
        input_img = Image.open(os.path.join(output_path, "colmap_" + str(offset), "images", cam))
        pipeline_output = pipe(input_img, denoising_steps=3, ensemble_size=2, show_progress_bar=False)
        depth_out = pipeline_output.depth_np
        depth_saveable = (depth_out * (2 **16-1)).astype("uint16")
        Image.fromarray(depth_saveable).save(os.path.join(depth_savedir, cam))
    

def preparecolmapfolders(offset=0, extension=".png", output_path="dataset"):
    savedir = os.path.join(output_path, "colmap_" + str(offset))
    os.makedirs(savedir, exist_ok=True)
    input_savedir = os.path.join(savedir, "input")
    os.makedirs(input_savedir, exist_ok=True)
    
    cameras = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]        

    for cam in cameras:
        if "colmap" in cam:
            continue
        imagepath = os.path.join(output_path, cam, str(offset) + extension)
        output_path = output_path.replace("\\", "/")
        imagesavepath = os.path.join(input_savedir, cam + extension)

        shutil.copy(imagepath, imagesavepath)

def copy_frames(folder, elements, output_path):
    for element in sorted(elements):
        frames = glob.glob(os.path.join(folder, element, "*.png"))
        frames += glob.glob(os.path.join(folder, element, "*.jpg"))

        for i, frame in enumerate(sorted(frames)):
            dest_name = os.path.join(folder, output_path, element, str(i) + frame[-4:])
            out_folder = os.path.join(folder, output_path, element)
            os.makedirs(out_folder, exist_ok=True)
            if frame == dest_name:
                continue
            shutil.copy(frame, dest_name)

def resize_frames(folder, elements, w=-1, endframe=60):
    for element in sorted(elements):
        print(f"resizing frames in {element}")
        frames = glob.glob(os.path.join(folder, element, "*.png"))
        frames += glob.glob(os.path.join(folder, element, "*.jpg"))

        for i, frame in enumerate(sorted(frames)):
            if i>endframe:
                break

            folder_out = os.path.join(folder, output_path, element)
            os.makedirs(folder_out, exist_ok=True)
            newname = os.path.join(folder_out, str(i) + frame[-4:])
            
            if frame == newname:
                continue
            
            img = imutils.resize(cv2.imread(frame), width=w)
            cv2.imwrite(newname, img)

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", default="", type=str)
    parser.add_argument("--output", "-o", default="dataset", type=str)
    parser.add_argument("--startframe", default=1, type=int)
    parser.add_argument("--endframe", default=60, type=int)
    parser.add_argument("--colmap_path", default="colmap", type=str)
    parser.add_argument("--resize_width", default=-1, type=int)
    parser.add_argument("--depth", action= "store_true", default=False)
    parser.add_argument("--skip_to", default=0, type=int)
    parser.add_argument("--max_cameras", default=-1, type=int)
    parser.add_argument("--hard_reset", action= "store_true", default=False)
    parser.add_argument("--feature_matcher", default="sift", type=str)
    parser.add_argument("--per_frame_points", default=False, type=bool)
    
    args = parser.parse_args()
    videopath = args.source

    startframe = args.startframe
    endframe = args.endframe

    pipe = None        

    if startframe >= endframe:
        print("Start frame must smaller than end frame")
        quit()
    if not os.path.exists(videopath):
        print("Input path does not exist")
        quit()
    
    if args.hard_reset:
        shutil.rmtree(args.output, ignore_errors=True)

    # 0- Prepare frames
    videoslist = glob.glob(os.path.join(videopath,"*.MOV"))
    extension = ".png"

    # Create an output path in the same folder if not specified
    if not os.path.exists(args.output):
        output_path = os.path.join(videopath, args.output)
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = args.output
    
    if len(videoslist) == 0:
        elements = [f for f in os.listdir(videopath) if os.path.isdir(os.path.join(videopath, f)) and "colmap_" not in f and "dataset" not in f]
        
        if ((args.max_cameras!=-1) and (len(elements) > args.max_cameras)):
            elements = elements[:args.max_cameras]

        images_sample = glob.glob(os.path.join(videopath, elements[0], "*.png"))
        images_sample += glob.glob(os.path.join(videopath, elements[0], "*.jpg"))
        if len(images_sample) == 0:
            print("No videos or images found")
            quit()
        
        if args.skip_to < 1:
            if args.resize_width == -1:
                print("found images, copying them to 0.png, 1.png, ...")
                copy_frames(videopath, elements, output_path)
            else:
                print("found images, resizing and exporting them to 0.png, 1.png, ...")
                resize_frames(videopath, elements, w=args.resize_width, endframe=endframe)
        extension = images_sample[0][-4:]

    elif args.skip_to < 1:
        for v in tqdm.tqdm(videoslist):
            print(f"start extracting {endframe-startframe} frames from videos")
            extractframes(v, startframe=startframe, endframe=endframe, w= args.resize_width, output_path=output_path)
            pass
        
    # 1- Create colmap folders for each frame, add images
    if args.skip_to < 2:
        print("start preparing colmap image input")
        for offset in range(startframe, endframe):
            preparecolmapfolders(offset, extension=extension, output_path=output_path)
    
    # 2 - Run mapper on the first frame
    if args.skip_to < 3:
        getColmapFrame(output_path, startframe, colmap_path=args.colmap_path, manual=False, feature_matcher=args.feature_matcher)
    
    # 3- Run colmap per-frame, use the poses from first frame for all
    if args.skip_to < 4:
        for offset in range(startframe+1, endframe):
            if args.per_frame_points:
                getColmapFrame(output_path, offset, colmap_path=args.colmap_path, manual=True, startframe=startframe, feature_matcher=args.feature_matcher)
            else:
                undistortFrame(output_path, offset, colmap_path=args.colmap_path, startframe=startframe)
    # 4- Estimate depth on frames
    if args.depth and args.skip_to < 5:
        from marigold.marigold_pipeline import MarigoldPipeline
        pipe = MarigoldPipeline.from_pretrained(
            "prs-eth/marigold-lcm-v1-0",
            torch_dtype=torch.float16
        )
        pipe.to("cuda")

        for offset in tqdm.tqdm(range(startframe, endframe), total=endframe-startframe):
            estimate_depth(offset, output_path=output_path, pipe=pipe)
    
    print("done")