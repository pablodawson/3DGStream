import argparse
import os
import ffmpeg
import tqdm

parser = argparse.ArgumentParser(description='Preprocess equirect videos')

parser.add_argument('--source', '-s', type=str, required=True, help='Path to input video')
parser.add_argument('--output', '-o', type=str, required=False, help='Path to output video')
parser.add_argument('--start_time', type=str, default="00:00", help='Start time for trimming')
parser.add_argument('--end_time', type=str, default="00:00", help='End time for trimming')

args = parser.parse_args()


if args.output is None:
    args.output = os.path.join(args.source, 'frames2')
    os.makedirs(args.output, exist_ok=True)

input_path = args.source

videos = os.listdir(input_path)
videos = [video for video in videos if video.endswith('.mp4')]

for i, video in tqdm.tqdm(enumerate(videos), total=len(videos)):

    if args.start_time == "00:00" and args.end_time == "00:00":
        input_video = ffmpeg.input(os.path.join(input_path, video))
    else:
        input_video = ffmpeg.input(os.path.join(input_path, video), ss=args.start_time)

    input_split = input_video.split()

    fov = 90
    face1 = input_split[0].filter("v360", "equirect", output="rectilinear", 
                        v_fov=fov, h_fov=fov, w='1100', h='1100', yaw=00, pitch=00)
    face2 = input_split[0].filter("v360", "equirect", output="rectilinear", 
                        v_fov=fov, h_fov=fov, w='1100', h='1100', yaw=180, pitch=00)

    all_faces = [face1, face2]

    for j, face in enumerate(all_faces):
        index = len(all_faces)*i + j

        output_folder = os.path.join(args.output, f"cam{index}")

        os.makedirs(output_folder, exist_ok=True)
        face.output(os.path.join(output_folder, '%04d.jpg'), qscale=2, loglevel='error').run()

