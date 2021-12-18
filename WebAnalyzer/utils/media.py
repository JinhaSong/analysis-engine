import time

from AnalysisEngine import settings

import os, datetime
import subprocess
import cv2
from datetime import timedelta

from utils import Logging


def get_directory():
    date_today = datetime.date.today()
    directory = date_today.strftime("%Y%m%d")
    return directory


def get_timestamp():
    date_now = datetime.datetime.now()
    timestamp = date_now.strftime("%H%M%S")
    return timestamp


def get_filename(path):
    return str(path).split("/")[-1].split(".")[0]


def get_video_dir_path(video_url):
    date_dir_path = os.path.join(settings.MEDIA_ROOT, get_directory())
    if not os.path.exists(date_dir_path):
        os.mkdir(date_dir_path)

    if "http" in video_url:
        dir_path = os.path.join(settings.MEDIA_ROOT, get_directory(), str(video_url).split("/")[-1]).split(".")[0]
        url = os.path.join(get_directory(), str(video_url).split("/")[-1]).split(".")[0]
    else :
        dir_path = video_url.split(".")[0]
        url = dir_path.replace(settings.MEDIA_ROOT, "")

    if not os.path.exists(dir_path) :
        os.mkdir(dir_path)
    else :
        timestamp = get_timestamp()
        dir_path = dir_path + "_" + timestamp
        url = dir_path.replace(settings.MEDIA_ROOT, "")

        os.mkdir(dir_path)

    return dir_path, url


def get_audio_filename(filename, ext):
    date_dir_path = os.path.join(settings.MEDIA_ROOT, get_directory())

    if not os.path.exists(date_dir_path):
        os.mkdir(date_dir_path)

    timestamp = get_timestamp()
    audio_sub_names = ["aed", "asc", "asr"]
    paths = []
    urls = []
    sub_dirs = []
    file_lists = []
    video_dir = os.path.join(settings.MEDIA_ROOT, get_directory(), filename)
    if not os.path.exists(video_dir):
        os.mkdir(video_dir)
    for sub_name in audio_sub_names:
        path = os.path.join(video_dir, filename + "_" + sub_name + ext)
        url = os.path.join(get_directory(), filename + "_" + sub_name + ext)
        sub_dir = os.path.join(video_dir, sub_name)
        file_list_path = os.path.join(video_dir, sub_name + '-files.txt')
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        paths.append(path)
        urls.append(url)
        sub_dirs.append(sub_dir)
        file_lists.append(file_list_path)

    return paths, urls, sub_dirs, file_lists

def extract_audio(video_url, start_time, end_time):
    video_name = get_filename(video_url)
    paths, urls, sub_dirs, file_lists = get_audio_filename(video_name, ".wav")

    if end_time == "00:00:00.00":
        ffmpeg_commands = [
            'ffmpeg -loglevel 8 -y -i {} -acodec pcm_s16le -ac 1 -ar 16000 {}'.format(video_url, paths[1]),
        ]
    else:
        ffmpeg_commands = [
            "ffmpeg -loglevel 8 -y -i {} -ss {} -to {} -acodec pcm_s16le -ac 1 -ar 16000 {}".format(
                video_url,
                start_time,
                end_time,
                paths[0]), # aed(audio event detection)
        ]
    sox_commands = [
        "sox --i {}".format(paths[0]),
    ]

    for ffmpeg, sox in zip(ffmpeg_commands, sox_commands) :
        start_time = time.time()
        print(Logging.i("Start extraction wav file from video"))
        os.system(ffmpeg)
        os.system(sox)
        end_time = time.time()
        print(Logging.i("Ended extraction(time: {})".format(end_time - start_time)))

    return paths, sub_dirs, file_lists


def extract_frames(video_url, extract_fps):
    frame_dir_path, url = get_video_dir_path(video_url)

    command = "ffmpeg -y -hide_banner -loglevel panic -i {} -vsync 2 -q:v 0 -vf fps={} {}/%05d.jpg".format(video_url, extract_fps, frame_dir_path)
    os.system(command)

    framecount = len(os.listdir(frame_dir_path))
    frame_url_list = []
    frame_path_list = []
    for frame_num in range(1, framecount + 1):
        path = settings.MEDIA_ROOT + os.path.join(url, "{0:05d}.{1}".format(frame_num,"jpg"))
        frame_url_list.append(os.path.join(url, str(frame_num) + ".jpg"))
        frame_path_list.append(path)

    return frame_path_list, frame_url_list


def get_video_metadata(video_path):
    ffprobe_command = ['ffprobe', '-show_format', '-pretty', '-loglevel', 'quiet', video_path]

    ffprobe_process = subprocess.Popen(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    metadata, error = ffprobe_process.communicate()
    metadata = metadata.decode("utf-8")

    infos = metadata.split("\n")
    json_metadata = {}
    for info in infos:
        if "=" in info:
            info = info.split("=")
            key = info[0]
            value = info[1]
            json_metadata[key] = value
    video_capture = cv2.VideoCapture(video_path)
    json_metadata['extract_fps'] = round(video_capture.get(cv2.CAP_PROP_FPS))
    video_capture.release()

    return json_metadata

def frames_to_timecode (frames, fps):
    td = timedelta(seconds=(frames / fps))
    return str(td)
