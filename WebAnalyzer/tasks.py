# -*- coding: utf-8 -*-
from AnalysisEngine.config import DEBUG
from AnalysisEngine.celerys import app
from celery.signals import worker_init, worker_process_init
from billiard import current_process
from utils import Logging

@worker_init.connect
def model_load_info(**__):
    print(Logging.i("===================="))
    print(Logging.s("Worker Analyzer Initialize"))
    print(Logging.s("===================="))

@worker_process_init.connect
def module_load_init(**__):
    global analyzer

    if not DEBUG:
        worker_index = current_process().index
        print(Logging.i("====================\n"))
        print(Logging.s("Worker Id: {0}".format(worker_index)))
        print(Logging.s("===================="))

    # TODO:
    #   - Add your model
    #   - You can use worker_index if you need to get and set gpu_id
    #       - ex) gpu_id = worker_index % TOTAL_GPU_NUMBER
    from Modules.textrank.main import TextRank
    analyzer = TextRank()


@app.task
def analyzer_by_image(file_path):
    result = analyzer.inference_by_image(file_path)
    return result

@app.task
def analyzer_by_video(data, video_info, analysis_type):
    if analysis_type == 'video' :
        result = analyzer.inference_by_video(data, video_info)
    elif analysis_type == 'audio' :
        result = analyzer.inference_by_audio(data, video_info)
    elif analysis_type == 'text' :
        result = analyzer.inference_by_text(data, video_info)
    return result


# For development version
if DEBUG:
    print(Logging.i("===================="))
    print(Logging.s("Development"))
    print(Logging.s("===================="))
    module_load_init()
