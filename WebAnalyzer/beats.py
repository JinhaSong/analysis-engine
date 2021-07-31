import os, shutil, datetime
from AnalysisEngine.settings import MEDIA_ROOT
from AnalysisEngine.celerys import app
from WebAnalyzer import models
from utils import Logging


@app.task
def delete_old_database(days=0):
    if not os.path.exists(MEDIA_ROOT):
        return 0

    date_today = datetime.date.today()
    date_delta = datetime.timedelta(days)
    date_point = date_today - date_delta

    # Delete DB
    old_imagemodel_database = models.ImageModel.objects.filter(uploaded_date__lte=date_point)
    old_imagemodel_database_count = old_imagemodel_database.count()
    old_imagemodel_database.delete()

    old_videomodel_database = models.VideoModel.objects.filter(uploaded_date__lte=date_point)
    old_videomodel_database_count = old_imagemodel_database.count()
    old_videomodel_database.delete()

    # Delete Image Folder
    date_point_dir = str(filter(str.isdigit, date_point.isoformat()))
    for old_image_dir in os.listdir(MEDIA_ROOT):
        if old_image_dir < date_point_dir:
            shutil.rmtree(os.path.join(MEDIA_ROOT, old_image_dir))

    print(Logging.i("===================="))
    print(Logging.s(" Delete Old Image"))
    print(Logging.s(" - Date Point: {0}".format(date_point)))
    print(Logging.s("===================="))

    return old_imagemodel_database_count + old_videomodel_database_count