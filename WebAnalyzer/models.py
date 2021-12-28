from django.db import models

# Create your models here.
from rest_framework import exceptions
from AnalysisEngine.config import DEBUG
from WebAnalyzer.tasks import analyzer_by_data
from WebAnalyzer.utils import filename
from django.db.models import JSONField

import ast


class Final(models.Model):
    token = models.AutoField(primary_key=True)
    video = models.FileField(upload_to=filename.default, null=True)
    video_url = models.TextField(null=True)
    module_results_url = models.TextField(null=False)
    uploaded_date = models.DateTimeField(auto_now_add=True)
    updated_date = models.DateTimeField(auto_now=True)
    result = JSONField(null=True)

    def save(self, *args, **kwargs):
        super(Final, self).save(*args, **kwargs)

        if DEBUG:
            task_get = ast.literal_eval(str(analyzer_by_data(self.module_results_url)))
        else:
            task_get = ast.literal_eval(str(analyzer_by_data.delay(self.module_results_url).get()))

        self.result = task_get
        super(Final, self).save()