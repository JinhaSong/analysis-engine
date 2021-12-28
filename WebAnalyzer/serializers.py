from rest_framework import serializers
from WebAnalyzer.models import *


class MultiModalSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = Final
        fields = ( 'token', 'video', 'video_url', 'module_results_url','uploaded_date', 'updated_date', 'result')
        read_only_fields = ('token', 'uploaded_date', 'updated_date', 'result')