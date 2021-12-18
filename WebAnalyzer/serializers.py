# -*- coding: utf-8 -*-
from rest_framework import serializers
from WebAnalyzer.models import *


class ImageSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = ImageModel
        fields = ('image', 'token', 'uploaded_date', 'updated_date', 'result')
        read_only_fields = ('token', 'uploaded_date', 'updated_date', 'result')


class VideoSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = VideoModel
        fields = ('token', 'video', 'uploaded_date', 'updated_date', 'video_url', 'video_info', 'video_text', 'extract_fps',
                  'start_time', 'end_time', 'analysis_type', 'result')
        read_only_fields = ('video_info', 'token', 'uploaded_date', 'updated_date', 'result')


class AudioSerializer(serializers.HyperlinkedModelSerializer):

    class Meta:
        model = AudioModel
        fields = ('video', 'audio', 'token', 'uploaded_date', 'updated_date')
        read_only_fields = ('video', 'token', 'uploaded_date', 'updated_date')
