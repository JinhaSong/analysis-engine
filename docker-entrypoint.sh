#!/usr/bin/env bash
set -e

cd /workspace
wget ftp://mldisk.sogang.ac.kr/hidf/scene_text_recognition/best_accuracy.pth -P /workspace/Modules/scenetext/weights
wget ftp://mldisk.sogang.ac.kr/hidf/scene_text_recognition/craft_mlt_25k.pth -P /workspace/Modules/scenetext/weights
wget ftp://mldisk.sogang.ac.kr/hidf/scene_text_recognition/craft_refiner_CTW1500.pth -P /workspace/Modules/scenetext/weights

sh run_migration.sh
python3 -c "import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'AnalysisEngine.settings'
import django
django.setup()
from django.contrib.auth.management.commands.createsuperuser import get_user_model
if not get_user_model().objects.filter(username='$DJANGO_SUPERUSER_USERNAME'):
    get_user_model()._default_manager.db_manager().create_superuser(username='$DJANGO_SUPERUSER_USERNAME', email='$DJANGO_SUPERUSER_EMAIL', password='$DJANGO_SUPERUSER_PASSWORD')"
sh server_start.sh

trap 'sh server_shutdown.sh' EXIT

tail -f celery.log -f django.log

exec "$@"