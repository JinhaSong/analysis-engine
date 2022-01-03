import os, hashlib, datetime


def _get_directory():
    _date_today = datetime.date.today()
    _directory = _date_today.strftime("%Y%m%d")
    return _directory


def default(instance, filename):
    _path = os.path.join(_get_directory(), filename)
    return _path


def _get_image_directory():
    _date_now = datetime.datetime.now()
    _directory = _date_now.strftime("%Y%m%d%H%M%S")
    return _directory


def image_path(instance, filename):
    _path = os.path.join(_get_image_directory(), filename.split(".")[0], 'image', filename)
    return _path


def md5sum(instance, filename):
    _contents = instance.image.read()
    _base = hashlib.md5(bytes(_contents)).hexdigest()
    _ext = os.path.splitext(filename)[-1]
    _filename = "{0}{1}".format(_base, _ext)
    _path = os.path.join(_get_directory(), _filename)
    return _path


def sha256(instance, filename):
    _contents = instance.image.read()
    _base = hashlib.sha256(bytes(_contents)).hexdigest()
    _ext = os.path.splitext(filename)[-1]
    _filename = "{0}{1}".format(_base, _ext)
    _path = os.path.join(_get_directory(), _filename)
    return _path


def uploaded_date(instance, filename):
    _contents = datetime.datetime.now()
    _base = _contents.strftime("%H%M%S%f")
    _ext = os.path.splitext(filename)[-1]
    _filename = "{0}{1}".format(_base, _ext)
    _path = os.path.join(_get_directory(), _filename)
    return _path