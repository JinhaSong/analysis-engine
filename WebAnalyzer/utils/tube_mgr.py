from pyyoutube import Api
import urllib.parse as urlparse

from utils import Logging


class YouTubeInfo:
    '''
    Google YouTube API v3
    admin = giseok.choe@gmail.com
    '''
    YouTube_API_Key = 'AIzaSyBmAAOjm1y5_A5oULSHOhsiYYo3cffLQKQ'

    def __init__(self):
        self.is_loaded = False
        self.api = Api(api_key=self.YouTube_API_Key)
        self.video = []

    @staticmethod
    def get_id_from_url(url):
        url_data = urlparse.urlparse(url)
        query = urlparse.parse_qs(url_data.query)
        if query:
            return query['v'][0]
        else:
            return False

    @staticmethod
    def is_youtube_url(url):
        try:
            YouTubeInfo.get_id_from_url(url)
        except Exception as e:
            return False

        return True

    def set_youtube_url(self, url):
        video_id = self.get_id_from_url(url)
        if video_id:
            self.video = self.api.get_video_by_id(video_id=video_id)
            if self.video:
                self.is_loaded = True
            else:
                print(Logging.e('YouTube Parser Error!'))

    def get_title(self, is_safe_string=False):
        if self.is_loaded:
            ret = self.video.items[0].snippet.to_dict()['title']
        else:
            ret = False

        if is_safe_string:
            ret.encode('utf-8', errors='replace').decode('utf-8')

        return ret

    def get_desc(self):
        if self.is_loaded:
            ret = self.video.items[0].snippet.to_dict()['description']
        else:
            ret = False
        return ret