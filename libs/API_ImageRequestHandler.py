import os
import cv2
import mapillary.interface as mly
import requests
import numpy as np
from mapillary.models.exceptions import InvalidImageKeyError
import os, sys


class MapillaryImageFetcher:
    
    """
    Initialize the MapillaryImageFetcher class with specific access token and file URL to save the batch fetched
    """

    def __init__(self, access_token, save_folder='mapillary_images'):
        self.access_token = access_token
        self.save_folder = save_folder
        mly.set_access_token(access_token)
        self.mly_interface = mly
        os.makedirs(self.save_folder, exist_ok=True)


    
    """
    Fetch images process
    """
    def fetch_and_process_images(self, latitude, longitude, radius=100, resolution = 1024, max_images=250):
        class HiddenPrints:
            def __enter__(self):
                self._original_stdout = sys.stdout
                self._original_stderr = sys.stderr
                sys.stdout = open(os.devnull, 'a')
                sys.stderr = open(os.devnull, 'a')

            def __exit__(self, exc_type, exc_val, exc_tb):
                sys.stdout.close()
                sys.stdout = self._original_stdout
                sys.stderr = self._original_stderr
        
        print(f"Requesting street view images from Mapillary... (Latitude: {latitude}, Longitude: {longitude})")
        with HiddenPrints():
            try:
                data = self.mly_interface.get_image_close_to(longitude=longitude, latitude=latitude, radius=radius).to_dict()
            except Exception as e:
                print(f"Error fetching data from Mapillary: {e}")
                return
            
            cv2_image_list = []
            for index in range(max_images):
                image_id = data['features'][index]['properties']['id']

                try:
                    image_url = mly.image_thumbnail(image_id= image_id, resolution=resolution)
                    print("Requesting thumbnail for image ID:", image_id)
                    response = requests.get(image_url)
                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    cv2_image_list.append(cv2.imdecode(image_array, cv2.IMREAD_COLOR))
                except InvalidImageKeyError as e:
                    print(f"[Warning] Invalid image ID: {image_id}- skipping...")
                    continue
                except requests.exceptions.HTTPError as e:
                    print(f"[HTTP Error] Failed to fetch image {image_id}: {e}")
                    continue
                except Exception as e:
                    print(f"[Error] Unexpected error occurred for image {image_id}: {e}")
                    continue

        return cv2_image_list


"""
 Athens, Greece(Exarcheia)
 Naples,Italy (Centro Storico)
 Santiago, Chile (Barrio Yungay)
 Baltimore, USA (West Baltimore)
 Marseille, France (Noailles)
 Detroit, USA (Delray)
 Rio de Janeiro, Brazil (Lapa)
 Tijuana, Mexico (Zona Norte)
 Barcelona, Spain (El Raval)
 Johannesburg, South Africa (Yeoville)
"""    
if __name__ == "__main__":
    TOKEN = 'MLY|9398391603544096|d47348ca942d8d06c2d47825aa4c6f70'

    fetcher = MapillaryImageFetcher(access_token=TOKEN)