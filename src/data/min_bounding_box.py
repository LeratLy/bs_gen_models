from variables import MS_FOLDER, HC_FOLDER
from src.utils.preprocessing import find_bounding_box_multiple

if __name__ == '__main__':
    find_bounding_box_multiple([HC_FOLDER, MS_FOLDER])