import os


class PathInfo:
    def __init__(self):
        self.image_path = ""
        self.database_path = ""
        self.output_path = ""
        self.database_exists = False
        self.depth_path = ""
        self.record_path = ""


def ReadData(path) -> PathInfo:
    path_info = PathInfo()
    if os.path.exists(os.path.join(path, 'images')):
        path_info.image_path = os.path.join(path, 'images')
    elif os.path.exists(os.path.join(path, 'color')):
        path_info.image_path = os.path.join(path, 'color')
    else:
        path_info.image_path = path

    path_info.database_path = os.path.join(path, 'database.db')
    path_info.output_path = os.path.join(path, 'sparse')
    path_info.database_exists = os.path.exists(path_info.database_path)
    if os.path.exists(os.path.join(path, 'depth')):
        path_info.depth_path = os.path.join(path, 'depth')
    path_info.record_path = os.path.join(path, 'record')

    return path_info