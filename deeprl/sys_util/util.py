import os

def find_dir(root_dir, dir_name):
    for root, dirs, _ in os.walk(root_dir):
        if dir_name in dirs:
            return os.path.join(root, dir_name)
