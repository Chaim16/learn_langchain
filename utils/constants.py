import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG_PATH = os.path.join(BASE_DIR, "conf", "application.yaml")



if __name__ == '__main__':
    print(11111)
    print(BASE_DIR)