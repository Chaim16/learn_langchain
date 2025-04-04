import yaml

from utils.constants import CONFIG_PATH

with open(CONFIG_PATH, "r") as f:
    application_config = yaml.safe_load(f)



def openai_config():
    return application_config.get("openai")


if __name__ == '__main__':
    print(openai_config())