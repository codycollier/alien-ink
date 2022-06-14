""" Config management

"""

import json
import os.path


#------------------------------------------------------------------------------
# Configuration options
#------------------------------------------------------------------------------

# Google Colab & Drive
gdrive_root = "/content/gdrive"


# Kaggle
kaggle_username = None
kaggle_key = None


#------------------------------------------------------------------------------
# Config loading and utils
#------------------------------------------------------------------------------

# Overall config sources
config_dirs = ["/orb", os.path.expanduser("~/"), ]
config_file = ".alien-ink-secrets.json"


def load():
    """Load configuration"""

    # Find config file or return early
    cf = None
    for config_dir in config_dirs:
        cf_candidate = os.path.join(config_dir, config_file)
        if os.path.exists(cf_candidate):
            cf = cf_candidate
            break

    if cf is None:
        return

    # Load the config file contents
    conf = None
    with open(cf, 'r') as cfile:
        conf = json.loads(cfile.read())

    # Custom overrides
    if "kaggle_username" in conf:
        kaggle_username = conf["kaggle_username"]
        os.environ["KAGGLE_USERNAME"] = kaggle_username

    if "kaggle_key" in conf:
        kaggle_key = conf["kaggle_key"]
        os.environ["KAGGLE_KEY"] = kaggle_key

    return


if __name__ == "__main__":

    load()







