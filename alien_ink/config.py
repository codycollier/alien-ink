""" Config management and access

Note that config.ink, config.kaggle, etc are automatically set at import
time via a config file discovery and load procedure.

Generally, it's best to have the following files in place:

colab:
/content/gdrive/My Drive/Colab Notebooks/alien-ink/[alien-ink.json,kaggle.json]

mac/linux:
/orb/alien-ink/[alien-ink.json,kaggle.json]
~/.alien-ink/[alien-ink.json,kaggle.json]

"""

import json
import os.path
import sys


#------------------------------------------------------------------------------
# Main configuration
#------------------------------------------------------------------------------

# alien-ink
ink_root = None
ink_config_file = None
ink = {}

# Kaggle
kaggle_config_dir = None
kaggle = {}


#------------------------------------------------------------------------------
# Static internal configuration details
#------------------------------------------------------------------------------

# Google Colab & Drive
_gdrive_root = "/content/gdrive"

# Overall config sources
_config_dirs = (os.path.join(os.path.expanduser("~/"), ".alien-ink"), 
               "/orb/alien-ink/",
               "/content/gdrive/My Drive/Colab Notebooks/alien-ink/",
               )
_config_file_ink = "alien-ink.json"
_config_file_kaggle = "kaggle.json"


#------------------------------------------------------------------------------
# Config helpers
#------------------------------------------------------------------------------


def in_colab():
    """Return true if inside a google collab notebook"""
    if 'google.colab' in sys.modules:
        return True
    return False


def maybe_setup_colab():
    """Setup any colab stuff"""
    if in_colab():
        from google.colab import drive
        drive.mount('/content/gdrive')


def load_config_ink():
    """Find the alien-ink config file and load"""
    global ink_config_file, ink_root, ink

    # find config file
    for _config_dir in _config_dirs:
        cf_candidate = os.path.join(_config_dir, _config_file_ink)
        if os.path.exists(cf_candidate):
            ink_config_file = cf_candidate
            ink_root = _config_dir
            break

    # load the config file contents
    if ink_config_file:
        with open(ink_config_file, 'r') as cfile:
            ink = json.loads(cfile.read())
            ink["data_root"] = os.path.join(ink_root, "data-ink")
    else:
        raise Exception("alien ink config file is missing")

    return


def load_config_kaggle():
    """Initialize items needed prior to kaggle imports

    In particular, KAGGLE_CONFIG_DIR needs to be set prior to the following
    import, if the kaggle config file is anywhere but the default home dir. So,
    in order to standardize and automate across my different environments 
    including colab, I override the config path.
    """
    global kaggle_config_dir, kaggle

    # find the config file
    for _config_dir in _config_dirs:
        cf_candidate = os.path.join(_config_dir, _config_file_kaggle)
        if os.path.exists(cf_candidate):
            kaggle_config_dir = _config_dir  # note, dir not full path to file
            break

    # set the path as an env later used by the kaggle lib
    if kaggle_config_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = kaggle_config_dir
        with open(cf_candidate, 'r') as cfile:
            kaggle = json.loads(cfile.read())
            kaggle["data_root"] = os.path.join(ink_root, "data-kaggle")
    else:
        raise Exception("kaggle config file is missing")

    return


def __autoload():
    """Automatically find and setup various configs"""
    maybe_setup_colab()
    load_config_ink()
    load_config_kaggle()


#------------------------------------------------------------------------------
# config automation
#------------------------------------------------------------------------------

# automatically run this at import time
__autoload()



if __name__ == "__main__":

    from pprint import pprint

    print("")
    print("alien-ink config:")
    pprint(alien, indent=2, width=50)

    print("")
    print("kaggle config:")
    pprint(kaggle, indent=2, width=50)

    print("")
