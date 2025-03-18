# Standard library
import os  # noqa
from importlib.metadata import PackageNotFoundError, version  # noqa
import configparser  # noqa
import pandas as pd  # noqa
import numpy as np  # noqa: E402
from appdirs import user_config_dir, user_data_dir  # noqa: E402
import astropy.units as u


def get_version():
    try:
        return version("pandorareq")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-2]) + "/tests/"

from pandorasat import get_logger  # noqa: E402

logger = get_logger("pandorareq")


CONFIGDIR = user_config_dir("packagename")
os.makedirs(CONFIGDIR, exist_ok=True)
CONFIGPATH = os.path.join(CONFIGDIR, "config.ini")

logger = get_logger("packagename")


def reset_config():
    """Set the config to defaults."""
    # use this function to set your default configuration parameters.
    config = configparser.ConfigParser()
    config["SETTINGS"] = {
        "log_level": "WARNING",
        "data_dir": user_data_dir("pandorareq"),
    }
    config["NIRDA_REQUIREMENTS"] = {
        "REQUIRED_MAG": 9,
        "REQUIRED_TEFF": 3500,
        "REQUIRED_SNR": 6000,
        "REQUIRED_R": 30,
        "REQUIRED_INTS": 900,
        "LAM": 1300,
        "NFOWLER": 4,
        "NFOWLER_GROUPS": 2,
        "NFRAMES": 24,
    }

    with open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


def load_config() -> configparser.ConfigParser:
    """
    Loads the configuration file, creating it with defaults if it doesn't exist.

    Returns
    -------
    configparser.ConfigParser
        The loaded configuration.
    """

    config = configparser.ConfigParser()

    if not os.path.exists(CONFIGPATH):
        # Create default configuration
        reset_config()
    config.read(CONFIGPATH)
    return config


def save_config(config: configparser.ConfigParser) -> None:
    """
    Saves the configuration to the file.

    Parameters
    ----------
    config : configparser.ConfigParser
        The configuration to save.
    app_name : str
        Name of the application.
    """
    with open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


config = load_config()

# Use this to check that keys you expect are in the config file.
# If you update the config file and think users may be out of date
# add the config parameters to this loop to check and reset the config.
for key in ["data_dir", "log_level"]:
    if key not in config["SETTINGS"]:
        logger.error(
            f"`{key}` missing from the `packagename` config file. Your configuration is being reset."
        )
        reset_config()
        config = load_config()

DATADIR = config["SETTINGS"]["data_dir"]
logger.setLevel(config["SETTINGS"]["log_level"])


def display_config() -> pd.DataFrame:
    dfs = []
    for section in config.sections():
        df = pd.DataFrame(
            np.asarray([(key, value) for key, value in dict(config[section]).items()])
        )
        df["section"] = section
        df.columns = ["key", "value", "section"]
        df = df.set_index(["section", "key"])
        dfs.append(df)
    return pd.concat(dfs)


from .nirdatester import NIRDATester
