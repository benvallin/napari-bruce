# %% Import required libraries ----

import os
import sys
import subprocess
import importlib.resources
import json
import shutil
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

# %% _get_desktop_path() ----


def _get_desktop_path() -> Path:
    """Best-effort path to the user's Desktop directory, cross-platform.

    Asks the OS where the Desktop is (it can be redirected, e.g. by OneDrive on
    Windows, or localized / relocated on Linux) and falls back to ~/Desktop, then
    to the home directory if no Desktop is reachable.

    Returns:
      Path: path to the Desktop directory, or to the home directory as fallback.

    """

    home = Path.home()

    if sys.platform == "win32":

        try:

            import ctypes

            # FOLDERID_Desktop = {B4BFCC3A-DB2C-424C-B029-7FE99A87C641}
            guid = (ctypes.c_ubyte * 16)(
                0x3A,
                0xCC,
                0xBF,
                0xB4,
                0x2C,
                0xDB,
                0x4C,
                0x42,
                0xB0,
                0x29,
                0x7F,
                0xE9,
                0x9A,
                0x87,
                0xC6,
                0x41,
            )
            buf = ctypes.c_wchar_p()
            ctypes.windll.shell32.SHGetKnownFolderPath(
                ctypes.byref(guid), 0, None, ctypes.byref(buf)
            )
            path = Path(buf.value)
            ctypes.windll.ole32.CoTaskMemFree(buf)

            if path.is_dir():
                return path

        except Exception:

            pass

    elif sys.platform.startswith("linux"):

        try:

            out = subprocess.run(
                ["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=2
            ).stdout.strip()

            if out and Path(out).is_dir():
                return Path(out)

        except Exception:

            pass

    desktop = home / "Desktop"

    return desktop if desktop.is_dir() else home


# %% get_config_file_path() ----


def get_config_file_path() -> str:
    """Get path to config file.

    Returns:
      str: path to config file expected location on disk.

    """

    output = os.path.join(importlib.resources.files("napari_bruce"), "config.json")

    return output


# %% list_stardist_models() ----


def list_stardist_models() -> dict:
    """List available StarDist models.

    Returns:
      dict: dict with keys and values representing model name and source, respectively.

    """

    pretrained = ["2D_versatile_fluo", "2D_paper_dsb2018", "2D_demo"]

    pretrained = {x: "pretrained" for x in sorted(pretrained, key=str.lower)}

    user_models_dir = Path(
        os.path.join(importlib.resources.files("napari_bruce"), "stardist_models")
    )

    if not (os.path.exists(user_models_dir) and os.path.isdir(user_models_dir)):

        user_defined = {}

    else:

        tmp = [i for i in user_models_dir.iterdir() if i.is_dir()]

        tmp = {Path(i).name: [j.name for j in i.iterdir()] for i in tmp}

        # Only require the files StarDist needs to load a model for inference.
        # Training artifacts (logs/, weights_last.h5) are intentionally not
        # required — the bundled models ship without them to keep the wheel
        # small, and user-added models need only these three files.
        tmp = [
            k
            for k, v in tmp.items()
            if all(
                i in v
                for i in [
                    "config.json",
                    "thresholds.json",
                    "weights_best.h5",
                ]
            )
        ]

        user_defined = {x: "user-defined" for x in sorted(tmp, key=str.lower)}

    output = {**pretrained, **user_defined}

    return output


# %% list_laser_functions() ----


def list_laser_functions() -> list:
    """List available laser functions.

    Returns:
      list: available laser function.

    """

    output = [
        "Cut",
        "JointCut",
        "CloseCut",
        "LPC",
        "LineAutoLPC",
        "AutoLPC",
        "CloseCut + AutoLPC",
        "RoboLPC",
        "CenterRoboLPC",
    ]

    return output


# %% list_tube_ids() ----


def list_tube_ids() -> list:
    """List available tube IDs.

    Returns:
      list: available tube IDs.

    """

    output = ["manual"] + [f"Tube {i}" for i in list(range(1, 9))]

    return output


# %% list_channel_colors() ----


def list_channel_colors() -> list:
    """List available channel colors.

    Returns:
      list: available channel colors.

    """

    output = [
        "purple",
        "cyan",
    ]

    return output


# %% list_population_colors() ----


def list_population_colors() -> list:
    """List available population colors.

    Returns:
      list: available population colors.

    """

    output = [
        "magenta",
        "yellow",
        "cyan",
        "darkred",
        "grey",
    ]

    return output


# %% check_config_integrity() ----


class ConfigError(RuntimeError):
    """Raised when the napari-bruce configuration is invalid."""

    pass


def check_config_integrity(config: dict) -> None:
    """Check the integrity of the config dict.

    Args:
      config: dict representing configuration.

    Side effects:
      Raises a ConfigError if config is not a dict or a malformed dict.

    """

    if not isinstance(config, dict):

        raise ConfigError(f"JSON stored in config file is not a dict.")

    def check_dict_kv(exp_kv, in_dict, dict_nm):

        if not isinstance(in_dict, dict):

            raise ConfigError(f"{dict_nm} must be a dict.")

        missing_keys = [x for x in exp_kv.keys() if x not in in_dict.keys()]

        if len(missing_keys) > 0:

            tmp = ", ".join(missing_keys)

            raise ConfigError(f"Missing {dict_nm} key(s): {tmp}.")

        invalid_vals = [
            x for x in exp_kv.keys() if not isinstance(in_dict[x], exp_kv[x])
        ]

        if len(invalid_vals) > 0:

            tmp = ", ".join(invalid_vals)

            raise ConfigError(f"Invalid value type for {dict_nm} key(s): {tmp}.")

        return

    exp_main_kv = {
        "in_dir_path": str,
        "out_dir_path": str,
        "channels": dict,
        "min_pct_ovl_ch0_by_ch1": (int, float),
        "min_pct_ovl_ch1_by_ch0": (int, float),
        "elements": dict,
        "channels_annotation": dict,
        "prediction_settings": dict,
    }

    exp_ch_kv = {"0": dict, "1": dict}

    exp_subch_kv = {
        "name": str,
        "background_subtraction": bool,
        "radius": (int, float),
        "robust_normalization": bool,
        "low_pct": (int, float),
        "high_pct": (int, float),
        "stardist_model": str,
        "min_area_um2": (int, float),
        "color": str,
    }

    # The 5 cross-channel populations; "extract_z_position" and "tube_id_matching"
    # also live in config["elements"] but are validated separately below, not as
    # populations.
    population_keys = [
        "ch0-pos/ch1-neg",
        "ch0-pos/ch1-pos",
        "ch0-pos/ch1-amb",
        "ch1-pos/ch0-neg",
        "ch1-pos/ch0-amb",
    ]

    exp_elem_kv = {k: dict for k in population_keys} | {
        "extract_z_position": bool,
        "tube_id_matching": dict,
    }

    exp_subelem_kv = {
        "color": str,
        "collect": bool,
        "laser_function": str,
        "tube_id": str,
    }

    exp_ch_annot_kv = {"0": dict, "1": dict}

    exp_subch_annot_kv = {
        "background_subtraction": bool,
        "radius": (int, float),
        "robust_normalization": bool,
        "low_pct": (int, float),
        "high_pct": (int, float),
        "color": str,
    }

    # 'radius' is only required when background_subtraction is on, and
    # 'low_pct' / 'high_pct' only when robust_normalization is on; when present
    # they are still type- and range-checked.
    exp_pred_settings_kv = {
        "background_subtraction": bool,
        "robust_normalization": bool,
    }

    opt_pred_settings_kv = {
        "radius": (int, float),
        "low_pct": (int, float),
        "high_pct": (int, float),
    }

    check_dict_kv(exp_kv=exp_main_kv, in_dict=config, dict_nm="config main")

    for i in ["min_pct_ovl_ch0_by_ch1", "min_pct_ovl_ch1_by_ch0"]:

        if not (config[i] >= 0 and config[i] <= 100):

            raise ConfigError(f"config['{i}'] must be between 0 and 100.")

    check_dict_kv(
        exp_kv=exp_ch_kv, in_dict=config["channels"], dict_nm="config channels"
    )

    available_models = [k for k in list_stardist_models().keys()]

    available_channel_colors = list_channel_colors()

    for i in config["channels"].keys():

        check_dict_kv(
            exp_kv=exp_subch_kv,
            in_dict=config["channels"][i],
            dict_nm=f"config channel {i}",
        )

        if config["channels"][i]["stardist_model"] not in available_models:

            raise ConfigError(
                f"config['channels']['{i}']['stardist_model'] must be one of {', '.join(available_models)}."
            )

        if config["channels"][i]["color"] not in available_channel_colors:

            raise ConfigError(
                f"config['channels']['{i}']['color'] must be one of {', '.join(available_channel_colors)}."
            )

        for j in ["low_pct", "high_pct"]:

            if not (config["channels"][i][j] >= 0 and config["channels"][i][j] <= 100):

                raise ConfigError(
                    f"config['channels']['{i}']['{j}'] must be between 0 and 100."
                )

        if not (config["channels"][i]["radius"] > 0):

            raise ConfigError(
                f"config['channels']['{i}']['radius'] must be greater than 0."
            )

    check_dict_kv(
        exp_kv=exp_elem_kv, in_dict=config["elements"], dict_nm="config elements"
    )

    available_laser_functions = list_laser_functions()

    available_tube_ids = list_tube_ids()

    available_population_colors = list_population_colors()

    for i in population_keys:

        check_dict_kv(
            exp_kv=exp_subelem_kv,
            in_dict=config["elements"][i],
            dict_nm=f"config elements {i}",
        )

        if config["elements"][i]["laser_function"] not in available_laser_functions:

            raise ConfigError(
                f"config['elements']['{i}']['laser_function'] must be one of {', '.join(available_laser_functions)}."
            )

        if config["elements"][i]["tube_id"] not in available_tube_ids:

            raise ConfigError(
                f"config['elements']['{i}']['tube_id'] must be one of {', '.join(available_tube_ids)}."
            )

        if config["elements"][i]["color"] not in available_population_colors:

            raise ConfigError(
                f"config['elements']['{i}']['color'] must be one of {', '.join(available_population_colors)}."
            )

    tube_id_matching = config["elements"]["tube_id_matching"]

    check_dict_kv(
        exp_kv={"enabled": bool, "sets": list},
        in_dict=tube_id_matching,
        dict_nm="config elements tube_id_matching",
    )

    seen_matches = []

    for idx, tube_id_set in enumerate(tube_id_matching["sets"]):

        if not isinstance(tube_id_set, dict):

            raise ConfigError(
                f"config['elements']['tube_id_matching']['sets'][{idx}] must be a dict."
            )

        check_dict_kv(
            exp_kv={"match": str, "tube_ids": dict},
            in_dict=tube_id_set,
            dict_nm=f"config elements tube_id_matching set {idx}",
        )

        match = tube_id_set["match"]

        if not match.strip():

            raise ConfigError(
                f"config['elements']['tube_id_matching']['sets'][{idx}]['match'] "
                "must not be empty (an empty string matches every filename)."
            )

        if match.casefold() in seen_matches:

            raise ConfigError(
                f"config['elements']['tube_id_matching']['sets'] has duplicate match string: '{match}'."
            )

        seen_matches.append(match.casefold())

        set_keys = list(tube_id_set["tube_ids"].keys())

        if set(set_keys) != set(population_keys):

            raise ConfigError(
                f"config['elements']['tube_id_matching']['sets'][{idx}]['tube_ids'] keys must match "
                f"the population keys: {', '.join(population_keys)}."
            )

        for key, tube_id in tube_id_set["tube_ids"].items():

            if not isinstance(tube_id, str) or tube_id not in available_tube_ids:

                raise ConfigError(
                    f"config['elements']['tube_id_matching']['sets'][{idx}]['tube_ids']['{key}'] "
                    f"must be one of {', '.join(available_tube_ids)}."
                )

    check_dict_kv(
        exp_kv=exp_ch_annot_kv,
        in_dict=config["channels_annotation"],
        dict_nm="config channels annotation",
    )

    for i in config["channels_annotation"].keys():

        check_dict_kv(
            exp_kv=exp_subch_annot_kv,
            in_dict=config["channels_annotation"][i],
            dict_nm=f"config channel annotation {i}",
        )

        if config["channels_annotation"][i]["color"] not in available_channel_colors:

            raise ConfigError(
                f"config['channels_annotation']['{i}']['color'] must be one of {', '.join(available_channel_colors)}."
            )

        for j in ["low_pct", "high_pct"]:

            if not (
                config["channels_annotation"][i][j] >= 0
                and config["channels_annotation"][i][j] <= 100
            ):

                raise ConfigError(
                    f"config['channels_annotation']['{i}']['{j}'] must be between 0 and 100."
                )

        if not (config["channels_annotation"][i]["radius"] > 0):

            raise ConfigError(
                f"config['channels_annotation']['{i}']['radius'] must be greater than 0."
            )

    # Default image preprocessing settings per StarDist model (the settings the
    # model was trained with). Keys are model names; they are deliberately not
    # checked against the available models, which differ between machines.
    for model_nm in config["prediction_settings"].keys():

        entry = config["prediction_settings"][model_nm]

        check_dict_kv(
            exp_kv=exp_pred_settings_kv,
            in_dict=entry,
            dict_nm=f"config prediction_settings '{model_nm}'",
        )

        required = {}

        if entry["background_subtraction"]:
            required["radius"] = opt_pred_settings_kv["radius"]

        if entry["robust_normalization"]:
            required["low_pct"] = opt_pred_settings_kv["low_pct"]
            required["high_pct"] = opt_pred_settings_kv["high_pct"]

        check_dict_kv(
            exp_kv=required,
            in_dict=entry,
            dict_nm=f"config prediction_settings '{model_nm}'",
        )

        present = {k: v for k, v in opt_pred_settings_kv.items() if k in entry}

        check_dict_kv(
            exp_kv=present,
            in_dict=entry,
            dict_nm=f"config prediction_settings '{model_nm}'",
        )

        for j in ["low_pct", "high_pct"]:

            if j in entry and not (0 <= entry[j] <= 100):

                raise ConfigError(
                    f"config['prediction_settings']['{model_nm}']['{j}'] must be between 0 and 100."
                )

        if "radius" in entry and not (entry["radius"] > 0):

            raise ConfigError(
                f"config['prediction_settings']['{model_nm}']['radius'] must be greater than 0."
            )

    return


# %% write_config() ----


def write_config(config: dict, path: str | Path | None = None) -> None:
    """Write 'config' to the config.json file as UTF-8, atomically.

    Args:
      config (dict): dict representing configuration.
      path (str | Path | None): destination path; defaults to get_config_file_path().

    Side effects:
      Overwrites the config.json file (or 'path') if it already exists.

    """

    if path is None:

        path = get_config_file_path()

    # Write to a sibling tmp file then atomically replace, so a crash/kill
    # mid-write can never leave a truncated config.json behind -- a prior
    # incident lost the real config.json to corruption with no way to recover
    # it. encoding="utf-8" avoids the Windows cp1252 UnicodeEncodeError this
    # project already hit once for non-ASCII channel/population labels.
    tmp_path = f"{path}.tmp"

    with open(tmp_path, "w", encoding="utf-8") as f:

        json.dump(config, f, indent=2)

    os.replace(tmp_path, path)


# %% make_default_config() ----


def make_default_config() -> dict:
    """Generate default configuration and write to config.json file.

    Returns:
      dict: dict representing default configuration.

    Side effects:
      Writes default configuration to config.json file. This function overwrites the file if it already exists.

    """

    path = get_config_file_path()

    in_dir_path = str(Path.home())

    out_dir_path = str(Path(_get_desktop_path(), "napari_bruce_results"))

    output = {
        "in_dir_path": in_dir_path,
        "out_dir_path": out_dir_path,
        "channels": {
            0: {
                "name": "TH",
                "background_subtraction": True,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 0.0500,
                "high_pct": 99.9500,
                "stardist_model": "stardist_th_cam_visu_no_bs",
                "min_area_um2": 100,
                "color": "purple",
            },
            1: {
                "name": "pSyn",
                "background_subtraction": True,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 1.0000,
                "high_pct": 99.9990,
                "stardist_model": "stardist_psyn_cam_visu_no_bs",
                "min_area_um2": 50,
                "color": "cyan",
            },
        },
        "min_pct_ovl_ch0_by_ch1": 20,
        "min_pct_ovl_ch1_by_ch0": 80,
        "elements": {
            "ch0-pos/ch1-neg": {
                "color": "magenta",
                "collect": True,
                "laser_function": "RoboLPC",
                "tube_id": "Tube 1",
            },
            "ch0-pos/ch1-pos": {
                "color": "yellow",
                "collect": True,
                "laser_function": "RoboLPC",
                "tube_id": "Tube 2",
            },
            "ch1-pos/ch0-neg": {
                "color": "cyan",
                "collect": True,
                "laser_function": "RoboLPC",
                "tube_id": "Tube 3",
            },
            "ch0-pos/ch1-amb": {
                "color": "darkred",
                "collect": False,
                "laser_function": "RoboLPC",
                "tube_id": "manual",
            },
            "ch1-pos/ch0-amb": {
                "color": "grey",
                "collect": False,
                "laser_function": "RoboLPC",
                "tube_id": "manual",
            },
            # When True, the Z (focus) stage position extracted from the .zvi
            # metadata is written to the exported element list's Z column;
            # when False, that column is populated with the default missing
            # value ("-"), as it was before Z extraction was implemented.
            "extract_z_position": False,
            # When 'enabled' is True, the set whose 'match' string is found
            # (case-insensitive substring) in the loaded image file name
            # overrides the per-population 'tube_id' defaults above.
            "tube_id_matching": {
                "enabled": False,
                "sets": [
                    {
                        "match": "-l-",
                        "tube_ids": {
                            "ch0-pos/ch1-neg": "Tube 1",
                            "ch0-pos/ch1-pos": "Tube 2",
                            "ch1-pos/ch0-neg": "Tube 3",
                            "ch0-pos/ch1-amb": "manual",
                            "ch1-pos/ch0-amb": "manual",
                        },
                    },
                    {
                        "match": "-r-",
                        "tube_ids": {
                            "ch0-pos/ch1-neg": "Tube 4",
                            "ch0-pos/ch1-pos": "Tube 5",
                            "ch1-pos/ch0-neg": "Tube 6",
                            "ch0-pos/ch1-amb": "manual",
                            "ch1-pos/ch0-amb": "manual",
                        },
                    },
                ],
            },
        },
        "channels_annotation": {
            0: {
                "background_subtraction": True,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 0.0500,
                "high_pct": 99.9500,
                "color": "purple",
            },
            1: {
                "background_subtraction": True,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 1.0000,
                "high_pct": 99.9990,
                "color": "cyan",
            },
        },
        "prediction_settings": {
            "2026.04.21-06.10_stardist_th_cam": {
                "background_subtraction": False,
                "robust_normalization": True,
                "low_pct": 0.05,
                "high_pct": 99.95,
            },
            "2026.04.21-06.10_stardist_psyn_cam": {
                "background_subtraction": False,
                "robust_normalization": True,
                "low_pct": 0.05,
                "high_pct": 99.9999,
            },
            "2026.04.21_stardist_th_cam": {
                "background_subtraction": False,
                "robust_normalization": True,
                "low_pct": 0.05,
                "high_pct": 99.95,
            },
            "2026.04.21_stardist_psyn_cam": {
                "background_subtraction": False,
                "robust_normalization": True,
                "low_pct": 0.05,
                "high_pct": 99.9999,
            },
            "stardist_th_cam_visu": {
                "background_subtraction": True,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 0.0500,
                "high_pct": 99.9500,
            },
            "stardist_psyn_cam_visu": {
                "background_subtraction": True,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 1.0000,
                "high_pct": 99.9990,
            },
            "stardist_th_cam_visu_no_bs": {
                "background_subtraction": False,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 0.0500,
                "high_pct": 99.9500,
            },
            "stardist_psyn_cam_visu_no_bs": {
                "background_subtraction": False,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 1.0000,
                "high_pct": 99.9990,
            },
            "stardist_th_ben_visu": {
                "background_subtraction": True,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 0.0500,
                "high_pct": 99.9500,
            },
            "stardist_psyn_ben_visu": {
                "background_subtraction": True,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 1.0000,
                "high_pct": 99.9990,
            },
            "stardist_th_ben_visu_no_bs": {
                "background_subtraction": False,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 0.0500,
                "high_pct": 99.9500,
            },
            "stardist_psyn_ben_visu_no_bs": {
                "background_subtraction": False,
                "radius": 50,
                "robust_normalization": True,
                "low_pct": 1.0000,
                "high_pct": 99.9990,
            },
        },
    }

    write_config(output, path=path)

    return output


# %% get_config() ----


def get_config() -> dict:
    """Read configuration from config.json file and check integrity.

    Returns:
      dict: dict representing configuration. If config.json file does not exist, creates it and returns default config.

    """

    path = get_config_file_path()

    if os.path.exists(path):

        try:

            with open(path, "r", encoding="utf-8") as f:

                output = json.load(f)

        except json.JSONDecodeError:

            raise ValueError(f"Invalid JSON in config file.")

        except OSError:

            raise OSError(f"Could not read config file.")

        check_config_integrity(config=output)

        for i in ["channels", "channels_annotation"]:
            output[i] = {int(k): v for k, v in output[i].items()}

    else:

        output = make_default_config()

    return output


# %% open_in_editor() ----


def open_in_editor(path: str) -> None:
    """Open 'path' in the system's default editor / viewer.

    Args:
      path (str): path to file.

    """

    path = str(Path(path).resolve())

    if sys.platform == "darwin":  # macOS

        subprocess.run(["open", "-a", "TextEdit", path], check=False)

    elif sys.platform.startswith("win"):  # Windows

        subprocess.run(["notepad.exe", path], check=False)

    else:  # Linux / others

        subprocess.run(["xdg-open", path], check=False)


# %% add_stardist_model() ----


def add_stardist_model(src: str) -> None:
    """Add StarDist model to napari-bruce.

    Args:
      src: path to input StarDist model directory.

    Side effects:
      Copies the input StarDist model directory into the napari-bruce src directory.

    """

    in_dir_path = Path(src).expanduser()

    if not os.path.exists(in_dir_path):

        raise FileNotFoundError(f"Input path does not exist: {in_dir_path}.")

    if not os.path.isdir(in_dir_path):

        raise NotADirectoryError(f"Input path is not a directory: {in_dir_path}.")

    out_dir_path = os.path.join(
        os.path.join(importlib.resources.files("napari_bruce"), "stardist_models"),
        Path(in_dir_path).name,
    )

    shutil.copytree(
        src=in_dir_path,
        dst=out_dir_path,
        copy_function=shutil.copy2,
        dirs_exist_ok=True,
    )


# %% get_version() ----


def get_version() -> str:
    """Get package version.

    Returns:
      str: package version.

    """

    # Read the live __version__ from the source so editable installs reflect the
    # working tree without a reinstall; fall back to the installed metadata.
    try:

        from napari_bruce import __version__

        return __version__

    except Exception:

        try:

            return version("napari-bruce")

        except PackageNotFoundError:

            return "unknown"
