# %% Set up ----

# Import required libraries
import os
import sys
import argparse
import subprocess
import shutil
import time
import threading
import itertools
import tempfile
from pathlib import Path
from . import configuration
from . import workflow

# %% _launch_napari_with_plugin() ----


def _is_unicode_supported():

    enc = getattr(sys.stdout, "encoding", None) or ""

    return "utf" in enc.lower()


def _spin_until(condition, message="Starting napari-bruce...", delay=0.08, timeout=None):

    if not sys.stdout.isatty():

        start = time.time()

        while True:

            if condition():

                return True

            if timeout is not None and (time.time() - start) > timeout:

                return False

            time.sleep(0.05)

    stop_event = threading.Event()

    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if _is_unicode_supported() else "|/-\\"

    line_len = len(message) + 2

    def _spin():

        for f in itertools.cycle(frames):

            if stop_event.is_set():

                break

            sys.stdout.write(f"\r{f} {message}")

            sys.stdout.flush()

            time.sleep(delay)

    t = threading.Thread(target=_spin, daemon=True)

    t.start()

    start = time.time()

    try:

        while True:

            if condition():

                return True

            if timeout is not None and (time.time() - start) > timeout:

                return False

            time.sleep(0.05)

    finally:

        stop_event.set()

        t.join()

        sys.stdout.write("\r" + " " * line_len + "\r")

        sys.stdout.flush()


def _launch_napari_with_plugin(timeout=60.0):

    ready_file = Path(os.path.join(tempfile.gettempdir(), "napari_bruce_ready"))

    try:

        if ready_file.exists():

            ready_file.unlink()

    except Exception:

        pass

    env = os.environ.copy()

    env["NAPARI_BRUCE_READY_FILE"] = str(ready_file)

    env.setdefault("QT_SCALE_FACTOR", "1")

    cmd = ["napari", "--with", "napari-bruce"]

    try:

        proc = subprocess.Popen(cmd, env=env)

        def condition():

            if ready_file.exists():

                return True

            return proc.poll() is not None

        print("\n")

        _spin_until(condition=condition, timeout=timeout)

        if ready_file.exists():

            print("\u2714 Program started\n")

        elif proc.poll() is not None:

            print(f"napari process exited with code {proc.returncode}\n")

        else:

            print("Timeout waiting for napari to signal ready")

        returncode = proc.wait()

        return returncode

    except FileNotFoundError:

        print(
            "FileNotFoundError: 'napari' not found. Ensure it is installed and on PATH.\n",
            file=sys.stderr,
        )

        raise SystemExit(1)


# %% cli_main() ----


def cli_main(argv: list[str] | None = None) -> None:

    try:

        parser = argparse.ArgumentParser(
            prog="bruce",
            description="Command-line interface for the napari-bruce plugin.",
        )

        parser.add_argument(
            "--show-config-path",
            action="store_true",
            help="print the path of the configuration file and exit",
        )

        parser.add_argument(
            "--open-config-file",
            action="store_true",
            help="open the configuration file in the default text editor",
        )

        parser.add_argument(
            "--edit-config",
            action="store_true",
            help="edit the configuration in a GUI window",
        )

        parser.add_argument(
            "--reset-config",
            action="store_true",
            help="reset the configuration to defaults and exit",
        )

        parser.add_argument(
            "--gpu-status",
            action="store_true",
            help="check if GPU(s) are visible to TensorFlow",
        )

        parser.add_argument(
            "--list-models",
            action="store_true",
            help="list available StarDist models",
        )

        parser.add_argument(
            "--add-model",
            metavar="MODEL_DIR",
            help="add the StarDist model located at MODEL_DIR to napari-bruce",
        )

        parser.add_argument(
            "--zvi-to-dict",
            nargs=2,
            metavar=("IN_DIR", "OUT_DIR"),
            help="construct an image set from 2-channel .zvi files located at IN_DIR and save to OUT_DIR/imgs.pkl",
        )

        parser.add_argument(
            "--keep-ome-tiff",
            action="store_true",
            help="with --zvi-to-dict: delete the intermediate ome.tiff directory after processing",
        )

        parser.add_argument(
            "--save-tiff",
            action="store_true",
            help="with --zvi-to-dict: write single-channel .tiff files at OUT_DIR/tiff/",
        )

        parser.add_argument(
            "--annotate",
            nargs=2,
            metavar=("IN_DIR", "OUT_DIR"),
            help="open the annotation GUI to label the image set at IN_DIR/imgs.pkl (or IN_DIR/imgs_annotated.pkl if exists) and save image/mask pairs to OUT_DIR/imgs_annotated.pkl",
        )

        parser.add_argument(
            "-v",
            "--version",
            action="store_true",
            help="show the napari-bruce version and exit",
        )

        args = parser.parse_args(argv)

        if args.show_config_path:

            config_path = configuration.get_config_file_path()
            if not os.path.exists(config_path):
                configuration.make_default_config()
            print(config_path)

            return

        if args.open_config_file:

            config_path = configuration.get_config_file_path()
            if not os.path.exists(config_path):
                configuration.make_default_config()
            print(f"Opening config file at:\n{config_path}")
            configuration.open_in_editor(config_path)

            return

        if args.edit_config:

            config_path = configuration.get_config_file_path()
            if not os.path.exists(config_path):
                configuration.make_default_config()

            from .config_editor import launch_config_editor

            launch_config_editor()

            return

        if args.reset_config:

            config_path = configuration.get_config_file_path()
            configuration.make_default_config()
            print(f"Configuration reset to defaults at:\n{config_path}")

            return

        if args.gpu_status:

            print("\nChecking TensorFlow / StarDist GPU status...")

            try:

                os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

                import tensorflow as tf

            except Exception as e:

                print(
                    f"\n🔴 TensorFlow could not be imported.\n{type(e).__name__}: {e}\n"
                )

                return

            gpus = tf.config.list_physical_devices("GPU")

            if not gpus:

                print("\n🔴 No GPU visible to TensorFlow.\nStarDist will run on CPU.\n")

                return

            print(f"\n🟢 GPU(s) visible to TensorFlow: {gpus}")

            if sys.platform.startswith("win"):

                if shutil.which("ptxas.exe") is None:

                    print("\nWARNING: 'ptxas.exe' not found in PATH.")
                    print(
                        "CUDA Toolkit is likely not installed correctly; TensorFlow may fall back to CPU."
                    )
                    print("Install CUDA Toolkit 11.2 and add its 'bin' folder to PATH.")

                else:

                    print("\n🟢 CUDA compiler (ptxas.exe) found in PATH.")

            try:

                with tf.device("/GPU:0"):

                    a = tf.random.normal([2000, 2000])
                    b = tf.random.normal([2000, 2000])
                    start = time.time()
                    c = tf.matmul(a, b)
                    _ = c.numpy()
                    elapsed = time.time() - start

                print(
                    f"\n🟢 GPU functional test succeeded (matmul time: {elapsed:.3f}s)"
                )

                if elapsed > 2.0:

                    print(
                        "\nGPU test ran but was slow.\nThis may indicate CPU fallback."
                    )

                else:

                    print("\n🟢 StarDist should run on GPU correctly.\n")

            except Exception as e:

                print(f"\n🔴 GPU functional test failed.\n{type(e).__name__}: {e}")
                print(
                    "TensorFlow detected a GPU but cannot execute on it.\nStarDist will fall back to CPU.\n"
                )

            return

        if args.list_models:

            models = configuration.list_stardist_models()
            models = "\n- ".join(f"{k}: {v}" for k, v in models.items())
            print(f"Available StarDist models:\n- {models}")

            return

        if args.add_model is not None:

            configuration.add_stardist_model(args.add_model)
            print(f"Added StarDist model from:\n{args.add_model}")

            return

        if args.zvi_to_dict is not None:

            in_dir_path, out_dir_path = args.zvi_to_dict

            ome_tiff = True if args.keep_ome_tiff else False

            tiff = True if args.save_tiff else False

            config = configuration.get_config()

            low_pct_dict = {
                0: config["channels_annotation"][0]["low_pct"],
                1: config["channels_annotation"][1]["low_pct"],
            }

            high_pct_dict = {
                0: config["channels_annotation"][0]["high_pct"],
                1: config["channels_annotation"][1]["high_pct"],
            }

            workflow.zvi_to_dict(
                in_dir_path=in_dir_path,
                out_dir_path=out_dir_path,
                low_pct_dict=low_pct_dict,
                high_pct_dict=high_pct_dict,
                ome_tiff=ome_tiff,
                tiff=tiff,
            )

            return

        if args.annotate is not None:

            in_dir_path, out_dir_path = args.annotate

            from .annotation_widget import (
                validate_annotation_inputs,
                launch_annotation_viewer,
            )

            validate_annotation_inputs(
                in_dir_path=in_dir_path, out_dir_path=out_dir_path
            )

            launch_annotation_viewer(in_dir_path=in_dir_path, out_dir_path=out_dir_path)

            return

        if args.version:

            print(r"""
        '            '
     /*/    '   '    \*\
   /**/     |\_/|     \**\
  *****-----*****-----*****
 |********* Bruce *********|
  ****/-\***********/-\****
   |*|   \*********/   |*|
    \*\    \*****/    /*/
      \\     \*/     //
        '     '     '
        """)

            print(f"napari-bruce {configuration.get_version()}\n")

            return

        configuration.get_config()

        if shutil.which("java") is None:

            raise RuntimeError("java not found on PATH; please install OpenJDK.")

        _launch_napari_with_plugin()

    except Exception as e:

        print(f"{type(e).__name__}: {e}", file=sys.stderr)

        raise SystemExit(1)
