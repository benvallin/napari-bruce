# %% Import required libraries ----

import pickle
from pathlib import Path
import tifffile
import shutil
from . import workflow

# %% zvi_to_dict() ----


def zvi_to_dict(
    in_dir_path: str, out_dir_path: str, ome_tiff: bool = False, tiff: bool = False
):

    in_dir_path = Path(in_dir_path).expanduser()
    out_dir_path = Path(out_dir_path).expanduser()

    # Create output directory if necessary
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Define output subdirectories
    ome_tiff_dir_path = Path(out_dir_path, "ome.tiff")
    tiff_dir_path = Path(out_dir_path, "tiff")

    # List image files
    img_fn = list(in_dir_path.rglob("*zvi"))

    # Construct imgs dict
    print("Constructing imgs dict...\n")

    imgs = {}
    imgs_excluded = []

    for i in img_fn:

        fn = i.stem

        # Convert PALM .zvi file to OME-TIFF
        # => Write multi channel images to ome.tiff files at out_dir_path/ome.tiff/
        workflow.convert_zvi_to_ome(
            file=i,
            out_dir_path=ome_tiff_dir_path,
            jar_pkg="napari_bruce.bioformats",
            jar_name="bioformats_package.jar",
        )

        # Load images and associated metadata
        ome_tiff_file_path = Path(ome_tiff_dir_path, f"{fn}.ome.tiff")

        try:

            raw_data, raw_metadata = workflow.load_ome_tiff(file=ome_tiff_file_path)

        except workflow.InvalidImageError:

            print(f"Excluding {fn}")

            imgs_excluded.append(fn)

            continue

        # Subset data and metadata to the first 2 channels
        data = dict(list(raw_data.items())[:2])

        metadata = {
            **raw_metadata,
            "channels": dict(list(raw_metadata["channels"].items())[:2]),
        }

        # For each channel, perform robust normalization
        for i, k in enumerate(data.keys()):

            norm_img = workflow.robust_normalization(
                img=data[k]["img"], low_pct=0.05, high_pct=99.9999
            )

            data[k] = {**data[k], "norm_img": norm_img}

        imgs[fn] = {"data": data, "metadata": metadata}

    # Write imgs dict to file
    with open(Path(out_dir_path, "imgs.pkl"), "wb") as f:
        pickle.dump(imgs, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Delete out_dir_path/ome.tiff/ is requested
    if not ome_tiff:
        shutil.rmtree(ome_tiff_dir_path)

    # Write imgs_excluded list to file
    Path(out_dir_path, "images_excluded.txt").write_text("\n".join(imgs_excluded))

    # Write single channel images to tiff files at out_dir_path/tiff/ is requested
    if tiff:
        tiff_dir_path.mkdir(parents=True, exist_ok=True)
        for i in imgs.keys():
            try:

                for j in imgs[i]["data"]:
                    tifffile.imwrite(
                        Path(tiff_dir_path, f"{i}_{j}_img.tiff"),
                        imgs[i]["data"][j]["img"],
                    )
                    tifffile.imwrite(
                        Path(tiff_dir_path, f"{i}_{j}_norm_img.tiff"),
                        imgs[i]["data"][j]["norm_img"],
                    )
            except:
                continue

    print("Job complete!")
