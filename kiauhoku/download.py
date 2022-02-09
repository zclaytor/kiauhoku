import os
import requests
import tarfile

from .stargrid import grids_path, load_eep_grid


grids_url = "https://zenodo.org/api/records/4287717"


def download(name, kind="eep", create_interpolator=True):
    # create cache directory
    if not os.path.exists(grids_path):
        os.makedirs(grids_path)

    # check permanent record locator to get latest version
    r = requests.get(grids_url)
    if r.ok:
        record_id = r.json()["id"]
        fname = f"{name}_{kind}.tar.gz"
        my_url = f"https://zenodo.org/record/{record_id}/files/{fname}"

        # download and extract files
        r = requests.get(my_url)
        if r.ok:
            tgz_file = os.path.join(grids_path, fname)
            with open(tgz_file, "wb") as f:
                f.write(r.content)
            with tarfile.open(tgz_file) as g:
                g.extractall(grids_path)
        else:
            raise requests.exceptions.RequestException(
                f"Bad request to url: {my_url}"
            )
    else:
        raise requests.exceptions.RequestException(
            f"Bad request to url: {grids_url}"
        )

    if (kind == "eep") and create_interpolator:
        # save grid interpolator
        grid = load_eep_grid(name=name)
        interp = grid.to_interpolator()
        interp.to_pickle(path=os.path.join(grids_path, name, "interpolator.pkl"))
        del grid, interp

    return