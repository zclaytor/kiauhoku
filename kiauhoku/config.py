'''
config.py

Set basic configurations to be used by the rest of the package.

Contains
--------
`grids_path` (str): directory to install stellar model grids. Defaults to a
    hidden cache directory created in the home directory: ~/.kiauhoku/grids

`grids_url` (str): Zenodo URL from which to download model grids. This URL
    is the generic one for the repository; it defaults to the latest version.
'''

import os
from socket import gethostname


grids_url_base = "https://zenodo.org/api/records"
grids_url = os.path.join(grids_url_base, "4287717")

grids_version_url = { # hard coding this until I find a better solution
    "2.0": "6041150",
    "2.0.0": "6041150",
    "2.0.1": "6597404",
    "2.0.3": "10975758",
    "2.1.0": "11264222",
}
grids_version_url = {
    key: os.path.join(grids_url_base, grids_version_url[key]) for key in grids_version_url}


if "ufhpc" in gethostname():
    grids_path = "/blue/jtayar/shared/kiauhoku_grids"
else:
    grids_path = os.path.join(os.path.expanduser('~/'), '.kiauhoku/grids')
