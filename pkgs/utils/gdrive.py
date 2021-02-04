import hashlib
import os

import requests

from tqdm import tqdm


def sha256_checksum(filename, block_size=65536):
    sha256 = hashlib.sha256()
    with open(filename, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha256.update(block)
    return sha256.hexdigest()


def download_file_from_gdrive(
    dst,
    google_id=None,
    url=None,
    file_size=None,
    desc=None,
    sha256=None,
    delete_on_err=True,
):
    assert (
        google_id is not None or url is not None
    ), "required google_id or url"

    if url is not None:
        if "://drive.google.com/" not in url:
            # assume url redirect to gdrive
            r = requests.get(url)
            url = r.url

        assert (
            "://drive.google.com/" in url
        ), "expected url to be a google drive url"

        google_id = url.split("/")[-2]

    gdoc_url = "https://docs.google.com/uc"

    session = requests.Session()

    params = {"id": google_id, "export": "download"}
    response = session.get(gdoc_url, params=params, stream=True)
    token = _get_confirm_token(response)

    if token:
        params["confirm"] = token
        response = session.get(gdoc_url, params=params, stream=True)
    else:
        raise IOError("unable to obtain token for download")

    _save_response_content(response, dst, file_size=file_size, desc=desc)

    if sha256 is not None:
        calc_hash = sha256_checksum(dst)

        if calc_hash != sha256 and delete_on_err:
            os.remove(dst)

            raise ValueError(
                f"unexpected sha256 checksum: {calc_hash} (expected: {sha256})"
            )


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, dst, file_size=None, desc=None):
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc=desc)

    with (open(dst, "ab")) as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)

    pbar.close()
