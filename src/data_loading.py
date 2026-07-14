"""
Data loading utilities for the Algonauts 2021 Mini-Track dataset.
Adapted from the NMA Computational Neuroscience course notebook
(content creator: Kshitij Dwivedi).
"""

import os
import io
import pickle
import zipfile

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()


def download_algonauts_data(data_dir="data/raw"):
    """
    Downloads and extracts the Algonauts 2021 dataset using the dropbox
    link stored in the .env file.

    Parameters
    ----------
    data_dir : str
        Directory where the data should be extracted.
    """
    dropbox_link = os.getenv("DROPBOX_DATASET_LINK")

    if dropbox_link is None:
        raise ValueError(
            "DROPBOX_DATASET_LINK not set. "
            "Copy .env.example to .env and fill in the link."
        )

    os.makedirs(data_dir, exist_ok=True)

    fname1 = os.path.join(data_dir, "participants_data_v2021")
    fname2 = os.path.join(data_dir, "AlgonautsVideos268_All_30fpsmax")

    if not os.path.exists(fname1) or not os.path.exists(fname2):
        print("Downloading Algonauts data...")
        r = requests.get(dropbox_link)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_dir)
        print("Download complete.")
    else:
        print("Data already downloaded.")

    # example.nii, needed later for whole-brain visualization
    example_nii_url = (
        "https://github.com/Neural-Dynamics-of-Visual-Cognition-FUB/"
        "Algonauts2021_devkit/raw/main/example.nii"
    )
    example_nii_path = os.path.join(data_dir, "example.nii")
    if not os.path.exists(example_nii_path):
        r = requests.get(example_nii_url, allow_redirects=True)
        with open(example_nii_path, "wb") as fh:
            fh.write(r.content)


def _load_pickle(filename):
    """Loads a pickle file saved with Python 2 (latin1 encoding)."""
    with open(filename, "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


def get_fmri(fmri_dir, roi):
    """
    Loads fMRI data for a given ROI.

    Parameters
    ----------
    fmri_dir : str
        Path to the subject's fMRI data directory.
    roi : str
        Name of the ROI (e.g. "FFA", "V1", "STS", "WB").

    Returns
    -------
    np.ndarray
        Matrix of shape (num_videos, num_voxels), averaged across
        repetitions.
    np.ndarray, optional
        Voxel mask, only returned when roi == "WB".
    """
    roi_file = os.path.join(fmri_dir, roi + ".pkl")
    roi_data = _load_pickle(roi_file)

    # average across repetitions
    roi_data_train = np.mean(roi_data["train"], axis=1)

    if roi == "WB":
        voxel_mask = roi_data["voxel_mask"]
        return roi_data_train, voxel_mask

    return roi_data_train


def load_all_rois(fmri_dir, subject, rois=None):
    """
    Loads fMRI data for all regions of interest for a given subject.

    Parameters
    ----------
    fmri_dir : str
        Path to the base fMRI data directory (mini_track).
    subject : str
        Subject folder name, e.g. "sub01".
    rois : list of str, optional
        Which ROIs to load. Defaults to all 9 standard ROIs.

    Returns
    -------
    dict
        Maps ROI name -> np.ndarray of shape (num_videos, num_voxels)
    """
    if rois is None:
        rois = ["V1", "V2", "V3", "V4", "LOC", "EBA", "FFA", "STS", "PPA"]

    sub_dir = os.path.join(fmri_dir, subject)
    return {roi: get_fmri(sub_dir, roi) for roi in rois}

def get_video_list(video_dir):
    """
    Returns a sorted list of paths to all video files in the given directory.

    Parameters
    ----------
    video_dir : str
        Path to the folder containing the Algonauts video clips.

    Returns
    -------
    list of str
        Sorted list of full paths to .mp4 video files.
    """
    video_files = [
        os.path.join(video_dir, f)
        for f in os.listdir(video_dir)
        if f.endswith(".mp4")
    ]
    video_files.sort()
    return video_files


def load_video_frames(video_path, num_frames=8):
    """
    Loads a fixed number of evenly spaced frames from a video file.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    num_frames : int
        Number of frames to sample evenly across the video.

    Returns
    -------
    np.ndarray
        Array of shape (num_frames, height, width, 3), dtype uint8.
    """
    from decord import VideoReader
    from decord import cpu

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # evenly spaced frame indices across the whole clip
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = vr.get_batch(indices).asnumpy()

    return frames