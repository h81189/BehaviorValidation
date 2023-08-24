import sys
from dappy import DataStruct as ds
from dappy import visualization as vis
import moviepy.editor as mp  # requests, urllib3, charset_normalizer, idna
import numpy as np
from typing import List
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from dappy import read
from dappy import write
import json


def read_NTU(path_to_folder: str, skip_actions: List):
    """Reads in NTU dataset

    Parameters
    ----------
    path_to_folder
        Path to directory in which data files are contained
    skip_actions
        Action labels to not load

    Returns
    -------
    pose_list
        List of pose sequences of shape
        (# frames, # keypoints, 3 xyz coordinates)
    meta_df
        Pandas DataFrame with Scene, Camera, Person, Take, and Action
        metadata for each pose sequence in pose_list
    """
    # Retrieve all file paths in directory
    pathlist = Path(path_to_folder).glob("*skeleton.npy")

    # Initialize pose list and meta DataFrame
    pose_list = []
    meta_df = pd.DataFrame(
        [], columns=["Scene", "View", "Person", "Take", "Action"], dtype=int
    )

    for i, path in enumerate(tqdm.tqdm(list(pathlist))):
        # Parse path name for sample metadata
        filename = path.parts[-1]
        meta = re.split("\D", filename.split(".")[0])[1:]
        meta = [int(ID) for ID in meta]

        # Check metadata and add to pose list and meta DataFrames
        if meta[-1] not in skip_actions:
            data = np.load(path, allow_pickle=True).item()
            pose_list += [data["skel_body0"][:, :, [0, 2, 1]]]
            meta_df.loc[len(meta_df)] = meta

    return pose_list, meta_df


def sort_pose_meta(pose_list, meta_df, by: List, reset_index: bool = False):
    """Sorts list of pose sequences and meta
    DataFrame by indicated categories

    Parameters
    ----------
    pose_list
        List of pose sequences of shape
        (# frames, # keypoints, 3 xyz coordinates)
    meta_df
        Pandas DataFrame with Scene, Camera, Person, Take, and Action
        metadata for each pose sequence in pose_list
    by
        List of column names in `meta_df` by which to sort
    reset_index, optional
        Choice to reset the index of `meta_df`, by default False

    Returns
    -------
    sorted_pose
        List of sorted pose sequences of shape
        (# frames, # keypoints, 3 xyz coordinates)
    sorted_meta
        Sorted Pandas DataFrame with Scene, Camera, Person, R(?), and Action
        metadata for each pose sequence in pose_list
    """
    sorted_meta = meta_df.sort_values(by=by, axis=0)
    sorted_pose = [pose_list[i] for i in sorted_meta.index]

    if reset_index:
        sorted_meta.reset_index(drop=True)

    return sorted_pose, sorted_meta


def NTU_plot_files(
    file_names: List = [],
    name: str = "temp.mp4",
    rgb: bool = False,
    avgsmooth: bool = False,
):
    skels = 0
    if type(file_names[0]) == str:
        file_names = [[i] for i in file_names]
    for i in range(len(file_names)):
        data = []
        for k in range(len(file_names[i])):
            try:
                data.append(
                    np.load(
                        path_to_NTU_skels + file_names[i][k] + ".skeleton.npy",
                        allow_pickle=True,
                    ).item()["skel_body0"][:, :, [0, 2, 1]]
                )
                data[k] = data[k] - np.average(data[k][0, :, :], axis=0)
            except:
                print("file " + file_names[i][k] + " not found")
        if data == []:
            continue
        data = align_skeletons(data)
        length = len(data[0])
        if avgsmooth:
            nvideos = 1
            avg = np.average(data, axis=0)
            dis = 5
            for j in range(len(avg[i]) - 1):
                avg[j] = np.median(avg[j : min(j + dis, len(avg) - 1)], axis=0)
            data = avg
        else:
            nvideos = len(data)
            for j in range(1, len(data)):
                data[0] = np.append(data[0], data[j], axis=0)
            data = data[0]
        vis.pose3D_arena(
            data,
            connectivity=NTUconn,
            # title=act_class[int(file_names[i][0][-3:])-1],
            frames=[int(length / 2) + length * i for i in range(nvideos)],
            N_FRAMES=length - 2,
            fps=30,
            dpi=100,
            VID_NAME=file_names[i][0] + "HumanSkel.mp4",
            SAVE_ROOT=config["out_path"] + "videos/",
        )
        if skels == 0:
            skels = mp.VideoFileClip(
                config["out_path"] + "videos/vis_" + file_names[i][0] + "HumanSkel.mp4"
            )
        else:
            skels = mp.concatenate_videoclips(
                [
                    skels,
                    mp.VideoFileClip(
                        config["out_path"]
                        + "videos/vis_"
                        + file_names[i][0]
                        + "HumanSkel.mp4"
                    ),
                ]
            )
    if rgb:
        clip = mp.VideoFileClip(
            path_to_vids + file_names[0][0][1:4] + "/" + file_names[0][0] + "_rgb.avi"
        )
        for i in range(1, len(file_names)):
            nextclip = mp.VideoFileClip(
                path_to_vids
                + file_names[i][0][1:4]
                + "/"
                + file_names[i][0]
                + "_rgb.avi"
            )
            clip = mp.concatenate_videoclips([clip, nextclip])
        # clip=mp.clips_array([skels,clip])
        clip = mp.CompositeVideoClip(
            [
                skels.set_position(("left", "center")).resize(1),
                clip.set_position(("right", "center")),
            ],
            size=(3000, 1080),
        )
        clip.write_videofile(config["out_path"] + "videos/" + name, codec="libx264")
    else:
        skels.write_videofile(config["out_path"] + "videos/" + name, codec="libx264")


# use pillow 9.0.1


def align_skeletons(pose: List, centerjoint: int = 1, alignframes=5):
    """rotates and aligns skeletons from all camera views

    Parameters
    ----------
    pose
        List of pose sequences of shape
        (# camera views, # frames, # keypoints, 3 xyz coordinates)
    centerjoint
        joint number to set as origin for all views
    alignframes
        number of frames at beginning and end of sequence to average the alignment over

    Returns
    -------
    pose
        pose sequence of the input sequences aligned and averaged into one. Has shape
        (# frames (lowest of 3 input views), # keypoints, 3 xyz coordinates)
    """
    min_sequence_length = min([len(pose[i]) for i in range(len(pose))])
    pose = [
        # i[0:minlen] - np.average(np.average(i[:, :, :], axis=1), axis=0) for i in pose
        i[0:min_sequence_length] - i[0, centerjoint, :]
        for i in pose
    ]
    for i in range(1, len(pose)):
        rotation_matrix = []
        for j in range(-alignframes, alignframes):
            rotation_matrix.append(Rotation.align_vectors(pose[0][j], pose[i][j])[0])
        rotation_matrix = [i.as_matrix() for i in rotation_matrix]
        rotation_matrix = np.average(rotation_matrix, axis=0)
        rotation_matrix = Rotation.from_matrix(rotation_matrix)
        for j in range(min_sequence_length):
            pose[i][j] = rotation_matrix.apply(pose[i][j])
    return pose


def get_raw_NUCLA(fname):
    files = os.listdir(path_to_NUCLA_skels)
    meta = []
    NUCLAfull = []
    for fi in tqdm(files):
        f = open(path_to_NUCLA_skels + fi)
        NUCLAfull.append(np.array(json.load(f)["skeletons"]))
        f.close()
        meta.append(
            [int(fi[1:3]), int(fi[5:7]), int(fi[9:11]), int(fi[13:15])]
        )  # scene, person, take, camera, action
    NUCLAfull = np.array(NUCLAfull)
    meta = np.array(meta)
    # np.save(config['out_path']+fname+"meta.npy", meta)
    with open(config["out_path"] + fname + "meta.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=",")
        csvWriter.writerow(["action", "person", "scene", "view"])
        csvWriter.writerows(meta)
    ids = np.concatenate([np.full(len(NUCLAfull[i]), i) for i in range(len(NUCLAfull))])
    write.pose_h5(np.concatenate(NUCLAfull), ids, config["out_path"] + fname + ".h5")
    return [NUCLAfull, meta]


def get_raw_NTU(fname, NTU60=False):
    files = os.listdir(path_to_NTU_skels)
    meta = []
    NTUfull = []
    for fi in tqdm(files):
        if int(fi[17:20]) in skipA:
            continue
        if NTU60 and int(fi[17:20]) > 60:
            continue
        data = np.load(path_to_NTU_skels + fi, allow_pickle=True).item()["skel_body0"][
            :, :, [0, 2, 1]
        ]
        NTUfull.append(data)
        meta.append(
            [int(fi[1:4]), int(fi[9:12]), int(fi[13:16]), int(fi[5:8]), int(fi[17:20])]
        )  # scene, person, take, camera, action
    NTUfull = np.array(NTUfull)
    meta = np.array(meta)
    # np.save(config['out_path']+fname+"meta.npy", meta)
    with open(config["out_path"] + fname + "meta.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=",")
        csvWriter.writerow(["scene", "person", "take", "view", "action"])
        csvWriter.writerows(meta)
    ids = np.concatenate([np.full(len(NTUfull[i]), i) for i in range(len(NTUfull))])
    write.pose_h5(np.concatenate(NTUfull), ids, config["out_path"] + fname + ".h5")
    return [NTUfull, meta]


def get_aligned_NTU(fname, NTU60=False):
    files = os.listdir(path_to_NTU_skels)
    meta = []
    NTUfull = {}
    newdata = []
    for fi in tqdm(files):
        if int(fi[17:20]) in skipA:
            continue
        if NTU60 and int(fi[17:20]) > 60:
            continue
        data = np.load(path_to_NTU_skels + fi, allow_pickle=True).item()["skel_body0"][
            :, :, [0, 2, 1]
        ]
        data = data - np.average(data[0, :, :], axis=0)
        if fi[:4] + fi[8:] not in NTUfull:
            NTUfull[fi[:4] + fi[8:]] = []
        NTUfull[fi[:4] + fi[8:]].append(data)
    for key in NTUfull.keys():
        if len(NTUfull[key]) != 3:
            continue
        data = align_skeletons(NTUfull[key])
        newdata.append(list(data))
        meta.append([int(key[1:4]), int(key[5:8]), int(key[9:12]), int(key[13:16])])
    avg = np.average(newdata, axis=1)
    dis = 5
    for i in range(len(avg)):
        for j in range(len(avg[i]) - 1):
            avg[i][j] = np.median(avg[i][j : min(j + dis, len(avg[i]) - 1)], axis=0)
    # np.save(config['out_path']+fname+"meta.npy", np.array(meta))
    with open(config["out_path"] + fname + "meta.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=",")
        csvWriter.writerow(["scene", "person", "take", "action"])
        csvWriter.writerows(meta)
    ids = np.concatenate([np.full(len(avg[i]), i) for i in range(len(avg))])
    # np.save(config['out_path']+fname, avg)
    write.pose_h5(np.concatenate(avg), ids, config["out_path"] + fname + ".h5")
    return [avg, meta]


def joint_err(NTUfull, save=False):
    avg = np.average(NTUfull, axis=1)
    view = np.array([NTUfull[:, i] - avg for i in range(3)])
    view = view * view
    view = np.sum(view, axis=0) / 3
    for j in range(len(view)):
        view[j] = np.sum(view[j], axis=2)
    view = np.array([np.average(view[i], axis=0) for i in range(len(view))])
    np.shape(view)
    jointavg = np.average(view, axis=0)
    print(jointavg)
    if save:
        np.save(config["out_path"] + "Errors.npy", view)


if __name__ == "__main__":
    config = read.config("./configs/extract_dataset.yaml")

    NTUconn = read.connectivity(
        path=config["skeleton_path"], skeleton_name=config["NTUskeleton"]
    )
    act_class = config["act_class"]
    path_to_NTU_skels = config["path_to_NTU_skels"]
    path_to_NUCLA_skels = config["path_to_NUCLA_skels"]
    path_to_vids = config["path_to_vids"]
    skipA = config["skip_action"]

    config = read.config(
        "/hpc/group/tdunn/hk276/validation/configs/extract_dataset.yaml"
    )

    pose_list, meta_df = read_NTU(
        config["path_to_NTU_skels"], skip_actions=config["skip_action"]
    )

    pose_list, meta_df = sort_pose_meta(
        pose_list, meta_df, by=["A", "S", "P", "R", "C"]
    )

    config = read.config("eheuh")
