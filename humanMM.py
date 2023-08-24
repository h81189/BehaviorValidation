from dappy import read, write
from dappy import visualization as vis
import numpy as np

# from IPython.display import Video
from pathlib import Path
import matplotlib.pyplot as plt
from dappy import preprocess
from dappy import write
from dappy import features
from dappy import DataStruct as ds
from dappy.embed import Embed
from dappy.embed import Watershed

analysis_key = "NUCLA"
config = read.config("./configs/" + analysis_key + ".yaml")

pose, ids = read.pose_h5(config["data_path"] + analysis_key + ".h5")
# pose=np.delete(pose,22572)
# pose=np.delete(pose,26767)
# ids=np.delete(ids,[22572,26767])
connectivity = read.connectivity(
    path=config["skeleton_path"], skeleton_name=config["skeleton_name"]
)
meta, meta_by_frame = read.meta(config["data_path"] + analysis_key + "meta.csv", id=ids)
# meta_by_frame=np.delete(meta_by_frame,[22572,26767])

import pdb

pdb.set_trace()

pose = preprocess.rotate_spine(
    preprocess.center_spine(pose, keypt_idx=config["spineM"]),
    keypt_idx=[config["spineM"], config["spineF"]],
)

angles, angle_labels = features.get_angles(pose, connectivity.angles)
ego_pose, ego_pose_labels = features.get_ego_pose(pose, connectivity.joint_names)
feats = np.concatenate([ego_pose, angles], axis=1)
labels = ego_pose_labels + angle_labels
del ego_pose, angles, ego_pose_labels, angle_labels
write.features_h5(feats, labels, path=config["out_path"] + "postural_feats.h5")

pc_feats, pc_labels = features.pca(
    feats, labels, categories=["ego_euc", "ang"], n_pcs=8, method="fbpca"
)
del feats, labels

wlet_feats, wlet_labels = features.wavelet(
    pc_feats, pc_labels, ids, sample_freq=30, freq=np.linspace(0.5, 5, 25) ** 2, w0=5
)

pc_wlet, pc_wlet_labels = features.pca(
    wlet_feats,
    wlet_labels,
    categories=["wlet_ego_euc", "wlet_ang"],
    n_pcs=8,
    method="fbpca",
)

del wlet_feats, wlet_labels
pc_feats = np.hstack((pc_feats, pc_wlet))
pc_labels += pc_wlet_labels
del pc_wlet, pc_wlet_labels

write.features_h5(
    pc_feats, pc_labels, path="".join([config["out_path"], "pca_feats.h5"])
)

data_obj = ds.DataStruct(
    pose=pose, id=ids, meta=meta, meta_by_frame=meta_by_frame, connectivity=connectivity
)

data_obj.features = pc_feats
data_obj = data_obj[:: config["downsample"], :]

embedder = Embed(
    embed_method=config["single_embed"]["method"],
    perplexity=config["single_embed"]["perplexity"],
    lr=config["single_embed"]["lr"],
)
data_obj.embed_vals = embedder.embed(data_obj.features, save_self=True)

# Watershed clustering
data_obj.ws = Watershed(
    sigma=config["single_embed"]["sigma"], max_clip=1, log_out=True, pad_factor=0.05
)
data_obj.data.loc[:, "Cluster"] = data_obj.ws.fit_predict(data=data_obj.embed_vals)

# Plot density
vis.density(
    data_obj.ws.density,
    data_obj.ws.borders,
    filepath=config["out_path"] + "density.png",
    show=True,
)

vis.density_cat(
    data=data_obj,
    column="action",
    watershed=data_obj.ws,
    n_col=4,
    filepath=config["out_path"] + "density_act.png",
    show=True,
)
