from features import *
import DataStruct as ds
import visualization as vis
import interface as itf
import numpy as np
import time
from IPython.display import Video
from embed import Watershed
import copy
from typing import Optional, Union, List

DATASET="NTU120"
# DATASET="NTU60"
# DATASET="PKU"
# DATASET="NUCLA"
# 286, 287 in NUCLA have 1 frame...

print(DATASET)

pstruct = ds.DataStruct(config_path = '/hpc/group/tdunn/hk276/tangosave/configs/path_configs/'+DATASET+'_datastruct_config.yaml')
pstruct.load_connectivity()
pstruct.load_pose()

import pdb;pdb.set_trace();

# abs_vel, abs_vel_labels  = get_velocities(pstruct.pose_3d, 
#                                           pstruct.exp_ids_full, 
#                                           pstruct.connectivity.joint_names,
#                                           joints=[3,1,0],
#                                           widths=[3,15,29],
#                                           sample_freq=30)

# save_h5(abs_vel, abs_vel_labels, path = ''.join([pstruct.out_path,'abs_vel.h5']))
# abs_vel, abs_vel_labels = read_h5(path = ''.join([pstruct.out_path,'abs_vel.h5']))

# Centering all joint locations to mid-spine
pose_locked = center_spine(pstruct.pose_3d,joint_idx=1)

# Rotates front spine to xz axis
if DATASET=="NUCLA":
  pose_locked = rotate_spine(pose_locked,joint_idx=[1,2])
else:
  pose_locked = rotate_spine(pose_locked,joint_idx=[1,20])

# vis.skeleton_vid3D(pose_locked,
#                    pstruct.connectivity,
#                    frames=[6713],
#                    N_FRAMES = 200,
#                    VID_NAME='jointissue.mp4',
#                    offset=[1,1],
#                    SAVE_ROOT=pstruct.out_path)

# Getting relative velocities
# rel_vel, rel_vel_labels = get_velocities(pose_locked,
#                                          pstruct.exp_ids_full, 
#                                          pstruct.connectivity.joint_names,
#                                          joints=np.delete(np.arange(25),1),
#                                          widths=[3,15,29],
#                                          sample_freq=30)


# save_h5(rel_vel, rel_vel_labels, path = ''.join([pstruct.out_path,'rel_vel.h5']))
# rel_vel, rel_vel_labels = read_h5(path = ''.join([pstruct.out_path,'rel_vel.h5']))

angles, angle_labels = get_angles(pose_locked,
                                  pstruct.connectivity.angles)

np.nan_to_num(angles,copy=False,nan=0)

save_h5(angles, angle_labels, path = ''.join([pstruct.out_path,'angles.h5']))
# angles, angle_labels = read_h5(path = ''.join([pstruct.out_path,'angles.h5']))

# ang_vel, ang_vel_labels = get_angular_vel(angles,
#                                           angle_labels,
#                                           pstruct.exp_ids_full,
#                                           sample_freq=30)

# save_h5(ang_vel, ang_vel_labels, path = ''.join([pstruct.out_path,'ang_vel.h5']))
# ang_vel, ang_vel_labels = read_h5(path = ''.join([pstruct.out_path,'ang_vel.h5']))


ego_pose, ego_pose_labels  = get_ego_pose(pose_locked,
                                          pstruct.connectivity.joint_names)

save_h5(ego_pose, ego_pose_labels, path = ''.join([pstruct.out_path,'ego_pose.h5']))
# ego_pose, ego_pose_labels = read_h5(path = ''.join([pstruct.out_path,'ego_pose.h5']))

# features = np.concatenate([abs_vel, rel_vel, ego_pose, angles, ang_vel], axis=1)
# labels = abs_vel_labels + rel_vel_labels + ego_pose_labels + angle_labels + ang_vel_labels

features = np.concatenate([ego_pose, angles], axis=1)
labels = ego_pose_labels + angle_labels

# Clear memory
# del pose_locked, abs_vel, rel_vel, ego_pose, angles, ang_vel
# del abs_vel_labels, rel_vel_labels, ego_pose_labels, angle_labels, ang_vel_labels
del pose_locked, ego_pose, angles
del ego_pose_labels, angle_labels

save_h5(features, labels, path = ''.join([pstruct.out_path,'postural_feats.h5']))
# features, labels =  read_h5(path = ''.join([pstruct.out_path,'postural_feats.h5']))

pc_feats, pc_labels = pca(features,
                          labels,
                          categories = ['ego_euc','ang'],
                        #   categories = ['abs_vel','rel_vel','ego_euc','ang','avel'],
                          n_pcs = 8,
                          method = 'fbpca')
# # print("PCA time: " + str(time.time() - t))

del features, labels

save_h5(pc_feats, pc_labels, path = ''.join([pstruct.out_path,'pc_feats.h5']))
# pc_feats, pc_labels = read_h5(path = ''.join([pstruct.out_path,'pc_feats.h5']))


wlet_feats, wlet_labels = wavelet(pc_feats, 
                                  pc_labels, 
                                  pstruct.exp_ids_full,
                                  sample_freq = 30,
                                  # freq = np.linspace(1,25,25),
                                  freq = np.linspace(0.5,5,25)**2,
                                  w0 = 5)

save_h5(wlet_feats, wlet_labels, path = ''.join([pstruct.out_path,'kinematic_feats.h5']))
# wlet_feats, wlet_labels = read_h5(path = ''.join([pstruct.out_path,'kinematic_feats.h5']))

pc_wlet, pc_wlet_labels = pca(wlet_feats,
                              wlet_labels,
                              categories = ['wlet_ego_euc','wlet_ang'],
                            #   categories = ['wlet_abs_vel','wlet_rel_vel','wlet_ego_euc','wlet_ang','wlet_avel'],
                              n_pcs = 8,
                              method = 'fbpca')
del wlet_feats, wlet_labels
pc_feats = np.hstack((pc_feats, pc_wlet))
pc_labels += pc_wlet_labels
del pc_wlet, pc_wlet_labels

### check angle issue
### get rid of abs and rel vel and avel in pc feats
### psutil for memory
### random forest - sklearn
### pytorch linear layer
### KNNEmbed in embed.py
### histogram for clusters within actions
### sigma value in fitsne.yaml - higher = fewer cluster

save_h5(pc_feats, pc_labels, path = ''.join([pstruct.out_path,'pca_feats.h5']))
# pc_feats, pc_labels = read_h5(path = ''.join([pstruct.out_path,'pca_feats.h5']))

print("start")

params = itf.read_params_config(config_path = '../configs/fitsne.yaml')

print("downsampling")

pstruct.features = pc_feats[::params['downsample']]
pstruct.frame_id = np.arange(0,pc_feats.shape[0],params['downsample'])
# print(len(pstruct.exp_id))
print(pstruct.exp_ids_full[::params['downsample']])
pstruct.exp_id = pstruct.exp_ids_full[::params['downsample']]
pstruct.downsample = params['downsample']
pstruct.load_meta()

# import pdb; pdb.set_trace();

# Run the rest of the analysis
pstruct = itf.run_analysis(params_config = '../configs/fitsne.yaml',
                 ds = pstruct)

vis.scatter(pstruct.embed_vals, show=True, 
            filepath=''.join([pstruct.out_path, 'final_scatter.png']))

vis.density(pstruct.ws.density, pstruct.ws.borders, show=True,
            filepath = ''.join([pstruct.out_path,'final_density.png']))

# k=0
# currvid=[]
# act_covs=[[] for i in range(120)]
# cov=[]
# cmax=max(pstruct.data.Cluster)
# for i in range(len(pstruct.data.exp_id)):
#     print(i)
#     if pstruct.data.exp_id[i]!=k:
#         cov=[]
#         for j in range(1,cmax+1):
#             print(j)
#             cov.append(currvid.count(j)/len(currvid))
#         act_covs[pstruct.data.action[i-1]-1].append(cov)
#         k=k+1
#     currvid.append(pstruct.data.Cluster[i])


import pdb;pdb.set_trace();
act_covs=[[] for i in range(120)]
cmax=max(pstruct.data.Cluster)
vidstarts=np.append(np.unique(pstruct.data.exp_id,return_index=True)[1],-1)
leng=len(pstruct.data.exp_id)
for i in range(len(vidstarts)-1):
    print(i)
    curr=np.unique(pstruct.data.Cluster[vidstarts[i]:vidstarts[i+1]], return_counts=True)
    freqs=curr[1]
    pos=curr[0]
    freqs=freqs/sum(freqs)
    t=np.zeros(cmax)
    t.put(pos-1,freqs)
    act_covs[pstruct.data.action[vidstarts[i]]-1].append(t)

import pdb;pdb.set_trace();
print(act_covs)

np.save("/hpc/group/tdunn/hk276/tangosave/src/action_cluster_occupancy_vec.npy", act_covs)

# import pdb; pdb.set_trace();

vis.density_cat(data=pstruct, column='take', watershed=pstruct.ws, n_col=2, show=True,
                filepath = ''.join([pstruct.out_path,'density_by_take.png']))
vis.density_cat(data=pstruct, column='scene', watershed=pstruct.ws, n_col=4, show=True,
                filepath = ''.join([pstruct.out_path,'density_by_scene.png']))
vis.density_cat(data=pstruct, column='person', watershed=pstruct.ws, n_col=8, show=True,
                filepath = ''.join([pstruct.out_path,'density_by_person.png']))
vis.density_cat(data=pstruct, column='action', watershed=pstruct.ws, n_col=6, show=True,
                filepath = ''.join([pstruct.out_path,'density_by_action.png']))

print("done!!")