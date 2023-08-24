import numpy as np
import DataStruct as ds
import matplotlib.pyplot as plt
import pandas as pd

pstruct = ds.DataStruct(config_path = '/hpc/group/tdunn/hk276/tangosave/configs/path_configs/NTU60_datastruct_config.yaml')
pstruct.load_connectivity()
pstruct.load_pose()

act_class=["drink water","eat meal/snack","brushing teeth","brushing hair","drop","pickup","throw","sitting down","standing up","clapping","reading","writing","tear up paper","wear jacket","take off jacket","wear a shoe","take off a shoe","wear on glasses","take off glasses","put on a hat/cap","take off a hat/cap","cheer up","hand waving","kicking something","reach into pocket","hopping","jump up","phone call","play on phone","type on a keyboard","point to something","take a selfie","check watch","rub hands together","nod head/bow","shake head","wipe face","salute","put palms together","cross hands in front","sneeze/cough","staggering","falling","headache","stomachache/heart pain","backache","neck ache","nausea or vomiting","use fan/feel warm","punching other person","kicking other person","pushing other person","pat back of other person","point at person","hugging other person","giving something to other person","touch other person\'s pocket","handshaking","walking towards each other","walking apart","put on headphone","take off headphone","shoot at the basket","bounce ball","tennis bat swing","juggling table tennis balls","hush (quite)","flick hair","thumb up","thumb down","make ok sign","make victory sign","staple book","counting money","cutting nails","cutting paper (scissors)","snapping fingers","open bottle","sniff (smell)","squat down","toss a coin","fold paper","ball up paper","play magic cube","apply cream on face","apply cream on hand back","put on bag","take off bag","put something into bag","take something out of bag","open a box","move heavy objects","shake fist","throw up cap/hat","hands up (both hands)","cross arms","arm circles","arm swings","running on the spot","butt kicks (kick backward)","cross toe touch","side kick","yawn","stretch oneself","blow nose","hit other person with something","wield knife towards other person","knock over other person (hit with body)","grab other person’s stuff","shoot at other person with a gun","step on foot","high-five","cheers and drink","carry something with other person","take a photo of other person","follow other person","whisper in other person’s ear","exchange things with other person","support somebody with hand","finger-guessing game (playing rock-paper-scissors)"]
meta=pd.read_csv("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/NTU60ViewFullMeta.csv")

# import pdb;pdb.set_trace()


data=pstruct
connectivity=pstruct.connectivity
fps=30
frames=[430,120345,541323,2261381,923421,3524283,2061159,2111171,1532324,1800002]

#4058507 max
N_FRAMES=100
name='10acts.png'
offset= [0,0],
title= "3D Human Pose Dataset",
fsize= 12,
aspect= [0.6,0.6,1.5]
SAVE_ROOT=pstruct.out_path+'videos/'

preds = data.pose_3d

COLOR = connectivity.colors
links = connectivity.links
links_expand = links
# total_frames = N_FRAMES*len(frames)#max(np.shape(f[list(f.keys())[0]]))

## Expanding connectivity for each frame to be visualized
num_joints = max(max(links))+1
next_con = [(x+num_joints, y+num_joints) for x,y in links]
links_expand=links_expand+next_con

offset=offset[0]
# get dannce predictions

fig = plt.figure(figsize=(12, 6))
# fig, axes = plt.subplots(1, 2)
# for i in range(len(axes)):

for i in range(10):
    # ax_3d=axes[i]
    ax_3d = fig.add_subplot(2, 5, i+1, projection='3d')

    pose_3d =preds[frames[i],:,:]
    x_lim1, x_lim2 = np.min(pose_3d[:, 0])-offset[0], np.max(pose_3d[:, 0])+offset[0]
    y_lim1, y_lim2 = np.min(pose_3d[:, 1])-offset[0], np.max(pose_3d[:, 1])+offset[0]
    z_lim1, z_lim2 = np.min(pose_3d[:, 2])-offset[1], np.max(pose_3d[:, 2])+offset[1]
    kpts_3d=pose_3d

    # plot 3d moving skeletons
    ax_3d.scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2],  marker='.', color='black', linewidths=0.5)
    for color, (index_from, index_to) in zip(COLOR, links_expand):
        xs, ys, zs = [np.array([kpts_3d[index_from, j], kpts_3d[index_to, j]]) for j in range(3)] 
        ax_3d.plot3D(xs, ys, zs, c=color, lw=2)

    ax_3d.set_xlim(x_lim1, x_lim2)
    ax_3d.set_ylim(y_lim1, y_lim2)
    ax_3d.set_zlim(z_lim1, z_lim2)
    # import pdb;pdb.set_trace()
    ax_3d.set_title(act_class[meta.iloc[pstruct.exp_ids_full[frames[i]]]['action']-1], fontsize=12)
    ax_3d.set_box_aspect(aspect)
    plt.axis('off')


fig.suptitle('NTU RGB+D 60: 3D Human Pose Samples', fontsize=20)
plt.savefig(SAVE_ROOT+name,dpi=1600)
