from features import *
import DataStruct as ds
import visualization as vis
import interface as itf
import numpy as np
import pandas as pd
from embed import Watershed
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import random

algo='fitsne'
# algo='umap'

pstruct = ds.DataStruct(config_path = '/hpc/group/tdunn/hk276/tangosave/configs/path_configs/NUCLA_datastruct_config.yaml')
# pstruct = ds.DataStruct(config_path = '/hpc/group/tdunn/hk276/tangosave/configs/path_configs/NTU60_datastruct_config.yaml')
# pstruct = ds.DataStruct(config_path = '/hpc/group/tdunn/hk276/tangosave/configs/path_configs/NTU120_datastruct_config.yaml')
pstruct.load_connectivity()
pstruct.load_pose()

pc_feats, pc_labels = read_h5(path = '/hpc/group/tdunn/hk276/tangosave/src/NUCLAcapture/pca_feats.h5')
# pc_feats, pc_labels = read_h5(path = '/hpc/group/tdunn/hk276/tangosave/src/NTU60view/pca_feats.h5')
# pc_feats, pc_labels = read_h5(path = '/hpc/group/tdunn/hk276/tangosave/src/NTU120view/pca_feats.h5')
pc_feats=np.float32(pc_feats[:,[0,1,2,3,4,8,9,10,11,12,16,17,18,19,20,24,25,26,27,28]])

params = itf.read_params_config(config_path = '../configs/fitsne.yaml')
params['label']=algo
params['single_embed']['method']=algo
params['downsample']=1

print("downsampling")

pstruct.features = pc_feats[::params['downsample']]
pstruct.frame_id = np.arange(0,pc_feats.shape[0],params['downsample'])
# print(len(pstruct.exp_id))
print(pstruct.exp_ids_full[::params['downsample']])
pstruct.exp_id = pstruct.exp_ids_full[::params['downsample']]
pstruct.downsample = params['downsample']
pstruct.load_meta()

print("done downsampling")

params_config = '../configs/fitsne.yaml'
params = itf.params_process(params_config)
pstruct, embedder = itf.embed_pipe(pstruct, params, save_embedder = params['save_embedder'])

# import pdb;pdb.set_trace()

rf_avg=[[] for i in range(2)]
rf_std=[[] for i in range(2)]

num=[10,3]

meta=pd.read_csv("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/NUCLAmeta.csv")
# meta=pd.read_csv("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/NTU60ViewFullMeta.csv")
# meta=pd.read_csv("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/NTU120ViewFullMeta.csv")

numclusters=[]
# for sig in [round(i**4,2) for i in np.arange(2**0.25,25**0.25,0.05)]: # loop across sigma values
number=16
sigrange=[round(i**number,2) for i in np.arange(8**(1/number),50**(1/number),0.15/number)]
sigrange.reverse()
for sig in [8]:
# for sig in np.arange(30,15,-1): # loop across sigma values

    # act_covs=[[] for i in range(60)]
    cov=[]
    act_covs=[]
    person_covs=[[] for i in range(10)] # nth row has all indices of the nth person's covs in the 
    view_covs=[[] for i in range(3)]
    scene_covs=[[] for i in range(17)]

    pstruct = itf.watershed_cluster(pstruct,
                            sigma = sig,
                            column = params['density_by_column'],
                            plots_label = "final",
                            save_ws = True)

    vis.density(pstruct.ws.density, pstruct.ws.borders, show=True,
                filepath = ''.join([pstruct.out_path,'sigma_'+str(sig)+'_density.png']))

    cmax=max(pstruct.data.Cluster)
    numclusters.append(len(np.unique(pstruct.data.Cluster)))
    vidstarts=np.append(np.unique(pstruct.data.exp_id,return_index=True)[1],-1)
    leng=len(pstruct.data.exp_id)
    for i in range(len(vidstarts)-1): # loop across each video to get COV, sort into action cov
        curr=np.unique(pstruct.data.Cluster[vidstarts[i]:vidstarts[i+1]], return_counts=True)
        freqs=curr[1]
        pos=curr[0]
        freqs=freqs/sum(freqs)
        t=np.zeros(cmax)
        t.put(pos-1,freqs)
        t=t[np.unique(pstruct.data.Cluster)-1]
        cov.append(t)
        act_covs.append(pstruct.data.action[vidstarts[i]])
        person_covs[meta['person'][i]-1].append(i)
        scene_covs[meta['take'][i]-1].append(i)
        view_covs[meta['view'][i]-1].append(i)
        # view_covs[pstruct.data.action[vidstarts[i]]-1].append(meta['view'][i])

    acts=np.array(act_covs)
    cov=np.array(cov)
    comb=[person_covs,view_covs]
    
    for j in range(2):
        rfacc=[]
        for i in range(num[j]):
            X_test=cov[comb[j][i]]
            y_test=acts[comb[j][i]]
            X_train=np.delete(cov,comb[j][i],axis=0)
            y_train=np.delete(acts,comb[j][i],axis=0)

            rf = RandomForestClassifier(n_estimators = 100)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            # rfacc.append(metrics.accuracy_score(y_test, y_pred))
            rfacc.append(metrics.cohen_kappa_score(y_test, y_pred))

        rf_avg[j].append(np.average(rfacc))
        rf_std[j].append(np.std(rfacc))

    print(rf_avg)
    print(rf_std)
    print(numclusters)

print(algo)
print("NUCLA accuracies, using "+algo)