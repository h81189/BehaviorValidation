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
from tqdm import tqdm

algo='fitsne'
# algo='umap'
d='120' # NTU 120 or 60 here

# import pdb; pdb.set_trace();

print(d)
print(algo)


pstruct = ds.DataStruct(config_path = '/hpc/group/tdunn/hk276/tangosave/configs/path_configs/NTU'+d+'_datastruct_config.yaml')
pstruct.load_connectivity()
pstruct.load_pose()

pc_feats, pc_labels = read_h5(path = '/hpc/group/tdunn/hk276/tangosave/src/NTU'+d+'capture/pca_feats.h5')
pc_feats=np.float32(pc_feats[:,[0,1,2,3,4,8,9,10,11,12,16,17,18,19,20,24,25,26,27,28]])

params = itf.read_params_config(config_path = '../configs/fitsne.yaml')
params['label']=algo
params['single_embed']['method']=algo
params['downsample']=1

print("downsampling")

pstruct.features = pc_feats[::params['downsample']]
pstruct.frame_id = np.arange(0,pc_feats.shape[0],params['downsample'])
print(pstruct.exp_ids_full[::params['downsample']])
pstruct.exp_id = pstruct.exp_ids_full[::params['downsample']]
pstruct.downsample = params['downsample']
pstruct.load_meta()

print("done downsampling")

params_config = '../configs/fitsne.yaml'
params_config = '/hpc/group/tdunn/hk276/tangosave/configs/fitsne.yaml'
params = itf.params_process(params_config)
pstruct, embedder = itf.embed_pipe(pstruct, params, save_embedder = params['save_embedder'])

print("ok")

if d[0]=='6':
    trainsubjects=np.array([1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38])-1
    trainviews=[1,2] # really 2,3
    trainsets=[trainsubjects,trainviews]
    names=['person','view']
elif d[0]=='1':
    trainsubjects=np.array([1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103])-1
    trainscenes=[2*(i+1)-1 for i in range(16)]
    trainsets=[trainsubjects,trainscenes]
    names=['person','scene']
else:
    raise Exception("invalid dataset name")

rf_avg=[[] for i in range(2)]


# meta=pd.read_csv("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/NUCLAmeta.csv")
# meta=pd.read_csv("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/NTU60ViewFullMeta.csv")
meta=pd.read_csv("/hpc/group/tdunn/hk276/CAPTURE_demo/Python_analysis/engine/NTU"+d+"ViewFullMeta.csv")
meta['view']+=1
# import pdb;pdb.set_trace()
numclusters=[]

# for sig in [round(i**4,2) for i in np.arange(2**0.25,25**0.25,0.05)]: # loop across sigma values
num=16
sigrange=[round(i**num,2) for i in np.arange(8**(1/num),50**(1/num),0.15/num)]
sigrange.reverse()

for sig in [8]:
    # act_covs=[[] for i in range(60)]
    cov=[]
    act_covs=[]
    person_covs=[[] for i in range(40)] # nth row has all indices of the nth person's covs in the 
    view_covs=[[] for i in range(3)]
    scene_covs=[[] for i in range(17)]
    if d=='120':
        person_covs=[[] for i in range(106)] # nth row has all indices of the nth person's covs in the 
        view_covs=[[] for i in range(3)]
        scene_covs=[[] for i in range(32)]

    pstruct = itf.watershed_cluster(pstruct,
                            sigma = sig,
                            column = params['density_by_column'],
                            plots_label = "final",
                            save_ws = True)
                    
    # import pdb; pdb.set_trace()

    vis.density(pstruct.ws.density, pstruct.ws.borders, show=True,
                filepath = ''.join([pstruct.out_path,'sigma_'+str(sig)+'_density.png']))

    cmax=max(pstruct.data.Cluster)
    numclusters.append(len(np.unique(pstruct.data.Cluster)))
    # np.save("pstructdata60.npy",pstruct.data,allow_pickle=True)
    vidstarts=np.append(np.unique(pstruct.data.exp_id,return_index=True)[1],-1)
    leng=len(pstruct.data.exp_id)
    for i in tqdm(range(len(vidstarts)-1)): # loop across each video to get COV, sort into action cov :: OPTIMIZE THIS
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
        scene_covs[meta['scene'][i]-1].append(i)
        view_covs[meta['view'][i]-1].append(i)
        # view_covs[pstruct.data.action[vidstarts[i]]-1].append(meta['view'][i])

    acts=np.array(act_covs)
    cov=np.array(cov)
    if d=='60':
        comb=[person_covs,view_covs]
    else:
        comb=[person_covs,scene_covs]

    for k in range(2): # get accuracies for cross-each
        inds=np.concatenate(np.take(comb[k],trainsets[k]))
        X_test=np.delete(cov,inds,axis=0)
        y_test=np.delete(acts,inds,axis=0)
        X_train=cov[inds]
        y_train=acts[inds]
        rf = RandomForestClassifier(n_estimators = 100)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        # rf_avg[k].append(metrics.accuracy_score(y_test, y_pred))
        rf_avg[k].append(metrics.cohen_kappa_score(y_test, y_pred))
    print(rf_avg)
    rftemp=rf_avg.copy()
    rftemp.append(numclusters)
    np.save("./NTU"+d+"results"+algo+"single.npy",np.array(rftemp),allow_pickle=True)
    print(numclusters)
print(d)
print(algo)
print("NTU"+d+" accuracies, using "+algo)