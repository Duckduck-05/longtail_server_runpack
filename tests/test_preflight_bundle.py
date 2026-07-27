import hashlib, json
from pathlib import Path
import numpy as np
from ltx.config import LoadedCampaign, Task
from ltx.preflight import _check_semantic_bundle


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ids_hash(ids): return hashlib.sha256("\n".join(ids.tolist()).encode()).hexdigest()


def test_semantic_bundle_certifies_matched_permutation(tmp_path):
    root=Path(__file__).resolve().parents[1]
    labels=np.repeat(np.arange(2),4); ids=np.array([f"i{i}" for i in range(8)])
    images=np.zeros((8,4,4,3),dtype=np.uint8); manifest=tmp_path/'m.npz'
    np.savez_compressed(manifest,images=images,train_labels=labels,sample_ids=ids,fine_labels=np.tile(np.arange(4),2))
    manifest.with_suffix('.json').write_text(json.dumps({'sample_ids_sha256':ids_hash(ids),'file_sha256':sha(manifest)}))
    weights={
      'oracle':np.array([2,1,.7,.3,2,1,.7,.3]),
      'predictive':np.array([1.6,1.2,.8,.4,1.6,1.2,.8,.4]),
      'pointfit':np.array([1.5,1.1,.9,.5,1.5,1.1,.9,.5]),
      'permutation':np.array([.8,1.6,.4,1.2,1.2,.4,1.6,.8]),
    }
    tasks=[]
    for method in ['lt',*weights]:
        mc={'name':method}
        if method=='lt': mc['generated_weight']='uniform_manifest'
        else:
            p=tmp_path/f'{method}.npy'; np.save(p,weights[method]); mc['weight_file']=str(p)
            side={'dataset_name':'m','dataset_fingerprint':sha(manifest),'sample_ids_sha256':ids_hash(ids),'num_samples':8,
                  'weights_file':str(p),'weights_sha256':sha(p),'method':method,'normalization':'mean_one',
                  'effective_sample_size':float(weights[method].sum()**2/(weights[method]**2).sum()),
                  'representation':'DINOv2','K':4,'fine_labels_used_for_training':method=='oracle'}
            p.with_suffix('.json').write_text(json.dumps(side))
        tasks.append(Task(id=method,campaign='c',stage='decisive_semantic_gate',adapter='coral',method=method,seed=0,priority=1,
            dataset={'frozen_manifest':str(manifest)},train={},eval={},method_config=mc,repository={},runtime={},retry={},
            semantic_eval_command='x {samples} {labels} {output} {manifest} {method} {seed}',run_dir=str(tmp_path/method)))
    campaign=LoadedCampaign(root=root,config_path=root/'configs/deadline_full.yaml',raw={},server={},tasks=tasks)
    checks=_check_semantic_bundle(campaign,tasks)
    assert not [c for c in checks if c.level=='ERROR'], [(c.name,c.message) for c in checks]
