#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--images',required=True,help='uint8 N,H,W,C .npy')
ap.add_argument('--train-labels',required=True)
ap.add_argument('--fine-labels',default='')
ap.add_argument('--sample-ids',default='')
ap.add_argument('--output',required=True)
args=ap.parse_args()
images=np.load(args.images); labels=np.load(args.train_labels).astype(np.int64)
if images.dtype!=np.uint8 or images.ndim!=4: raise ValueError('images must be uint8 [N,H,W,C]')
if len(images)!=len(labels): raise ValueError('length mismatch')
kwargs={'images':images,'train_labels':labels}
if args.fine_labels:
    fine=np.load(args.fine_labels).astype(np.int64)
    if len(fine)!=len(labels): raise ValueError('fine label length mismatch')
    kwargs['fine_labels']=fine
if args.sample_ids:
    ids=np.load(args.sample_ids).astype(str)
else:
    ids=np.array([hashlib.sha256(images[i].tobytes()+labels[i].tobytes()).hexdigest() for i in range(len(labels))])
kwargs['sample_ids']=ids
out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(out,**kwargs)
ids_hash=hashlib.sha256('\n'.join(ids.tolist()).encode()).hexdigest()
meta={'path':str(out.resolve()),'num_samples':len(labels),'num_classes':int(labels.max())+1,
      'sample_ids_sha256':ids_hash,'file_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
      'created_at':datetime.now(timezone.utc).isoformat(),
      'training_label_semantics':'Must be documented by the locked experiment exporter.',
      'fine_labels_training_use':False}
out.with_suffix('.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
print(json.dumps(meta,indent=2))
