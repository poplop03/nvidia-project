#!/usr/bin/env python3
"""Encode image groups with OpenAI CLIP in one GPU process."""
import argparse, hashlib, json
from pathlib import Path
import clip
import numpy as np
import torch
from PIL import Image

p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--model',default='ViT-B-16'); p.add_argument('--device',default='cuda'); a=p.parse_args()
name={'ViT-B-16':'ViT-B/16','ViT-B-32':'ViT-B/32'}.get(a.model,a.model)
model,preprocess=clip.load(name,device=a.device,jit=False); model.eval()
data=json.loads(a.manifest.read_text()); a.output_dir.mkdir(parents=True,exist_ok=True)
weight=Path.home()/'.cache/clip'/(a.model+'.pt')
# clip cache filename is URL basename, normally ViT-B-16.pt
if not weight.exists(): weight=Path.home()/'.cache/clip'/'ViT-B-16.pt'
digest=hashlib.sha256(weight.read_bytes()).hexdigest() if weight.exists() else None
meta={'model_name':a.model,'model_library_name':name,'model_version':f'openai-clip-{digest[:12]}' if digest else 'openai-clip','model_sha256':digest,'device':a.device,'dimension':None,'files':{}}
for section,filename in [('views','view_embeddings.npy'),('objects','object_embeddings.npy')]:
 vectors=[]; entries=[]
 for item in data[section]:
  batch=torch.stack([preprocess(Image.open(path).convert('RGB')) for path in item['paths']]).to(a.device)
  with torch.no_grad(): values=model.encode_image(batch).float(); values/=values.norm(dim=-1,keepdim=True); value=values.mean(0); value/=value.norm()
  vectors.append(value.cpu().numpy()); entries.append({k:v for k,v in item.items() if k!='paths'}|{'input_paths':item['paths'],'vector_index':len(vectors)-1})
 array=np.stack(vectors).astype('float32') if vectors else np.empty((0,512),dtype='float32')
 np.save(a.output_dir/filename,array); meta['files'][filename]={'shape':list(array.shape),'dtype':str(array.dtype),'entries':entries}
 meta['dimension']=array.shape[1] if len(array) else meta['dimension']
(a.output_dir/'index_metadata.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'complete','model':meta['model_version'],'dimension':meta['dimension'],'view_vectors':len(data['views']),'object_vectors':len(data['objects'])}))
