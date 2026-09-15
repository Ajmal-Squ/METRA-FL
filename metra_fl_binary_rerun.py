"""METRA-FL corrected binary edge-IDS rerun for Google Colab.

Usage in Colab
---------------
1. Runtime -> Change runtime type -> GPU.
2. Upload this file and run one command shown at the end of this docstring.

If the two official CSV files are not in /content/data, an upload dialog opens.
Required names (case-insensitive matching is supported):
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv

Outputs are written to Google Drive, including raw CSV files,
mean/95% CI summaries, LaTeX tables, manuscript macros, plots, checkpoints, and
a ZIP archive. Every experiment is checkpointed immediately and automatically
skipped after a restart. Values are measured; no scores are fabricated.

One command:
  !python /content/metra_fl_binary_rerun.py \
    --data-dir /content/drive/MyDrive/METRA_FL/data \
    --out /content/drive/MyDrive/METRA_FL/binary_rerun_results
"""

from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, shutil, subprocess, sys, time, warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

def install_if_needed():
    req = {"numpy":"numpy", "pandas":"pandas", "sklearn":"scikit-learn",
           "torch":"torch", "matplotlib":"matplotlib", "seaborn":"seaborn"}
    missing=[]
    for mod,pkg in req.items():
        try: __import__(mod)
        except ImportError: missing.append(pkg)
    if missing:
        subprocess.check_call([sys.executable,"-m","pip","install","-q"]+missing)
install_if_needed()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support,
    roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore", category=FutureWarning)

ATTACKS=["Analysis","Backdoor","DoS","Exploits","Fuzzers","Generic","Reconnaissance","Shellcode","Worms"]
NORMAL="Normal"

@dataclass
class Config:
    mode:str="quick"; seeds:Tuple[int,...]=(11,); rounds:int=8; clients:int=4
    clients_scale:Tuple[int,...]=(4,10); dirichlet:Tuple[float,...]=(0.3,)
    local_epochs:int=1; batch:int=512; lr:float=1e-3; weight_decay:float=1e-4
    hidden:int=64; heads:int=4; layers:int=1; dropout:float=0.15
    malicious_fracs:Tuple[float,...]=(0.0,0.2); attacks:Tuple[str,...]=("label_flip","sign_flip","backdoor")
    aggregators:Tuple[str,...] = ("fedavg","fedprox","median","trimmed_mean","krum","trust_v2x")
    alphas:Tuple[float,...]=(0.3,); trim_ratio:float=0.2; prox_mu:float=0.01
    trust_threshold:float=0.25; reputation_decay:float=0.8; stale_eta:float=0.1
    max_train:int=45000; max_test:int=18000; zero_day_families:Tuple[str,...]=("Exploits","Fuzzers","Generic")
    zero_rounds:int=6; bootstrap:int=1000; workers:int=0

def config_for(mode):
    if mode=="full":
        return Config(mode="full", seeds=(11,29,47,71,101), rounds=20, clients=20,
          clients_scale=(4,10,20,50,100), dirichlet=(0.1,0.3,0.5,1.0), local_epochs=1,
          batch=1024, hidden=96, layers=2, malicious_fracs=(0,.1,.2,.3,.4),
          attacks=("label_flip","sign_flip","scaling","gaussian","backdoor"),
          max_train=175341, max_test=82332, zero_day_families=tuple(ATTACKS), zero_rounds=15)
    return Config()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def locate_or_upload(data_dir:Path):
    data_dir.mkdir(parents=True,exist_ok=True)
    files=list(data_dir.glob("*.csv"))+list(Path("/content").glob("*.csv"))
    def pick(token):
        cand=[p for p in files if token in p.name.lower().replace("-","_")]
        return cand[0] if cand else None
    tr=pick("training_set") or pick("training")
    te=pick("testing_set") or pick("testing")
    if tr and te: return tr,te
    try:
        from google.colab import files as colab_files
        print("Upload the official UNSW-NB15 training and testing CSV files.")
        uploaded=colab_files.upload()
        for name,data in uploaded.items(): (data_dir/name).write_bytes(data)
        return locate_or_upload(data_dir)
    except Exception as e:
        raise FileNotFoundError("Place official training/testing CSV files in /content/data") from e

def clean_labels(s:pd.Series):
    x=s.fillna(NORMAL).astype(str).str.strip().str.replace("-","",regex=False).str.lower()
    mapping={a.lower():a for a in ATTACKS}; mapping.update({"normal":NORMAL,"nan":NORMAL,"":NORMAL,"backdoors":"Backdoor"})
    return x.map(mapping).fillna(s.astype(str).str.strip().str.title())

def stratified_cap(df,n,seed):
    if n<=0 or len(df)<=n:return df.sample(frac=1,random_state=seed).reset_index(drop=True)
    parts=[]
    for _,g in df.groupby("attack_cat",dropna=False):
        k=max(1,round(n*len(g)/len(df))); parts.append(g.sample(min(k,len(g)),random_state=seed))
    out=pd.concat(parts).drop_duplicates()
    if len(out)>n: out=out.sample(n,random_state=seed)
    return out.sample(frac=1,random_state=seed).reset_index(drop=True)

def load_data(train_path,test_path,cfg,seed):
    tr=pd.read_csv(train_path,low_memory=False); te=pd.read_csv(test_path,low_memory=False)
    tr.columns=[c.strip().lower() for c in tr.columns]; te.columns=[c.strip().lower() for c in te.columns]
    if "attack_cat" not in tr: raise ValueError("CSV must include attack_cat")
    tr["attack_cat"]=clean_labels(tr["attack_cat"]); te["attack_cat"]=clean_labels(te["attack_cat"])
    tr=stratified_cap(tr,cfg.max_train,seed); te=stratified_cap(te,cfg.max_test,seed+1)
    drop=[c for c in ("id","label","attack_cat") if c in tr.columns]
    features=[c for c in tr.columns if c not in drop and c in te.columns]
    ytr=tr.attack_cat.values; yte=te.attack_cat.values
    Xtr=tr[features].replace([np.inf,-np.inf],np.nan); Xte=te[features].replace([np.inf,-np.inf],np.nan)
    cats=[c for c in features if Xtr[c].dtype=="object"]
    nums=[c for c in features if c not in cats]
    pre=ColumnTransformer([
      ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),nums),
      ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                       ("oh",OneHotEncoder(handle_unknown="ignore",sparse_output=False,min_frequency=2))]),cats)],
      sparse_threshold=0)
    Xtr=pre.fit_transform(Xtr).astype("float32"); Xte=pre.transform(Xte).astype("float32")
    classes=[NORMAL]+[a for a in ATTACKS if a in set(ytr)|set(yte)]
    enc={c:i for i,c in enumerate(classes)}
    return Xtr,np.array([enc[v] for v in ytr]),Xte,np.array([enc[v] for v in yte]),classes,pre

class TrustNet(nn.Module):
    """Compact latent-token TCN-Transformer for tabular network-flow records."""
    def __init__(self,d,c,h=64,drop=.15):
        super().__init__(); self.tokens=8; self.dim=max(16,h//2)
        self.inp=nn.Linear(d,self.tokens*self.dim)
        self.tcn=nn.Sequential(nn.Conv1d(self.dim,self.dim,3,padding=1,groups=self.dim),
            nn.Conv1d(self.dim,self.dim,1),nn.GELU(),nn.Dropout(drop),
            nn.Conv1d(self.dim,self.dim,3,padding=2,dilation=2,groups=self.dim),
            nn.Conv1d(self.dim,self.dim,1),nn.GELU())
        layer=nn.TransformerEncoderLayer(self.dim,4,self.dim*2,drop,batch_first=True,
            activation="gelu",norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=1)
        self.gate=nn.Linear(self.dim,1);self.proj=nn.Sequential(nn.Linear(self.dim,h),nn.LayerNorm(h),nn.GELU())
        self.head=nn.Linear(h,c)
    def embed(self,x):
        z=self.inp(x).reshape(-1,self.tokens,self.dim);z=z+self.tcn(z.transpose(1,2)).transpose(1,2)
        z=self.transformer(z);w=torch.softmax(self.gate(z),dim=1);return F.normalize(self.proj((w*z).sum(1)),dim=1)
    def forward(self,x): return self.head(self.embed(x))

def loader(X,y,batch,shuffle=True):
    return DataLoader(TensorDataset(torch.from_numpy(X),torch.from_numpy(y).long()),batch_size=batch,shuffle=shuffle)

def state_vec(state): return torch.cat([v.detach().float().cpu().reshape(-1) for v in state.values()])
def delta_state(local,global_): return OrderedDict((k,local[k].detach().cpu()-global_[k].detach().cpu()) for k in global_)
def add_delta(global_,delta,scale=1.): return OrderedDict((k,global_[k]+scale*delta[k]) for k in global_)

def partition_dirichlet(y,n,alpha,seed,min_size=20):
    rng=np.random.default_rng(seed); classes=np.unique(y)
    last=None
    for _ in range(25):
        out=[[] for _ in range(n)]
        for c in classes:
            ids=np.where(y==c)[0]; rng.shuffle(ids); p=rng.dirichlet(np.repeat(alpha,n)); cuts=(np.cumsum(p)*len(ids)).astype(int)[:-1]
            for j,a in enumerate(np.split(ids,cuts)):out[j].extend(a.tolist())
        if min(map(len,out))>=min_size:return [np.array(v,dtype=int) for v in out]
        last=out
    # Very small alpha can repeatedly leave one or more clients almost empty.
    # Repair only the minimum-size constraint by transferring samples from the
    # largest clients; this preserves the strongly non-IID allocation as much
    # as possible and makes the experiment deterministic/resumable.
    if len(y)<n*min_size:
        raise RuntimeError(f"Need at least {n*min_size} samples for {n} clients")
    out=last
    for target in sorted(range(n),key=lambda j:len(out[j])):
        while len(out[target])<min_size:
            donors=[j for j in range(n) if j!=target and len(out[j])>min_size]
            if not donors: raise RuntimeError("Could not repair client partition")
            donor=max(donors,key=lambda j:len(out[j]))
            take=min(min_size-len(out[target]),len(out[donor])-min_size)
            chosen=rng.choice(len(out[donor]),size=take,replace=False)
            chosen_set=set(np.atleast_1d(chosen).tolist())
            moved=[v for k,v in enumerate(out[donor]) if k in chosen_set]
            out[donor]=[v for k,v in enumerate(out[donor]) if k not in chosen_set]
            out[target].extend(moved)
    return [np.asarray(v,dtype=int) for v in out]

def local_train(global_state,X,y,classes,cfg,device,seed,prox=False,poison=None):
    seed_all(seed); yy=y.copy()
    if poison=="label_flip": yy=(yy+1)%classes
    model=TrustNet(X.shape[1],classes,cfg.hidden,cfg.dropout).to(device); model.load_state_dict(global_state)
    base={k:v.detach().clone().to(device) for k,v in global_state.items()}
    counts=np.bincount(yy,minlength=classes); w=len(yy)/(classes*np.maximum(counts,1)); lossfn=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    model.train()
    for _ in range(cfg.local_epochs):
        for xb,yb in loader(X,yy,cfg.batch):
            xb,yb=xb.to(device),yb.to(device)
            if poison=="backdoor":
                m=torch.rand(len(xb),device=device)<.25; xb[m,0]=6.; yb[m]=0 if classes==2 else min(1,classes-1)
            opt.zero_grad(); loss=lossfn(model(xb),yb)
            if prox: loss += cfg.prox_mu/2*sum((p-base[n]).pow(2).sum() for n,p in model.named_parameters())
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
    d=delta_state(model.state_dict(),global_state)
    if poison=="sign_flip": d=OrderedDict((k,-5*v) for k,v in d.items())
    elif poison=="scaling": d=OrderedDict((k,10*v) for k,v in d.items())
    elif poison=="gaussian": d=OrderedDict((k,v+torch.randn_like(v)*max(v.std().item(),1e-3)*5) for k,v in d.items())
    return d

def aggregate(deltas,sizes,kind,reps,cfg,ref_eval=None,staleness=None):
    vec=torch.stack([state_vec(d) for d in deltas]); n=len(deltas); stale=np.zeros(n) if staleness is None else np.asarray(staleness)
    evidence={"accepted":np.ones(n,dtype=bool),"trust":np.ones(n),"outlier":np.zeros(n)}
    if kind=="median": agg=vec.median(0).values
    elif kind=="trimmed_mean":
        k=int(n*cfg.trim_ratio); sv=vec.sort(0).values; agg=sv[k:n-k].mean(0) if n>2*k else sv.mean(0)
    elif kind=="krum" and n>=3:
        dist=torch.cdist(vec,vec).pow(2); keep=max(1,n-2); scores=torch.stack([torch.topk(dist[i],keep,largest=False).values.sum() for i in range(n)])
        agg=vec[scores.argmin()]
    elif kind=="trust_v2x":
        center=vec.median(0).values; norms=vec.norm(dim=1); med=norms.median(); mad=(norms-med).abs().median()+1e-9
        sim=F.cosine_similarity(vec,center.unsqueeze(0)).add(1).div(2).numpy()
        out=np.clip(((norms-med).abs()/(3*mad)).numpy(),0,1)
        val=np.array([ref_eval(d) if ref_eval else .5 for d in deltas])
        trust=np.clip(.30*sim+.35*val+.15*np.asarray(reps)+.20*(1-out),0,1)
        accepted=trust>=cfg.trust_threshold
        if not accepted.any():accepted[np.argmax(trust)]=True
        weights=trust*np.asarray(sizes)*np.exp(-cfg.stale_eta*stale)*accepted
        weights=weights/(weights.sum()+1e-12)
        coord_med=vec.median(0).values;coord_mad=(vec-coord_med).abs().median(0).values+1e-9
        safe=torch.maximum(torch.minimum(vec,coord_med+3*coord_mad),coord_med-3*coord_mad)
        agg=(safe*torch.tensor(weights).float().unsqueeze(1)).sum(0)
        evidence={"accepted":accepted,"trust":trust,"outlier":out}
    else:
        w=torch.tensor(sizes,dtype=torch.float32); w/=w.sum(); agg=(vec*w[:,None]).sum(0)
    template=deltas[0]; out=OrderedDict(); pos=0
    for k,v in template.items(): num=v.numel(); out[k]=agg[pos:pos+num].reshape(v.shape).to(v.dtype); pos+=num
    return out,evidence

@torch.no_grad()
def predict(model,X,batch,device):
    model.eval(); logits=[]; embeds=[]
    for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),batch_size=batch):
        xb=xb.to(device); logits.append(model(xb).cpu()); embeds.append(model.embed(xb).cpu())
    return torch.cat(logits).numpy(),torch.cat(embeds).numpy()

def multiclass_metrics(y,logits,classes):
    pred=logits.argmax(1); prob=torch.softmax(torch.tensor(logits),1).numpy(); ybin=label_binarize(y,classes=np.arange(classes))
    out={"accuracy":accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro",zero_division=0),
         "balanced_accuracy":balanced_accuracy_score(y,pred),"mcc":matthews_corrcoef(y,pred)}
    try: out["auprc"]=average_precision_score(y,prob[:,1]) if classes==2 else average_precision_score(ybin,prob,average="macro")
    except ValueError: out["auprc"]=np.nan
    try: out["auroc"]=roc_auc_score(y,prob[:,1]) if classes==2 else roc_auc_score(ybin,prob,average="macro",multi_class="ovr")
    except ValueError: out["auroc"]=np.nan
    cm=confusion_matrix(y,pred,labels=np.arange(classes)); fp=cm.sum(0)-np.diag(cm); tn=cm.sum()-cm.sum(0)-cm.sum(1)+np.diag(cm)
    out["fpr"]=float(fp[1]/max(fp[1]+tn[1],1)) if classes==2 else float(np.mean(fp/np.maximum(fp+tn,1)))
    return out

def train_federated(Xtr,ytr,Xval,yval,Xte,yte,classes,cfg,seed,alpha,nclients,aggregator,mal_frac,attack,device,rounds=None):
    seed_all(seed); parts=partition_dirichlet(ytr,nclients,alpha,seed); rounds=rounds or cfg.rounds
    model=TrustNet(Xtr.shape[1],classes,cfg.hidden,cfg.dropout).to(device); global_state=OrderedDict((k,v.detach().cpu()) for k,v in model.state_dict().items())
    reps=np.repeat(.5,nclients); rng=np.random.default_rng(seed); mcount=int(round(nclients*mal_frac)); malicious=set(rng.choice(nclients,mcount,replace=False).tolist()) if mcount else set()
    ref_n=min(1500,len(Xval)); ridx=rng.choice(len(Xval),ref_n,replace=False); Xref,yref=Xval[ridx],yval[ridx]
    history=[]; uploaded=0; ledger_bytes=0; audit_seconds=0.; start=time.perf_counter()
    for rnd in range(rounds):
        deltas=[]; sizes=[]; ids=[]
        for i,idx in enumerate(parts):
            d=local_train(global_state,Xtr[idx],ytr[idx],classes,cfg,device,seed+rnd*1000+i,aggregator=="fedprox",attack if i in malicious else None)
            deltas.append(d);sizes.append(len(idx));ids.append(i);uploaded+=sum(v.numel()*v.element_size() for v in d.values())
            # Measured local hash/audit-record construction; not a substitute for
            # an external Fabric/Quorum consensus benchmark.
            ta=time.perf_counter(); digest=hashlib.sha256(state_vec(d).numpy().tobytes()).hexdigest()
            ledger_bytes += len(json.dumps({"client":i,"round":rnd,"hash":digest,"version":rnd}).encode())
            audit_seconds += time.perf_counter()-ta
        base_model=TrustNet(Xtr.shape[1],classes,cfg.hidden,cfg.dropout).to(device); base_model.load_state_dict(global_state)
        base_loss=F.cross_entropy(torch.tensor(predict(base_model,Xref,cfg.batch,device)[0]),torch.tensor(yref)).item()
        def ref_eval(d):
            tmp=TrustNet(Xtr.shape[1],classes,cfg.hidden,cfg.dropout).to(device);tmp.load_state_dict(add_delta(global_state,d))
            loss=F.cross_entropy(torch.tensor(predict(tmp,Xref,cfg.batch,device)[0]),torch.tensor(yref)).item()
            return float(1/(1+math.exp(np.clip((loss-base_loss)*3,-20,20))))
        agg,ev=aggregate(deltas,sizes,aggregator,reps,cfg,ref_eval,np.zeros(nclients))
        global_state=add_delta(global_state,agg); reps=cfg.reputation_decay*reps+(1-cfg.reputation_decay)*ev["trust"]
        if rnd in {0,rounds-1}:
            model.load_state_dict(global_state); met=multiclass_metrics(yval,predict(model,Xval,cfg.batch,device)[0],classes);history.append({"round":rnd+1,**met})
    elapsed=time.perf_counter()-start; model.load_state_dict(global_state)
    t0=time.perf_counter(); logits,_=predict(model,Xte,cfg.batch,device); infer=(time.perf_counter()-t0)*1000/len(Xte)
    met=multiclass_metrics(yte,logits,classes)
    asr=np.nan
    if attack=="backdoor" and mal_frac>0:
        trigger=Xte.copy();trigger[:,0]=6.;pp=predict(model,trigger,cfg.batch,device)[0].argmax(1)
        asr=float((pp[yte==1]==0).mean()) if classes==2 and np.any(yte==1) else float((pp==min(1,classes-1)).mean())
    met.update({"train_seconds":elapsed,"inference_ms":infer,"bytes_total":uploaded,
      "model_mb":sum(p.numel()*p.element_size() for p in model.parameters())/2**20,
      "attack_success_rate":asr,"audit_hash_ms":1000*audit_seconds/max(1,rounds*nclients),"ledger_bytes":ledger_bytes,
      "malicious_reject_rate":float(1-ev["accepted"][list(malicious)].mean()) if malicious else np.nan,
      "benign_reject_rate":float(1-ev["accepted"][[i for i in range(nclients) if i not in malicious]].mean())})
    return model,met,history

def open_set_metrics(cal_y,cal_logits,cal_emb,known_logits,known_emb,unk_logits,unk_emb):
    # Prototypes, normalization statistics, and the decision threshold are fitted
    # only on known validation samples; the held-out attack never calibrates them.
    prot=np.stack([cal_emb[cal_y==c].mean(0) if np.any(cal_y==c) else np.zeros(cal_emb.shape[1]) for c in range(cal_logits.shape[1])])
    prot/=np.linalg.norm(prot,axis=1,keepdims=True)+1e-9
    def components(logits,emb):
        m=logits.max(1,keepdims=True);energy=-np.log(np.exp(logits-m).sum(1))-m[:,0]
        return energy,1-(emb@prot.T).max(1)
    ce,cd=components(cal_logits,cal_emb); em,dm=ce.mean(),cd.mean();es,ds=ce.std()+1e-9,cd.std()+1e-9
    def score(logits,emb):
        e,d=components(logits,emb);return (e-em)/es+(d-dm)/ds
    threshold=np.quantile(score(cal_logits,cal_emb),.95)
    sk=score(known_logits,known_emb); su=score(unk_logits,unk_emb); y=np.r_[np.zeros(len(sk)),np.ones(len(su))]; s=np.r_[sk,su]
    auc=roc_auc_score(y,s); ap=average_precision_score(y,s); fpr,tpr,_=roc_curve(y,s); pred=s>=threshold
    return {"unknown_auroc":auc,"unknown_aupr":ap,"unknown_f1":f1_score(y,pred),"fpr95":float(fpr[np.argmin(np.abs(tpr-.95))])}

def run_zero_day(Xtr,ytr,Xte,yte,classes,cfg,seed,device):
    rows=[]
    for fam in cfg.zero_day_families:
        if fam not in classes:continue
        u=classes.index(fam); trmask=ytr!=u; temask=yte!=u; umask=yte==u
        kept=[i for i in range(len(classes)) if i!=u]; remap={old:new for new,old in enumerate(kept)}
        yytr=np.array([remap[v] for v in ytr[trmask]]); yyte=np.array([remap[v] for v in yte[temask]])
        Xdev,Xval,ydev,yval=train_test_split(Xtr[trmask],yytr,test_size=.2,random_state=seed,stratify=yytr)
        model,_,_=train_federated(Xdev,ydev,Xval,yval,Xte[temask],yyte,len(kept),cfg,seed,.3,cfg.clients,"trust_v2x",.2,"sign_flip",device,cfg.zero_rounds)
        cl,ce=predict(model,Xval,cfg.batch,device);kl,ke=predict(model,Xte[temask],cfg.batch,device);ul,ue=predict(model,Xte[umask],cfg.batch,device)
        rows.append({"seed":seed,"held_out":fam,**open_set_metrics(yval,cl,ce,kl,ke,ul,ue)})
    return rows

def run_binary_zero_day(Xtr,ymulti_tr,Xte,ymulti_te,class_names,cfg,seed,device):
    """Detect one attack family absent from training against normal test traffic."""
    rows=[]
    for fam in cfg.zero_day_families:
        if fam not in class_names: continue
        u=class_names.index(fam);train_mask=ymulti_tr!=u
        ybin_tr=(ymulti_tr[train_mask]!=0).astype(np.int64)
        test_mask=(ymulti_te==0)|(ymulti_te==u);ybin_te=(ymulti_te[test_mask]==u).astype(np.int64)
        Xdev,Xval,ydev,yval=train_test_split(Xtr[train_mask],ybin_tr,test_size=.2,random_state=seed,stratify=ybin_tr)
        model,_,_=train_federated(Xdev,ydev,Xval,yval,Xte[test_mask],ybin_te,2,cfg,seed,.3,cfg.clients,
                                  "trust_v2x",0.0,"none",device,cfg.zero_rounds)
        logits,_=predict(model,Xte[test_mask],cfg.batch,device);prob=torch.softmax(torch.tensor(logits),1).numpy()[:,1]
        pred=logits.argmax(1);tn,fp,fn,tp=confusion_matrix(ybin_te,pred,labels=[0,1]).ravel();rfpr,rtpr,_=roc_curve(ybin_te,prob)
        rows.append({"seed":seed,"held_out":fam,"unknown_auroc":roc_auc_score(ybin_te,prob),
          "unknown_aupr":average_precision_score(ybin_te,prob),"unknown_f1":f1_score(ybin_te,pred,zero_division=0),
          "fpr95":float(rfpr[np.argmin(np.abs(rtpr-.95))]),"zero_day_fpr":float(fp/max(fp+tn,1)),"zero_day_recall":float(tp/max(tp+fn,1)),
          "zero_day_precision":float(tp/max(tp+fp,1))})
    return rows

def ci95(x):
    x=np.asarray(pd.Series(x).dropna(),float)
    return np.nan if len(x)<2 else student_t.ppf(.975,len(x)-1)*x.std(ddof=1)/math.sqrt(len(x))

def summarize(df,group,metrics):
    rows=[]
    for key,g in df.groupby(group,dropna=False):
        key=(key,) if not isinstance(key,tuple) else key; row=dict(zip(group,key));row["runs"]=len(g)
        for m in metrics:
            row[m+"_mean"]=g[m].mean();row[m+"_ci95"]=ci95(g[m])
        rows.append(row)
    return pd.DataFrame(rows)

def write_outputs(out,raw,zero,cfg):
    out.mkdir(parents=True,exist_ok=True); rdf=pd.DataFrame(raw); zdf=pd.DataFrame(zero)
    rdf.to_csv(out/"federated_raw.csv",index=False); zdf.to_csv(out/"zero_day_raw.csv",index=False)
    met=["accuracy","macro_f1","balanced_accuracy","mcc","auprc","auroc","fpr","train_seconds","inference_ms","bytes_total","model_mb","attack_success_rate","audit_hash_ms","ledger_bytes","malicious_reject_rate","benign_reject_rate"]
    s=summarize(rdf,["aggregator","alpha","clients","malicious_fraction","attack"],met) if len(rdf) else pd.DataFrame();s.to_csv(out/"federated_summary.csv",index=False)
    zs=summarize(zdf,["held_out"],["unknown_auroc","unknown_aupr","unknown_f1","fpr95","zero_day_fpr","zero_day_recall","zero_day_precision"]) if len(zdf) else pd.DataFrame();zs.to_csv(out/"zero_day_summary.csv",index=False)
    (out/"config.json").write_text(json.dumps(asdict(cfg),indent=2))
    latex=[]
    latex.append("% Auto-generated measured results. Inspect before including.\n")
    if len(s): latex.append(s.to_latex(index=False,float_format=lambda x:f"{x:.4f}",longtable=True,escape=True))
    if len(zs):latex.append(zs.to_latex(index=False,float_format=lambda x:f"{x:.4f}",escape=True))
    (out/"measured_tables.tex").write_text("\n\n".join(latex))
    best=s[(s.aggregator=="trust_v2x")&(s.malicious_fraction==0)] if len(s) else pd.DataFrame()
    macros=[]
    if len(best):
        r=best.sort_values("macro_f1_mean",ascending=False).iloc[0]
        for name,col in [("TrustAccuracy","accuracy_mean"),("TrustMacroFOne","macro_f1_mean"),("TrustMCC","mcc_mean"),("TrustFPR","fpr_mean")]:
            macros.append(f"\\newcommand{{\\{name}}}{{{100*r[col]:.2f}\\%}}" if col!="mcc_mean" else f"\\newcommand{{\\{name}}}{{{r[col]:.3f}}}")
    (out/"measured_macros.tex").write_text("\n".join(macros)+"\n")
    if len(s):
        p=s[(s.alpha==s.alpha.min())&(s.malicious_fraction==s.malicious_fraction.max())]
        if len(p):
            plt.figure(figsize=(8,4));plt.bar(p.aggregator,p.macro_f1_mean,yerr=p.macro_f1_ci95);plt.ylabel("Macro-F1");plt.xticks(rotation=30,ha="right");plt.tight_layout();plt.savefig(out/"robust_macro_f1.pdf");plt.close()
    shutil.make_archive(str(out),"zip",out)

def _read_jsonl(path):
    if not path.exists(): return []
    rows=[]
    for line in path.read_text().splitlines():
        try: rows.append(json.loads(line))
        except Exception: pass
    return rows

def _append_jsonl(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a") as fh: fh.write(json.dumps(row)+"\n")

def main():
    ap=argparse.ArgumentParser(description="Corrected binary METRA-FL edge-IDS rerun")
    ap.add_argument("--data-dir",default="/content/drive/MyDrive/METRA_FL/data")
    ap.add_argument("--out",default="/content/drive/MyDrive/METRA_FL/binary_rerun_results")
    ap.add_argument("--seeds",default="11,29,47",help="Comma-separated seeds")
    ap.add_argument("--rounds",type=int,default=30);ap.add_argument("--zero-rounds",type=int,default=20)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg=config_for("full");cfg.seeds=tuple(int(x) for x in args.seeds.split(","));cfg.rounds=args.rounds
    cfg.zero_rounds=args.zero_rounds;cfg.local_epochs=1;cfg.batch=1024;cfg.trust_threshold=.55
    cfg.dirichlet=(0.1,0.3,1.0);cfg.clients_scale=(4,20,50)
    cfg.attacks=("label_flip","sign_flip","backdoor");cfg.malicious_fracs=(0.0,0.2)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": print("WARNING: GPU not detected; enable a Colab GPU runtime.")
    print("METRA-FL corrected binary rerun | device:",device,"| seeds:",cfg.seeds,"| output:",out,flush=True)
    trp,tep=locate_or_upload(Path(args.data_dir))
    fed_ck=out/"federated_checkpoint.jsonl";zero_ck=out/"zero_day_checkpoint.jsonl"
    raw=_read_jsonl(fed_ck);zero=_read_jsonl(zero_ck);overall_start=time.perf_counter();completed_now=0
    env={"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,
         "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
         "seeds":cfg.seeds,"rounds":cfg.rounds,"zero_rounds":cfg.zero_rounds}
    (out/"environment.json").write_text(json.dumps(env,indent=2))
    for seed in cfg.seeds:
        print("\nPreparing seed",seed,flush=True)
        Xtr,ymulti_tr,Xte,ymulti_te,names,pre=load_data(trp,tep,cfg,seed);classes=2
        ytr=(ymulti_tr!=0).astype(np.int64);yte=(ymulti_te!=0).astype(np.int64)
        Xdev,Xval,ydev,yval=train_test_split(Xtr,ytr,test_size=.2,random_state=seed,stratify=ytr)
        tasks=[]
        tasks += [(0.3,20,a,0.,"none","baseline") for a in cfg.aggregators]
        tasks += [(0.3,n,a,0.,"none","scalability") for n in (4,50) for a in ("fedavg","trust_v2x")]
        tasks += [(al,20,a,0.,"none","non_iid") for al in (0.1,1.0) for a in ("fedavg","trust_v2x")]
        tasks += [(0.3,20,a,0.2,atk,"robustness") for a in ("fedavg","median","krum","trust_v2x") for atk in cfg.attacks]
        done={(r["seed"],r["alpha"],r["clients"],r["aggregator"],r["malicious_fraction"],r["attack"]) for r in raw}
        for ti,(alpha,nclients,agg,frac,atk,study) in enumerate(tasks,1):
            key=(seed,alpha,nclients,agg,frac,atk)
            if key in done: print("Skip completed:",study,key);continue
            print(f"Federated {ti}/{len(tasks)} | {study} |",key,flush=True)
            t0=time.perf_counter();_,met,_=train_federated(Xdev,ydev,Xval,yval,Xte,yte,classes,cfg,seed,alpha,nclients,agg,frac,atk,device)
            row={"seed":seed,"study":study,"alpha":alpha,"clients":nclients,"aggregator":agg,
                 "malicious_fraction":frac,"attack":atk,**met};raw.append(row);_append_jsonl(fed_ck,row);completed_now+=1
            print(f"Saved. Task time {(time.perf_counter()-t0)/60:.1f} min; completed this session {completed_now}.",flush=True)
        zdone={(r["seed"],r["held_out"]) for r in zero}
        for zi,fam in enumerate(ATTACKS,1):
            if (seed,fam) in zdone: print("Skip zero-day:",seed,fam);continue
            print(f"Zero-day {zi}/9 | seed {seed} | held out {fam}",flush=True)
            old=cfg.zero_day_families;cfg.zero_day_families=(fam,);rows=run_binary_zero_day(Xtr,ymulti_tr,Xte,ymulti_te,names,cfg,seed,device);cfg.zero_day_families=old
            for row in rows: zero.append(row);_append_jsonl(zero_ck,row)
            print("Saved zero-day",fam,flush=True)
        del Xtr,Xte,Xdev,Xval;torch.cuda.empty_cache()
        write_outputs(out,raw,zero,cfg)
        print("Seed",seed,"summaries saved.",flush=True)
    write_outputs(out,raw,zero,cfg)
    print(f"ALL COMPLETE in {(time.perf_counter()-overall_start)/3600:.2f} h")
    print("Final archive:",str(out)+".zip")

if __name__=="__main__":main()
