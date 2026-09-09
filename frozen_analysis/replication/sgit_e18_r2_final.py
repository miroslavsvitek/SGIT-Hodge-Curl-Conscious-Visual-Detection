from __future__ import annotations
import csv,json,re,sys,time,traceback,zipfile,gc
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from remote_zip import http_json,central_directory,download_member
from hgft_x2_math import lag_flow,project_coeff,spectral_trajectory,total_subspace_energy,swap_null,exact_sign_one

REP=json.loads((ROOT/"config"/"E18_R0_LITERAL_REPLICATION_CONFIG_1_0.json").read_text(encoding="utf-8"))
QC=json.loads((ROOT/"config"/"E18_R1_STRUCTURE_QC_LOCK_1_0.json").read_text(encoding="utf-8"))
AMEND=json.loads((ROOT/"config"/"E18_R2_TECHNICAL_READER_AMENDMENT_1_0.json").read_text(encoding="utf-8"))
RUN=json.loads((ROOT/"config"/"E18_R2_FINAL_RUN_CONFIG_1_0.json").read_text(encoding="utf-8"))

RESULTS=ROOT/"results";STATE=ROOT/"state";EXPORT=ROOT/"export"
for d in (RESULTS,STATE,EXPORT):d.mkdir(parents=True,exist_ok=True)
LOG=RESULTS/"E18_R2B_FINAL_RUN_LOG.txt"

def log(s=""):
    x=f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {s}"
    print(x,flush=True)
    with LOG.open("a",encoding="utf-8") as f:f.write(x+"\n")

def write_csv(p,rows):
    rows=list(rows);p=Path(p)
    if not rows:
        p.write_text("",encoding="utf-8");return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with p.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");w.writeheader();w.writerows(rows)

def cache_root():
    for s in RUN["runtime"]["persistent_cache_roots"]:
        p=Path(s)
        try:
            p.mkdir(parents=True,exist_ok=True);q=p/".probe";q.write_text("x");q.unlink();return p
        except Exception:pass
    p=ROOT/"cache";p.mkdir(exist_ok=True);return p
CACHE=cache_root()

def scalar_str(x):
    if isinstance(x,str):return x.strip()
    if isinstance(x,bytes):return x.decode("utf-8","ignore").strip()
    a=np.asarray(x)
    if a.ndim==0:return str(a.item()).strip()
    if a.size==1:return str(a.reshape(-1)[0]).strip()
    return "".join(str(y) for y in a.reshape(-1).tolist()).strip()

def normalize_labels(x):
    a=np.asarray(x,dtype=object).reshape(-1)
    return [scalar_str(v) for v in a]

def normalize_cells(x):
    # FieldTrip cell arrays load as object arrays. If a numeric stack is encountered,
    # split only along the obvious trial axis; otherwise fail closed.
    a=np.asarray(x)
    if a.dtype==object:
        return list(a.reshape(-1))
    if a.ndim==3:
        # choose first axis as trials only when other dimensions look like channels/time
        return [a[i] for i in range(a.shape[0])]
    if a.ndim==2:
        return [a]
    raise RuntimeError(f"unsupported cell/numeric representation shape={a.shape} dtype={a.dtype}")

def resolve_final_members():
    art=http_json("https://api.figshare.com/v2/articles/5418967")
    archives=[]
    for f in art.get("files",[]):
        n=str(f.get("name",""))
        if n.startswith("final_ERP_") and n.endswith(".zip"):
            archives.append((n,f["download_url"],int(f["size"]),central_directory(f["download_url"],int(f["size"]))))
    if not archives:raise RuntimeError("final_ERP archives missing")
    out={}
    for c in REP["dataset"]["frozen_candidates"]:
        sid=int(c["subject"]);sess=int(c["large_session"])
        target=f"Fp{sid}s{sess}_final.mat"
        matches=[]
        for an,url,size,cd in archives:
            for mn,e in cd.items():
                if mn.endswith(target):matches.append((an,url,mn,e))
        if len(matches)!=1:raise RuntimeError(f"{target}: match count={len(matches)}")
        out[(sid,sess)]=matches[0]
    return out

def load_reference():
    z=np.load(ROOT/"reference"/"HGFT_X2_DISCOVERY_COMPLEX.npz",allow_pickle=True)
    return {
        "names":[str(x) for x in z["channel_names"].tolist()],
        "edges":[tuple(map(int,x)) for x in z["edges"]],
        "curl_basis":np.asarray(z["curl_basis"],float),
        "grad_basis":np.asarray(z["grad_basis"],float),
        "curl_sel":np.asarray(z["curl_basis"][:,z["curl_sel_idx"]],float),
        "grad_sel":np.asarray(z["grad_basis"][:,z["grad_sel_idx"]],float)
    }

def baseline_standardize(X,times):
    ix=np.flatnonzero((times>=-0.09-1e-12)&(times<0.0-1e-12))
    if len(ix)<8:raise RuntimeError(f"too few baseline samples: {len(ix)}")
    mu=X[:,:,ix].mean(axis=(0,2))
    sd=X[:,:,ix].std(axis=(0,2))
    if not np.isfinite(mu).all() or not np.isfinite(sd).all() or np.any(sd<1e-10):
        raise RuntimeError("degenerate baseline mean/SD")
    return (X-mu[None,:,None])/sd[None,:,None]

def nuisance_residualize(X,conditions,objcorr):
    n,v,t=X.shape
    cols=[np.ones(n,float)]
    names=["intercept"]

    # critical condition 1-4 categorical, first level reference
    cond=np.asarray(conditions,int)
    levels=sorted(set(cond.tolist()))
    for lev in levels[1:]:
        cols.append((cond==lev).astype(float))
        names.append(f"condition={lev}")

    oc=np.asarray(objcorr,float)
    if not np.isfinite(oc).all():
        raise RuntimeError("objective correctness contains nonfinite values")
    if len(set(oc.tolist()))>=2:
        cols.append(oc)
        names.append("objective_correctness")

    D=np.column_stack(cols)
    Y=X.reshape(n,v*t)
    # label-blind OLS residualization, exactly subject-internal.
    R=Y-D@(np.linalg.pinv(D)@Y)
    return R.reshape(n,v,t),names

def subject_from_clean(path,sid,sess,ref):
    from scipy.io import loadmat
    m=loadmat(str(path),squeeze_me=True,struct_as_record=False,variable_names=["clean"])
    if "clean" not in m:raise RuntimeError("clean struct missing")
    clean=m["clean"]

    for f in AMEND["final_reader_required_clean_fields"]:
        if not hasattr(clean,f):raise RuntimeError(f"clean.{f} missing")

    labels=normalize_labels(clean.label)
    missing=[x for x in ref["names"] if x not in labels]
    extra=[x for x in labels if x not in ref["names"]]
    if missing:
        raise RuntimeError("missing frozen reference labels: "+",".join(missing[:8]))
    if len(extra)!=1 or extra[0]!="Nz":
        raise RuntimeError(f"unexpected extra clean labels: {extra}")
    if len(labels)!=65:
        raise RuntimeError(f"clean.label count={len(labels)}; expected 65 with single extra Nz")
    order=[labels.index(x) for x in ref["names"]]

    trialinfo=np.asarray(clean.trialinfo,float)
    if trialinfo.ndim==1:trialinfo=trialinfo[None,:]
    if trialinfo.shape[1] < 12:raise RuntimeError(f"trialinfo has {trialinfo.shape[1]} columns")

    trials=normalize_cells(clean.trial)
    times=normalize_cells(clean.time)
    if len(times)==1 and len(trials)>1:
        times=times*len(trials)
    if len(trials)!=len(times) or len(trials)!=len(trialinfo):
        raise RuntimeError(f"trial/time/trialinfo length mismatch {len(trials)}/{len(times)}/{len(trialinfo)}")

    kept_X=[];kept_times=[];kept_cond=[];kept_resp=[];kept_obj=[]
    n_critical=0;n_nonfinite=0;n_badshape=0
    for i,(tr,tm) in enumerate(zip(trials,times)):
        cond=int(round(float(trialinfo[i,1])))
        if cond not in (1,2,3,4):continue
        resp=int(round(float(trialinfo[i,11])))
        if resp not in (1,2,3):continue
        obj=float(trialinfo[i,10])
        n_critical+=1

        x=np.asarray(tr,float)
        if x.ndim!=2:
            n_badshape+=1;continue
        # Technical R2B adapter: orient against the full clean.label count (65),
        # then select exactly the frozen 64 reference channels by name/order.
        nlab=len(labels)
        if x.shape[0]==nlab:
            pass
        elif x.shape[1]==nlab:
            x=x.T
        else:
            n_badshape+=1;continue
        x=x[order,:]
        if x.shape[0]!=64:
            n_badshape+=1;continue
        t=np.asarray(tm,float).reshape(-1)
        if x.shape[1]!=len(t):
            n_badshape+=1;continue

        req=np.flatnonzero((t>=-0.09-1e-12)&(t<=0.56+1e-12))
        if len(req)<150:  # 250 Hz over 0.65 s -> ~163 samples
            n_badshape+=1;continue
        if not np.isfinite(x[:,req]).all():
            n_nonfinite+=1;continue

        kept_X.append(x[:,req])
        kept_times.append(t[req])
        kept_cond.append(cond);kept_resp.append(resp);kept_obj.append(obj)

    if not kept_X:raise RuntimeError("no finite retained critical trials")

    # Require identical source-native time grid for all retained trials.
    t0=kept_times[0]
    for t in kept_times[1:]:
        if len(t)!=len(t0) or not np.allclose(t,t0,atol=1e-10,rtol=0):
            raise RuntimeError("retained trials do not share identical 250-Hz time grid")
    X=np.stack(kept_X,axis=0)
    cond=np.asarray(kept_cond,int)
    resp=np.asarray(kept_resp,int)
    obj=np.asarray(kept_obj,float)

    n1=int(np.sum(resp==1));n2=int(np.sum(resp==2));n3=int(np.sum(resp==3))
    if n1<25 or n2<25:
        raise RuntimeError(f"post-QC PAS1/PAS2 gate {n1}/{n2}")

    X=baseline_standardize(X,t0)
    X,nuis=nuisance_residualize(X,cond,obj)

    primary=(resp==1)|(resp==2)
    Xp=X[primary]
    labels_primary=np.where(resp[primary]==1,"UNSEEN","SEEN")

    F=lag_flow(Xp,ref["edges"],lag=1)
    ftimes=t0[1:]
    Cc=project_coeff(F,ref["curl_sel"])
    Cg=project_coeff(F,ref["grad_sel"])
    wins=list(REP["trajectory"]["windows_s"].items())
    qcurl,rawcurl=spectral_trajectory(Cc,labels_primary,ftimes,wins)
    qgrad,rawgrad=spectral_trajectory(Cg,labels_primary,ftimes,wins)

    Eg=total_subspace_energy(F,ref["grad_basis"])
    Ec=total_subspace_energy(F,ref["curl_basis"])
    Et=np.sum(F*F,axis=1)
    Eh=np.maximum(Et-Eg-Ec,0.0)
    frac=[]
    for lab in ("UNSEEN","SEEN"):
        lm=labels_primary==lab
        for wname,w in wins:
            a,b=map(float,w)
            ix=np.flatnonzero((ftimes>=a-1e-12)&(ftimes<b-1e-12))
            tg=float(np.mean(Eg[lm][:,ix]));tc=float(np.mean(Ec[lm][:,ix]));th=float(np.mean(Eh[lm][:,ix]))
            den=tg+tc+th+1e-15
            frac.append({
                "subject":sid,"session":sess,"label":lab,"window":wname,
                "gradient_fraction":tg/den,"curl_fraction":tc/den,"harmonic_fraction":th/den
            })

    return {
        "qcurl":qcurl,"qgrad":qgrad,"fractions":frac,
        "PAS1":n1,"PAS2":n2,"PAS3":n3,
        "critical_seen":n_critical,
        "nonfinite_excluded":n_nonfinite,
        "badshape_excluded":n_badshape,
        "nuisance":" | ".join(nuis),
        "time_samples":len(t0),
        "tmin":float(t0[0]),"tmax":float(t0[-1])
    }

def package(result):
    (RESULTS/"SGIT_HGFT_E18_R2B_RESULT.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    s=result.get("statistics") or {}
    (RESULTS/"SGIT_HGFT_E18_R2B_REPORT.md").write_text(f"""# E18-R2 final literal HGFT-X2 replication

Technical status: **{result['technical_status']}**

Scientific status: **{result['scientific_status']}**

N evaluable: {result['evaluable_subjects']}

Primary curl:
- median LOSO score: {s.get('curl_median_template_score')}
- mean score: {s.get('curl_mean_template_score')}
- positive subjects: {s.get('curl_positive_subjects')}
- permutation p: {s.get('curl_permutation_p')}
- null median: {s.get('curl_null_median')}
- null q97.5: {s.get('curl_null_q975')}
- sign test: {s.get('curl_sign_test')}
- supported: {s.get('supported')}

Gradient specificity control:
- median score: {s.get('gradient_median_template_score')}
- permutation p: {s.get('gradient_permutation_p')}

Literal target:
PAS1 (no visual experience) vs PAS2 (weak visual experience), LARGE Gabor, critical conditions 1-4.

No PAS pooling, small-session rescue, mode/edge/window selection, resampling, graph rebuild, or alpha change.
""",encoding="utf-8")
    (STATE/"CONTINUATION_CHECKPOINT_E18_R2B.md").write_text(
        f"CURRENT_STEP=E18_R2B_FINAL_COMPLETE\n"
        f"TECHNICAL_STATUS={result['technical_status']}\n"
        f"SCIENTIFIC_STATUS={result['scientific_status']}\n"
        f"N_EVALUABLE={result['evaluable_subjects']}\n"
        f"NEXT={result['next_action']}\n",encoding="utf-8")
    stamp=time.strftime("%Y%m%d_%H%M%S")
    zout=EXPORT/f"SGIT_HGFT_E18_R2B_RESULTS_{stamp}.zip"
    with zipfile.ZipFile(zout,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(RESULTS.glob("*")):
            if p.is_file():z.write(p,arcname=p.name)
        z.write(STATE/"CONTINUATION_CHECKPOINT_E18_R2B.md",arcname="CONTINUATION_CHECKPOINT_E18_R2B.md")
        for q,arc in [
            (ROOT/"config"/"E18_R0_LITERAL_REPLICATION_CONFIG_1_0.json","provenance/E18_R0_LITERAL_REPLICATION_CONFIG_1_0.json"),
            (ROOT/"config"/"E18_R1_STRUCTURE_QC_LOCK_1_0.json","provenance/E18_R1_STRUCTURE_QC_LOCK_1_0.json"),
            (ROOT/"config"/"E18_R2_TECHNICAL_READER_AMENDMENT_1_0.json","provenance/E18_R2_TECHNICAL_READER_AMENDMENT_1_0.json"),
            (ROOT/"config"/"E18_R2_FINAL_RUN_CONFIG_1_0.json","provenance/E18_R2_FINAL_RUN_CONFIG_1_0.json"),
            (ROOT/"protocol"/"E18_R0_LITERAL_REPLICATION_PROTOCOL_LOCK.md","provenance/E18_R0_LITERAL_REPLICATION_PROTOCOL_LOCK.md"),
            (ROOT/"protocol"/"E18_R2_TECHNICAL_READER_AMENDMENT.md","provenance/E18_R2_TECHNICAL_READER_AMENDMENT.md"),
            (ROOT/"config"/"E18_R2B_CHANNEL_SELECTION_AMENDMENT_1_0.json","provenance/E18_R2B_CHANNEL_SELECTION_AMENDMENT_1_0.json"),
            (ROOT/"protocol"/"E18_R2B_CHANNEL_SELECTION_AMENDMENT.md","provenance/E18_R2B_CHANNEL_SELECTION_AMENDMENT.md"),
        ]:
            z.write(q,arcname=arc)
    log(f"RESULT ZIP: {zout.resolve()}")

def main():
    if LOG.exists():LOG.unlink()
    log("E18-R2B FINAL LITERAL REPLICATION RESUME START.")
    log("Scientific protocol: frozen R0. Technical QC: frozen R1. bad_chan requirement removed only as technical amendment.")
    log(f"Expected selective final-MAT download if cache empty: ~{RUN['expected_final_MAT_download_gb_decimal']:.2f} GB.")
    log("Downloads are resumable and subjects are processed sequentially.")

    members=resolve_final_members()
    ref=load_reference()
    data={};qc=[];fractions=[]

    candidates=REP["dataset"]["frozen_candidates"]
    for i,c in enumerate(candidates,1):
        sid=int(c["subject"]);sess=int(c["large_session"])
        an,url,mn,e=members[(sid,sess)]
        log(f"SUBJECT [{i}/{len(candidates)}] Fp{sid} session {sess} — member {int(e['compressed_size'])/1e6:.1f} MB")
        try:
            p=download_member(url,e,CACHE/"final_mat"/Path(mn).name,chunk=4_194_304)
            d=subject_from_clean(p,sid,sess,ref)
            data[sid]=d
            qc.append({
                "subject":sid,"large_session":sess,"status":"PASS",
                "PAS1_after_QC":d["PAS1"],"PAS2_after_QC":d["PAS2"],"PAS3_after_QC":d["PAS3"],
                "critical_seen":d["critical_seen"],
                "nonfinite_excluded":d["nonfinite_excluded"],
                "badshape_excluded":d["badshape_excluded"],
                "nuisance":d["nuisance"],
                "time_samples":d["time_samples"],"tmin":d["tmin"],"tmax":d["tmax"]
            })
            fractions.extend(d["fractions"])
        except Exception as ex:
            qc.append({"subject":sid,"large_session":sess,"status":"FAIL","reason":str(ex)})
            log(f"  FAIL: {ex}")
        finally:
            gc.collect()

    write_csv(RESULTS/"E18_R2B_SUBJECT_QC.csv",qc)
    write_csv(RESULTS/"E18_R2B_HODGE_ENERGY_FRACTIONS.csv",fractions)

    n=len(data)
    minN=REP["behavior"]["minimum_evaluable_subjects"]
    if n<minN:
        result={
            "technical_status":"TECHNICALLY_UNEVALUABLE",
            "scientific_status":"E18_R2_TOO_FEW_EVALUABLE_SUBJECTS_NO_SCIENTIFIC_VERDICT",
            "evaluable_subjects":n,
            "statistics":None,
            "next_action":"Do not lower frozen subject/trial/channel gates. Literal replication remains technically unevaluable."
        }
        package(result);return

    subs=sorted(data)
    Uc=np.stack([data[s]["qcurl"]["UNSEEN"] for s in subs])
    Sc=np.stack([data[s]["qcurl"]["SEEN"] for s in subs])
    Ug=np.stack([data[s]["qgrad"]["UNSEEN"] for s in subs])
    Sg=np.stack([data[s]["qgrad"]["SEEN"] for s in subs])

    P=REP["primary_confirmatory_test"]
    draws=int(P["permutation_draws"]);seed=int(P["seed"]);alpha=float(P["one_sided_alpha"])
    cs,cobs,cnull,cp=swap_null(Uc,Sc,draws,seed)
    gs,gobs,gnull,gp=swap_null(Ug,Sg,draws,seed+1)

    supported=(n>=minN and cobs>0 and cp<=alpha)

    write_csv(RESULTS/"E18_R2B_SUBJECT_TEMPLATE_SCORES.csv",[
        {"subject":sid,"curl_template_score":float(cs[i]),"gradient_template_score":float(gs[i])}
        for i,sid in enumerate(subs)
    ])
    write_csv(RESULTS/"E18_R2B_CURL_PERMUTATION_NULL.csv",[
        {"draw":i+1,"median_score":float(x)} for i,x in enumerate(cnull)
    ])
    write_csv(RESULTS/"E18_R2B_GRADIENT_PERMUTATION_NULL.csv",[
        {"draw":i+1,"median_score":float(x)} for i,x in enumerate(gnull)
    ])

    stats={
        "N_evaluable":n,
        "curl_median_template_score":float(cobs),
        "curl_mean_template_score":float(np.mean(cs)),
        "curl_positive_subjects":int(np.sum(cs>0)),
        "curl_permutation_p":float(cp),
        "curl_null_median":float(np.median(cnull)),
        "curl_null_q975":float(np.quantile(cnull,0.975)),
        "curl_sign_test":exact_sign_one(cs),
        "gradient_median_template_score":float(gobs),
        "gradient_mean_template_score":float(np.mean(gs)),
        "gradient_positive_subjects":int(np.sum(gs>0)),
        "gradient_permutation_p":float(gp),
        "gradient_null_median":float(np.median(gnull)),
        "gradient_null_q975":float(np.quantile(gnull,0.975)),
        "gradient_sign_test":exact_sign_one(gs),
        "alpha":alpha,
        "supported":bool(supported)
    }

    scientific=("E18_R2_LITERAL_HGFT_X2_MATCHED_REPLICATION_SUPPORTED"
                if supported else
                "E18_R2_LITERAL_HGFT_X2_MATCHED_REPLICATION_NOT_SUPPORTED")
    next_action=(
        "Independent literal matched replication supported: upgrade the curl-relational principle substantially in the SGIT evidence architecture; keep causality and explicit-trivector claims separate."
        if supported else
        "Literal matched replication not supported: close the exact HGFT-X2 direct-detection formulation; no post-hoc PAS/mode/edge/window/resampling rescue."
    )
    result={
        "technical_status":"PASS",
        "scientific_status":scientific,
        "evaluable_subjects":n,
        "statistics":stats,
        "next_action":next_action
    }
    package(result)

if __name__=="__main__":
    try:main()
    except Exception:
        traceback.print_exc()
        try:
            with LOG.open("a",encoding="utf-8") as f:traceback.print_exc(file=f)
        except Exception:pass
        raise SystemExit(99)
