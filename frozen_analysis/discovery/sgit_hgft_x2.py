from __future__ import annotations
import csv,io,json,re,sys,time,traceback,zipfile,urllib.request
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from remote_zip import central_directory,download_member
from hgft_x2_math import (
    graph_knn,build_hodge,lag_flow,project_coeff,spectral_trajectory,
    total_subspace_energy,swap_null,exact_sign_one
)
from synthetic_sanity import run as run_synthetic

CFG=json.loads((ROOT/"config"/"HGFT_X2_FROZEN_CONFIG_1_0.json").read_text(encoding="utf-8"))
RESULTS=ROOT/"results";STATE=ROOT/"state";EXPORT=ROOT/"export"
for d in (RESULTS,STATE,EXPORT):d.mkdir(parents=True,exist_ok=True)
LOG=RESULTS/"HGFT_X2_RUN_LOG.txt"

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
    for s in CFG["runtime"]["persistent_cache_roots"]:
        p=Path(s)
        try:
            p.mkdir(parents=True,exist_ok=True);q=p/".hgftx2";q.write_text("x");q.unlink();return p
        except Exception:pass
    p=ROOT/"cache";p.mkdir(exist_ok=True);return p
CACHE=cache_root()

def parse_tsv_bytes(b):
    txt=b.decode("utf-8-sig",errors="replace");lines=txt.splitlines();hi=None
    for i,line in enumerate(lines[:30]):
        if line.count("\t")>=2:
            f=[x.strip() for x in line.split("\t")]
            if "cond" in f and "stimlevel" in f:
                hi=i;break
    if hi is None:return [],[]
    rd=csv.DictReader(io.StringIO("\n".join(lines[hi:])),delimiter="\t")
    fields=[str(x).strip() for x in (rd.fieldnames or [])]
    rows=[{str(k).strip():("" if v is None else str(v).strip()) for k,v in r.items() if k is not None} for r in rd]
    return fields,rows

def lower(v):return str(v).strip().lower()
def norm(v):return str(v).strip()

def bad_indices_from_bytes(b,n):
    txt=b.decode("utf-8-sig",errors="replace").strip()
    vals=[int(x) for x in re.findall(r"(?<![\d.])-?\d+",txt) if int(x)>=0]
    if not vals:return set(),"EMPTY_OR_NO_NUMERIC"
    z=set(vals);o={x-1 for x in vals if x>=1}
    vz=all(0<=x<n for x in z);vo=all(0<=x<n for x in o)
    if 0 in z and vz:return z,"ZERO_BASED_EXPLICIT0"
    if n in z and vo:return o,"ONE_BASED_EXPLICITN"
    if vz and not vo:return z,"ZERO_BASED_ONLY"
    if vo and not vz:return o,"ONE_BASED_ONLY"
    if vz:return z,"ZERO_BASED_SOURCE_CONVENTION"
    raise RuntimeError("bad-trial index out of range")

def read_member_bytes(url,e):
    p=CACHE/"small_members"/Path(e["name"])
    download_member(url,e,p,chunk=1_048_576)
    return p.read_bytes()

def objective_correctness(records):
    def attempt(respcol):
        pairs=[]
        for r in records:
            a=lower(r.get("stimitem",""));b=lower(r.get(respcol,""))
            if not a or not b:return None
            pairs.append((a,b))
        sa={a for a,b in pairs};sb={b for a,b in pairs}
        if not (sa&sb):return None
        if len(sa&sb)<max(1,min(len(sa),len(sb))//2):return None
        return np.asarray([a==b for a,b in pairs],bool)
    for c in ("respobj","respobjnum"):
        if records and c in records[0]:
            x=attempt(c)
            if x is not None:return x,f"stimitem=={c}"
    return None,"UNRESOLVED"

def behavioral_records(df,bad):
    det=[]
    for i,r in enumerate(df):
        if lower(r.get("cond"))!="detect":continue
        resp=lower(r.get("respsubjnum"))
        if resp not in ("nothing","something","identify","clear"):continue
        try:lev=float(str(r.get("stimlevel")).replace(",","."))
        except:continue
        det.append({"df_index":i,"response":resp,"aware":resp!="nothing","level":lev,
                    "stimitem":norm(r.get("stimitem","")),"respobj":norm(r.get("respobj","")),
                    "respobjnum":norm(r.get("respobjnum",""))})
    membership={};chunks=[]
    for bi in range(len(det)//16):
        b=det[16*bi:16*(bi+1)]
        na=sum(bool(x["aware"]) for x in b);valid=6<=na<=10
        meanlev=float(np.mean([x["level"] for x in b]));ch=bi+1
        chunks.append({"chunk":ch,"aware_count":na,"valid":valid,"mean_level":meanlev})
        for x in b:membership[x["df_index"]]={"chunk":ch,"valid":valid,"mean_level":meanlev}
    out=[]
    for x in det:
        m=membership.get(x["df_index"])
        if not m or not m["valid"] or x["df_index"] in bad:continue
        out.append({**x,"chunk":m["chunk"],"level_centered":x["level"]-m["mean_level"],
                    "primary_label":"UNSEEN" if x["response"]=="nothing" else
                                    "SEEN" if x["response"]=="something" else "OTHER"})
    return out,chunks

def sensor_positions(info,eeg_names):
    pos=[];good=True
    for name in eeg_names:
        ch=info["chs"][info.ch_names.index(name)]
        p=np.asarray(ch["loc"][:3],float)
        if (not np.isfinite(p).all()) or np.linalg.norm(p)<1e-6:
            good=False;break
        pos.append(p)
    if good:return np.asarray(pos,float),"FIF_LOC"
    import mne
    mon=mne.channels.make_standard_montage("standard_1020")
    mp={k.lower():np.asarray(v,float) for k,v in mon.get_positions()["ch_pos"].items()}
    pos=[];missing=[]
    for name in eeg_names:
        if name.lower() not in mp:missing.append(name)
        else:pos.append(mp[name.lower()])
    if missing:raise RuntimeError("missing sensor positions: "+",".join(missing[:10]))
    return np.asarray(pos,float),"STANDARD_1020_FALLBACK"

def build_complex(ep):
    import mne
    picks=mne.pick_types(ep.info,eeg=True,meg=False,eog=False,stim=False,misc=False,exclude=[])
    names=[ep.ch_names[i] for i in picks]
    if len(names)<int(CFG["sensor_complex"]["minimum_graph_channels"]):
        raise RuntimeError(f"only {len(names)} EEG channels")
    pos,ps=sensor_positions(ep.info,names)
    A,sigma=graph_knn(pos,k=6)
    h=build_hodge(A,
                  ncurl=int(CFG["hodge_gft"]["selected_curl_modes"]),
                  ngrad=int(CFG["hodge_gft"]["selected_gradient_modes"]),
                  min_triangles=int(CFG["sensor_complex"]["minimum_triangles"]))
    h.update({"names":names,"positions":pos,"position_source":ps,"A":A,"sigma":sigma})
    return h

def verify_complex(ep,h):
    miss=[x for x in h["names"] if x not in ep.ch_names]
    if miss:raise RuntimeError("graph channels missing: "+",".join(miss[:10]))

def baseline_standardize(X,times):
    a,b=map(float,CFG["preprocessing"]["baseline_standardization"]["window_s"])
    ix=np.flatnonzero((times>=a-1e-12)&(times<b-1e-12))
    if len(ix)<8:raise RuntimeError("too few baseline samples")
    mu=X[:,:,ix].mean(axis=(0,2));sd=X[:,:,ix].std(axis=(0,2))
    if np.any(sd<1e-8):raise RuntimeError("degenerate sensor baseline")
    return (X-mu[None,:,None])/sd[None,:,None]

def nuisance_residualize(X,records):
    n,v,t=X.shape
    levels=np.asarray([r["level_centered"] for r in records],float)
    cols=[np.ones(n),levels];names=["intercept","level_centered"]
    stim=[r["stimitem"] for r in records];uniq=sorted(set(x for x in stim if x!=""))
    if len(uniq)>=2:
        for u in uniq[1:]:
            cols.append(np.asarray([1.0 if x==u else 0.0 for x in stim]))
            names.append("stimitem="+u)
    corr,mode=objective_correctness(records)
    if corr is not None and len(set(corr.tolist()))>=2:
        cols.append(corr.astype(float));names.append("objective_correct")
    D=np.column_stack(cols)
    Y=X.reshape(n,v*t)
    R=Y-D@(np.linalg.pinv(D)@Y)
    return R.reshape(n,v,t),names,mode

def load_subject(subject,group,entries,url,h=None,open_samples=False):
    import mne
    epoch_name=CFG["dataset"]["epoch_member_template"].format(group=group,subject=subject)
    df_name=CFG["dataset"]["df_member_template"].format(group=group,subject=subject)
    if epoch_name not in entries or df_name not in entries:raise RuntimeError("required member missing")
    fields,df=parse_tsv_bytes(read_member_bytes(url,entries[df_name]))
    if not df:raise RuntimeError("df parse failed")
    bad=set();bmode="NO_BAD_FILE"
    for tpl in CFG["dataset"]["bad_member_preference"]:
        name=tpl.format(group=group,subject=subject)
        if name in entries:
            bad,bmode=bad_indices_from_bytes(read_member_bytes(url,entries[name]),len(df));break
    fif=CACHE/"epo"/group/subject/Path(epoch_name).name
    download_member(url,entries[epoch_name],fif,chunk=4_194_304)
    ep=mne.read_epochs(str(fif),preload=False,verbose="ERROR")
    if len(ep)!=len(df):raise RuntimeError("epoch/df mismatch")
    if abs(float(ep.info["sfreq"])-float(CFG["preprocessing"]["expected_sfreq_hz"]))>0.01:raise RuntimeError("sfreq")
    if float(ep.tmin)>float(CFG["preprocessing"]["required_tmin_max"]) or float(ep.tmax)<float(CFG["preprocessing"]["required_tmax_min"]):
        raise RuntimeError("epoch timing")
    beh,chunks=behavioral_records(df,bad)
    meta={"subject":subject,"group":group,"n_epochs":len(ep),"n_df":len(df),"n_bad":len(bad),
          "bad_mode":bmode,"valid_chunks":sum(x["valid"] for x in chunks),
          "UNSEEN":sum(x["primary_label"]=="UNSEEN" for x in beh),
          "SEEN":sum(x["primary_label"]=="SEEN" for x in beh),
          "OTHER":sum(x["primary_label"]=="OTHER" for x in beh)}
    if h is None:h=build_complex(ep)
    else:verify_complex(ep,h)
    if not open_samples:return meta,None,h
    idx=[r["df_index"] for r in beh]
    sub=ep[idx]
    picks=[sub.ch_names.index(x) for x in h["names"]]
    X=sub.get_data(picks=picks,verbose="ERROR")
    times=np.asarray(sub.times,float)
    X=baseline_standardize(X,times)
    X,names,objmode=nuisance_residualize(X,beh)
    meta["nuisance_columns"]=" | ".join(names);meta["objective_mode"]=objmode
    return meta,{"X":X,"times":times,"records":beh},h

def subject_signature(data,h):
    X=data["X"];times=data["times"];records=data["records"]
    labels=np.asarray([r["primary_label"] for r in records],dtype=object)
    primary=(labels=="UNSEEN")|(labels=="SEEN")
    Xp=X[primary];lp=labels[primary]
    if np.sum(lp=="UNSEEN")<int(CFG["behavior"]["minimum_primary_trials_per_label_after_modal_QC"]) or \
       np.sum(lp=="SEEN")<int(CFG["behavior"]["minimum_primary_trials_per_label_after_modal_QC"]):
        raise RuntimeError("primary trial count below frozen gate")

    lag=int(CFG["relational_edge_signal"]["lag_samples"])
    F=lag_flow(Xp,h["edges"],lag=lag)
    ftimes=times[lag:]

    Cc=project_coeff(F,h["curl_sel"])
    Cg=project_coeff(F,h["grad_sel"])
    wins=list(CFG["trajectory"]["windows_s"].items())
    qcurl,rawcurl=spectral_trajectory(Cc,lp,ftimes,wins)
    qgrad,rawgrad=spectral_trajectory(Cg,lp,ftimes,wins)

    # full subspace energy fractions
    Eg=total_subspace_energy(F,h["grad_basis"])
    Ec=total_subspace_energy(F,h["curl_basis"])
    total=np.sum(F*F,axis=1)
    Eh=np.maximum(total-Eg-Ec,0.0)

    fractions=[]
    for lab in ("UNSEEN","SEEN"):
        m=lp==lab
        for wname,w in wins:
            a,b=map(float,w)
            ix=np.flatnonzero((ftimes>=a-1e-12)&(ftimes<b-1e-12))
            tg=float(np.mean(Eg[m][:,ix]));tc=float(np.mean(Ec[m][:,ix]));th=float(np.mean(Eh[m][:,ix]))
            den=tg+tc+th+1e-15
            fractions.append({"label":lab,"window":wname,
                              "gradient_fraction":tg/den,"curl_fraction":tc/den,"harmonic_fraction":th/den,
                              "gradient_energy":tg,"curl_energy":tc,"harmonic_energy":th})
    return qcurl,qgrad,rawcurl,rawgrad,fractions

def complex_outputs(h):
    write_csv(RESULTS/"HGFT_X2_COMPLEX_SUMMARY.csv",[{
        "nodes":len(h["names"]),"edges":len(h["edges"]),"triangles":len(h["triangles"]),
        "rank_B1":h["rank_B1"],"rank_B2":h["rank_B2"],"harmonic_dim":h["harmonic_dim"],
        "boundary_error":h["boundary_error"],"position_source":h["position_source"],"sigma":h["sigma"],
        "n_gradient_modes":h["grad_basis"].shape[1],"n_curl_modes":h["curl_basis"].shape[1]
    }])

    mode_rows=[];load_rows=[]
    for typ,vals,U,sel in [
        ("GRADIENT",h["grad_values"],h["grad_basis"],h["grad_sel_idx"]),
        ("CURL",h["curl_values"],h["curl_basis"],h["curl_sel_idx"])
    ]:
        for sm,ci in enumerate(sel):
            mode_rows.append({"subspace":typ,"selected_mode":sm+1,"subspace_rank":int(ci)+1,"eigenvalue":float(vals[ci])})
            v=U[:,ci];top=np.argsort(np.abs(v))[::-1][:12]
            for e in top:
                i,j=h["edges"][e]
                load_rows.append({"subspace":typ,"selected_mode":sm+1,"subspace_rank":int(ci)+1,
                                  "eigenvalue":float(vals[ci]),"edge_index":int(e),
                                  "sensor_i":h["names"][i],"sensor_j":h["names"][j],
                                  "loading":float(v[e]),"abs_loading":float(abs(v[e]))})
    write_csv(RESULTS/"HGFT_X2_MODE_MAP.csv",mode_rows)
    write_csv(RESULTS/"HGFT_X2_MODE_TOP_EDGE_LOADINGS.csv",load_rows)
    np.savez_compressed(RESULTS/"HGFT_X2_COMPLEX.npz",
        adjacency=h["A"],B1=h["B1"],B2=h["B2"],Lgrad=h["Lgrad"],Lcurl=h["Lcurl"],
        grad_values=h["grad_values"],grad_basis=h["grad_basis"],grad_sel_idx=h["grad_sel_idx"],
        curl_values=h["curl_values"],curl_basis=h["curl_basis"],curl_sel_idx=h["curl_sel_idx"],
        positions=h["positions"],channel_names=np.asarray(h["names"],dtype=object),
        edges=np.asarray(h["edges"],int),triangles=np.asarray(h["triangles"],int))

def consensus_mode_differences(subject_data,h):
    rows=[]
    win_names=list(CFG["trajectory"]["windows_s"].keys())
    for subspace in ("curl","grad"):
        for wi,wname in enumerate(win_names):
            for mi in range(12):
                diffs=[]
                for s,d in subject_data.items():
                    r=d["rawcurl" if subspace=="curl" else "rawgrad"]
                    diffs.append(float(r["SEEN"][wi]["prob"][mi]-r["UNSEEN"][wi]["prob"][mi]))
                vals=h["curl_values"] if subspace=="curl" else h["grad_values"]
                sel=h["curl_sel_idx"] if subspace=="curl" else h["grad_sel_idx"]
                rows.append({"subspace":subspace.upper(),"window":wname,"selected_mode":mi+1,
                             "eigenvalue":float(vals[sel[mi]]),
                             "mean_seen_minus_unseen_probability":float(np.mean(diffs)),
                             "median_seen_minus_unseen_probability":float(np.median(diffs)),
                             "abs_mean_difference":float(abs(np.mean(diffs)))})
    rows=sorted(rows,key=lambda r:r["abs_mean_difference"],reverse=True)
    for i,r in enumerate(rows):r["abs_rank"]=i+1
    write_csv(RESULTS/"HGFT_X2_CONSENSUS_MODE_DIFFERENCE.csv",rows)
    write_csv(RESULTS/"HGFT_X2_TOP30_MODE_DIFFERENCES.csv",rows[:30])

def package(result):
    (RESULTS/"SGIT_HGFT_X2_RESULT.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    s=result.get("statistics") or {}
    (RESULTS/"SGIT_HGFT_X2_REPORT.md").write_text(f"""# SGIT HGFT-X2 relational Hodge-GFT discovery

Technical status: **{result['technical_status']}**

Exploratory scientific status: **{result['scientific_status']}**

Experiment 2 / VAN3 touched: **FALSE**

N evaluable: {result['evaluable_subjects']}

Primary Hodge-curl trajectory:
- median LOSO score: {s.get('curl_median_template_score')}
- mean LOSO score: {s.get('curl_mean_template_score')}
- positive subjects: {s.get('curl_positive_subjects')}
- swap p: {s.get('curl_permutation_p')}
- null q97.5: {s.get('curl_null_q975')}
- sign test: {s.get('curl_sign_test')}
- candidate discovered: {s.get('candidate_discovered')}

Gradient specificity control:
- median LOSO score: {s.get('gradient_median_template_score')}
- swap p: {s.get('gradient_permutation_p')}

This is exploratory because Experiment-1 EEG had already been opened.
""",encoding="utf-8")
    (STATE/"CONTINUATION_CHECKPOINT_HGFT_X2.md").write_text(
        f"CURRENT_STEP=HGFT_X2_COMPLETE\nTECHNICAL_STATUS={result['technical_status']}\n"
        f"SCIENTIFIC_STATUS={result['scientific_status']}\nN_EVALUABLE={result['evaluable_subjects']}\n"
        f"EXPERIMENT2_TOUCHED=FALSE\nNEXT={result['next_action']}\n",encoding="utf-8")
    stamp=time.strftime("%Y%m%d_%H%M%S")
    out=EXPORT/f"SGIT_HGFT_X2_RESULTS_{stamp}.zip"
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(RESULTS.glob("*")):
            if p.is_file():z.write(p,arcname=p.name)
        z.write(STATE/"CONTINUATION_CHECKPOINT_HGFT_X2.md",arcname="CONTINUATION_CHECKPOINT_HGFT_X2.md")
        z.write(ROOT/"config"/"HGFT_X2_FROZEN_CONFIG_1_0.json",arcname="provenance/HGFT_X2_FROZEN_CONFIG_1_0.json")
        z.write(ROOT/"protocol"/"HGFT_X2_FROZEN_PROTOCOL.md",arcname="provenance/HGFT_X2_FROZEN_PROTOCOL.md")
    log(f"RESULT ZIP: {out.resolve()}")

def main():
    if LOG.exists():LOG.unlink()
    log("HGFT-X2 START — relational edge flow first, Hodge-GFT second.")
    log("Experiment 2 / VAN3 hard-forbidden.")

    syn=run_synthetic(int(CFG["synthetic_sanity"]["seed"]))
    (RESULTS/"HGFT_X2_SYNTHETIC_SANITY.json").write_text(json.dumps(syn,indent=2),encoding="utf-8")
    log("Synthetic sanity: "+json.dumps(syn))
    if not syn["pass"]:
        result={"technical_status":"PASS","scientific_status":"HGFT_X2_SYNTHETIC_SANITY_FAILED_HUMAN_NOT_RUN",
                "evaluable_subjects":0,"statistics":None,"experiment2_touched":False,
                "next_action":"Fix/reject only the Hodge-relational estimator before human interpretation."}
        package(result);return

    url=CFG["dataset"]["archive_download_url"]
    req=urllib.request.Request(url,headers={"User-Agent":"SGIT-HGFT-X2/1.0","Range":"bytes=0-0"})
    with urllib.request.urlopen(req,timeout=90) as r:
        cr=r.headers.get("Content-Range","")
        if "/" not in cr:raise RuntimeError("archive size unavailable")
        size=int(cr.split("/")[-1])
    entries=central_directory(url,size)

    h=None;meta=[];ready=[]
    for i,c in enumerate(CFG["dataset"]["frozen_candidates"],1):
        s,g=c["subject"],c["group"];log(f"METADATA [{i}/{CFG['dataset']['n_candidates']}] {s}")
        try:
            m,_,h=load_subject(s,g,entries,url,h=h,open_samples=False)
            meta.append({**m,"status":"PASS"});ready.append((s,g))
        except Exception as e:
            meta.append({"subject":s,"group":g,"status":"FAIL","reason":str(e)});log(f"  FAIL {e}")
    write_csv(RESULTS/"HGFT_X2_METADATA_GATE.csv",meta)
    if h is None or len(ready)<int(CFG["behavior"]["minimum_evaluable_subjects"]):
        result={"technical_status":"TECHNICALLY_UNEVALUABLE","scientific_status":"HGFT_X2_METADATA_TOO_FEW",
                "evaluable_subjects":0,"statistics":None,"experiment2_touched":False,
                "next_action":"Only technical graph/metadata repair allowed; do not change frozen science or touch Experiment 2."}
        package(result);return

    complex_outputs(h)

    subject_data={};qc=[];frac_rows=[]
    for i,(s,g) in enumerate(ready,1):
        log(f"HODGE [{i}/{len(ready)}] {s}")
        try:
            m,data,h=load_subject(s,g,entries,url,h=h,open_samples=True)
            qc0=sum(r["primary_label"]=="UNSEEN" for r in data["records"])
            qc1=sum(r["primary_label"]=="SEEN" for r in data["records"])
            if qc0<int(CFG["behavior"]["minimum_primary_trials_per_label_after_modal_QC"]) or \
               qc1<int(CFG["behavior"]["minimum_primary_trials_per_label_after_modal_QC"]):
                raise RuntimeError(f"primary trials {qc0}/{qc1}")
            qcurl,qgrad,rawcurl,rawgrad,fractions=subject_signature(data,h)
            subject_data[s]={"group":g,"qcurl":qcurl,"qgrad":qgrad,
                             "rawcurl":rawcurl,"rawgrad":rawgrad}
            for r in fractions:frac_rows.append({"subject":s,"group":g,**r})
            qc.append({"subject":s,"group":g,"status":"PASS","UNSEEN":qc0,"SEEN":qc1,
                       "nuisance_columns":m.get("nuisance_columns",""),
                       "objective_mode":m.get("objective_mode","")})
        except Exception as e:
            qc.append({"subject":s,"group":g,"status":"FAIL","reason":str(e)});log(f"  FAIL {e}")
    write_csv(RESULTS/"HGFT_X2_QC.csv",qc)
    write_csv(RESULTS/"HGFT_X2_HODGE_ENERGY_FRACTIONS.csv",frac_rows)

    n=len(subject_data)
    if n<int(CFG["behavior"]["minimum_evaluable_subjects"]):
        result={"technical_status":"TECHNICALLY_UNEVALUABLE","scientific_status":"HGFT_X2_POST_QC_TOO_FEW",
                "evaluable_subjects":n,"statistics":None,"experiment2_touched":False,
                "next_action":"Do not lower frozen gates."}
        package(result);return

    subs=sorted(subject_data)
    Uc=np.stack([subject_data[s]["qcurl"]["UNSEEN"] for s in subs])
    Sc=np.stack([subject_data[s]["qcurl"]["SEEN"] for s in subs])
    Ug=np.stack([subject_data[s]["qgrad"]["UNSEEN"] for s in subs])
    Sg=np.stack([subject_data[s]["qgrad"]["SEEN"] for s in subs])

    cs,cobs,cnull,cp=swap_null(Uc,Sc,
                               int(CFG["primary_discovery_test"]["permutation_draws"]),
                               int(CFG["primary_discovery_test"]["seed"]))
    gs,gobs,gnull,gp=swap_null(Ug,Sg,
                               int(CFG["primary_discovery_test"]["permutation_draws"]),
                               int(CFG["primary_discovery_test"]["seed"])+1)

    cand=(n>=int(CFG["behavior"]["minimum_evaluable_subjects"]) and cobs>0 and
          cp<=float(CFG["primary_discovery_test"]["one_sided_alpha"]))

    write_csv(RESULTS/"HGFT_X2_SUBJECT_TEMPLATE_SCORES.csv",[
        {"subject":s,"group":subject_data[s]["group"],
         "curl_template_score":float(cs[i]),"gradient_template_score":float(gs[i])}
        for i,s in enumerate(subs)
    ])
    write_csv(RESULTS/"HGFT_X2_CURL_PERMUTATION_NULL.csv",[
        {"draw":i+1,"median_score":float(x)} for i,x in enumerate(cnull)
    ])
    write_csv(RESULTS/"HGFT_X2_GRADIENT_PERMUTATION_NULL.csv",[
        {"draw":i+1,"median_score":float(x)} for i,x in enumerate(gnull)
    ])

    consensus_mode_differences(subject_data,h)

    stats={
        "N_evaluable":n,
        "curl_median_template_score":float(cobs),
        "curl_mean_template_score":float(np.mean(cs)),
        "curl_positive_subjects":int(np.sum(cs>0)),
        "curl_fraction_positive":float(np.mean(cs>0)),
        "curl_permutation_p":float(cp),
        "curl_null_median":float(np.median(cnull)),
        "curl_null_q975":float(np.quantile(cnull,0.975)),
        "curl_sign_test":exact_sign_one(cs),
        "gradient_median_template_score":float(gobs),
        "gradient_mean_template_score":float(np.mean(gs)),
        "gradient_positive_subjects":int(np.sum(gs>0)),
        "gradient_permutation_p":float(gp),
        "gradient_null_q975":float(np.quantile(gnull,0.975)),
        "gradient_sign_test":exact_sign_one(gs),
        "alpha":float(CFG["primary_discovery_test"]["one_sided_alpha"]),
        "candidate_discovered":bool(cand),
        "complex_edges":len(h["edges"]),
        "complex_triangles":len(h["triangles"]),
        "harmonic_dimension":h["harmonic_dim"]
    }
    scientific=("HGFT_X2_EXPLORATORY_CURL_RELATIONAL_TRAJECTORY_DISCOVERED"
                if cand else
                "HGFT_X2_EXPLORATORY_CURL_RELATIONAL_TRAJECTORY_NOT_DISCOVERED")
    next_action=("Freeze exact HGFT-X2 curl signature and a separate confirmatory Experiment-2 protocol before any Experiment-2 access."
                 if cand else
                 "Close this exact HGFT-X2 lag-flow/curl-spectrum formulation on Experiment 1; do not retune it on the same EEG.")
    result={"technical_status":"PASS","scientific_status":scientific,"evaluable_subjects":n,
            "statistics":stats,"experiment2_touched":False,"exploratory_only":True,
            "next_action":next_action}
    package(result)

if __name__=="__main__":
    try:main()
    except Exception:
        traceback.print_exc()
        try:
            with LOG.open("a",encoding="utf-8") as f:traceback.print_exc(file=f)
        except Exception:pass
        raise SystemExit(99)
