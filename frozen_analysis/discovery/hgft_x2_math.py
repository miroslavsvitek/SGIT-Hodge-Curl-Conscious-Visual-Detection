from __future__ import annotations
import math
import numpy as np

def safe_unit(v,eps=1e-12):
    v=np.asarray(v,float)
    n=float(np.linalg.norm(v))
    if not np.isfinite(n) or n<eps:return None
    return v/n

def cosine(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    na=np.linalg.norm(a);nb=np.linalg.norm(b)
    if na<1e-12 or nb<1e-12:return 0.0
    return float(np.dot(a,b)/(na*nb))

def graph_knn(pos,k=6):
    pos=np.asarray(pos,float);n=len(pos)
    d=np.linalg.norm(pos[:,None,:]-pos[None,:,:],axis=2)
    np.fill_diagonal(d,np.inf)
    nbr=np.argsort(d,axis=1)[:,:int(k)]
    ds=[d[i,j] for i in range(n) for j in nbr[i]]
    sigma=float(np.median(ds))
    if not np.isfinite(sigma) or sigma<=0:raise RuntimeError("bad sigma")
    A=np.zeros((n,n),float)
    for i in range(n):
        for j in nbr[i]:
            w=math.exp(-(d[i,j]**2)/(2*sigma*sigma))
            A[i,j]=max(A[i,j],w);A[j,i]=max(A[j,i],w)
    # connected
    seen={0};stack=[0]
    while stack:
        i=stack.pop()
        for j in np.flatnonzero(A[i]>0):
            if int(j) not in seen:
                seen.add(int(j));stack.append(int(j))
    if len(seen)!=n:raise RuntimeError(f"graph disconnected {len(seen)}/{n}")
    return A,sigma

def clique_complex(A):
    A=np.asarray(A,float);n=A.shape[0]
    edges=[]
    edge_index={}
    for i in range(n):
        for j in range(i+1,n):
            if A[i,j]>0:
                edge_index[(i,j)]=len(edges);edges.append((i,j))
    triangles=[]
    for i in range(n):
        for j in range(i+1,n):
            if A[i,j]<=0:continue
            for k in range(j+1,n):
                if A[i,k]>0 and A[j,k]>0:
                    triangles.append((i,j,k))
    E=len(edges);T=len(triangles)
    B1=np.zeros((n,E),float)
    for e,(i,j) in enumerate(edges):
        B1[i,e]=-1.0;B1[j,e]=1.0
    B2=np.zeros((E,T),float)
    for q,(i,j,k) in enumerate(triangles):
        # [j,k] - [i,k] + [i,j]
        B2[edge_index[(j,k)],q]=1.0
        B2[edge_index[(i,k)],q]=-1.0
        B2[edge_index[(i,j)],q]=1.0
    return edges,triangles,B1,B2

def eig_positive(M,tol=1e-9):
    vals,U=np.linalg.eigh(np.asarray(M,float))
    idx=np.flatnonzero(vals>tol)
    vals=vals[idx];U=U[:,idx]
    for c in range(U.shape[1]):
        j=int(np.argmax(np.abs(U[:,c])))
        if U[j,c]<0:U[:,c]*=-1
    return vals,U

def evenly_spaced_cols(n,nsel):
    if n<nsel:raise RuntimeError(f"only {n} positive modes, need {nsel}")
    idx=np.rint(np.linspace(0,n-1,int(nsel))).astype(int)
    if len(np.unique(idx))!=nsel:
        idx=np.array([int(math.floor(i*(n-1)/(nsel-1))) for i in range(nsel)],int)
    if len(np.unique(idx))!=nsel:raise RuntimeError("mode selection duplicate")
    return idx

def build_hodge(A,ncurl=12,ngrad=12,min_triangles=20):
    edges,triangles,B1,B2=clique_complex(A)
    if len(triangles)<min_triangles:
        raise RuntimeError(f"only {len(triangles)} triangles")
    err=float(np.max(np.abs(B1@B2))) if B2.size else 0.0
    if err>=1e-10:raise RuntimeError(f"boundary identity failed: {err}")
    Lg=B1.T@B1
    Lc=B2@B2.T
    vg,Ug=eig_positive(Lg)
    vc,Uc=eig_positive(Lc)
    ig=evenly_spaced_cols(len(vg),ngrad)
    ic=evenly_spaced_cols(len(vc),ncurl)
    rg=np.linalg.matrix_rank(B1, tol=1e-9)
    rc=np.linalg.matrix_rank(B2, tol=1e-9)
    hdim=len(edges)-rg-rc
    return {
        "edges":edges,"triangles":triangles,"B1":B1,"B2":B2,
        "Lgrad":Lg,"Lcurl":Lc,
        "grad_values":vg,"grad_basis":Ug,"grad_sel_idx":ig,"grad_sel":Ug[:,ig],
        "curl_values":vc,"curl_basis":Uc,"curl_sel_idx":ic,"curl_sel":Uc[:,ic],
        "rank_B1":int(rg),"rank_B2":int(rc),"harmonic_dim":int(hdim),
        "boundary_error":err
    }

def lag_flow(X,edges,lag=1,eps=1e-12):
    # X: trials x nodes x time
    X=np.asarray(X,float);n,_,T=X.shape
    if lag<1 or T<=lag:raise RuntimeError("invalid lag")
    a=np.array([i for i,j in edges],int);b=np.array([j for i,j in edges],int)
    xi0=X[:,a,:-lag]; xi1=X[:,a,lag:]
    xj0=X[:,b,:-lag]; xj1=X[:,b,lag:]
    num=xi0*xj1-xj0*xi1
    den=np.sqrt((xi0*xi0+xi1*xi1)*(xj0*xj0+xj1*xj1))+eps
    F=num/den
    F=np.clip(F,-1.0,1.0)
    return F

def project_coeff(F,U):
    # F: trials x edges x time; U: edges x modes
    return np.einsum("em,net->nmt",U,F,optimize=True)

def spectral_trajectory(coeff,labels,times,windows):
    # coeff: n x modes x t
    labels=np.asarray(labels,dtype=object)
    out={};raw={}
    for lab in ("UNSEEN","SEEN"):
        m=labels==lab
        if not np.any(m):raise RuntimeError(f"no {lab}")
        parts=[];raw[lab]=[]
        for _,w in windows:
            a,b=map(float,w)
            ix=np.flatnonzero((times>=a-1e-12)&(times<b-1e-12))
            if len(ix)<8:raise RuntimeError("too few window samples")
            e=np.mean(coeff[m][:,:,ix]**2,axis=(0,2))
            if not np.isfinite(e).all() or float(np.sum(e))<=1e-15:
                raise RuntimeError("degenerate spectral energy")
            p=e/np.sum(e)
            h=np.sqrt(p)  # Hellinger sphere
            parts.append(h)
            raw[lab].append({"energy":e,"prob":p})
        q=safe_unit(np.concatenate(parts))
        if q is None:raise RuntimeError("degenerate trajectory")
        out[lab]=q
    return out,raw

def total_subspace_energy(F,U):
    C=project_coeff(F,U)
    return np.sum(C*C,axis=1)  # trials x time

def loso_scores(U,S):
    U=np.asarray(U,float);S=np.asarray(S,float);n=len(U)
    sumU=U.sum(0);sumS=S.sum(0);out=np.zeros(n,float)
    for i in range(n):
        muU=safe_unit((sumU-U[i])/(n-1));muS=safe_unit((sumS-S[i])/(n-1))
        if muU is None or muS is None:raise RuntimeError("degenerate template")
        out[i]=0.5*((cosine(S[i],muS)-cosine(S[i],muU))+
                    (cosine(U[i],muU)-cosine(U[i],muS)))
    return out

def swap_null(U,S,draws,seed):
    U=np.asarray(U,float);S=np.asarray(S,float);n=len(U)
    scores=loso_scores(U,S);obs=float(np.median(scores))
    rng=np.random.default_rng(int(seed));null=np.zeros(int(draws),float)
    for b in range(int(draws)):
        sw=rng.random(n)<0.5
        Up=U.copy();Sp=S.copy();Up[sw]=S[sw];Sp[sw]=U[sw]
        null[b]=float(np.median(loso_scores(Up,Sp)))
    p=float((1+np.sum(null>=obs))/(1+len(null)))
    return scores,obs,null,p

def exact_sign_one(vals):
    vals=np.asarray(vals,float);vals=vals[np.isfinite(vals)&(vals!=0)]
    n=len(vals);k=int(np.sum(vals>0))
    if n==0:return {"n":0,"k_pos":0,"p":None}
    p=sum(math.comb(n,i) for i in range(k,n+1))/(2**n)
    return {"n":n,"k_pos":k,"p":float(p)}
