import sys, json, importlib
import numpy as np
sys.path.insert(0,'/home/claude/OT-exp-7')
import moo_utils as U, problems as PM
pesa=importlib.import_module('algorithms.17_pesa2')
wm=importlib.import_module('algorithms.04_weight_metric')
moba=importlib.import_module('algorithms.29_moba')
OUT={}
P1=PM.make_problem('P1')
f=lambda x:(float((x-3)**2),float((x-7)**2))

# ================= Weight Metric =================
pool=[3.0,4.0,5.0,6.0,7.0]
F=np.array([f(x) for x in pool]); z=np.array([0.0,0.0])
rows=[]
for w in [(0.5,0.5),(0.8,0.2)]:
    for p in [1,2,'inf']:
        sc=[]
        for fx in F:
            d=np.abs(fx-z); w_=np.array(w)
            hand = (w_*d).max() if p=='inf' else (np.sum(w_*d**p))**(1.0/p)   # independent formula
            code = wm.weight_metric(fx,z,w,p=np.inf if p=='inf' else p)
            assert abs(hand-code)<1e-12
            sc.append(hand)
        k=int(np.argmin(sc))
        # solver in repo
        bx,bf,bs=wm.solve_weight_metric(P1,z,np.array(w),p=np.inf if p=='inf' else p,candidates=np.array(pool).reshape(-1,1))
        assert abs(bx[0]-pool[k])<1e-12
        rows.append(dict(w=w,p=p,scores=[round(s,4) for s in sc],best_x=pool[k],best_F=[round(v,4) for v in F[k]]))
OUT['wm']=dict(pool=pool,F=F.tolist(),rows=rows)

# ================= PESA-II =================
class Scripted:
    def __init__(s,ints=(),rands=()): s.i=list(ints); s.r=list(rands); s.log=[]
    def integers(s,n,size=None):
        if size is None: v=s.i.pop(0); assert 0<=v<n; s.log.append(('int',n,v)); return v
        v=[s.i.pop(0) for _ in range(size)]; s.log.append(('int',n,v)); return np.array(v)
    def random(s): return s.r.pop(0)
xs=np.array([2.0,3.5,3.8,6.0]); Fp=np.array([f(x) for x in xs])
nd=U.pareto_front(Fp,np.zeros(4))
arch_X=xs[nd].reshape(-1,1); arch_F=Fp[nd]
cells,coords=pesa.compute_grid_indices(arch_F,n_divs=4)
sq,mem=pesa.compute_squeeze_factors(cells)
# hand grid
fmin=arch_F.min(0); fmax=arch_F.max(0); norm=(arch_F-fmin)/(fmax-fmin)
hand_cells=[tuple(np.clip(np.floor(r*4).astype(int),0,3)) for r in norm]
assert hand_cells==cells
# scripted tournament: 4 parents. draws: (c1,c2) pairs then member pick
cl=list(mem.keys())
rng=Scripted(ints=[0,1,0, 1,1,0, 0,1,0, 0,0,1])  # tournament1 cells(0,1) -> member pick ; etc
# draw order per selection: integers(len(cells),size=2) then (if tie: random()) then integers(len(members))
rng=Scripted(ints=[0,1, 0,  1,1, 0,  0,0, 1,  0,1, 0],rands=[0.3,0.7])
# need check tie handling: cells with equal squeeze occurs when both draws same cell(1,1)->c1==c2 ties -> random()
sel=pesa.region_based_selection(mem,sq,4,rng)
OUT['pesa1']=dict(xs=xs.tolist(),F=Fp.tolist(),nd=[int(i) for i in nd],arch_F=arch_F.tolist(),norm=norm.tolist(),
    cells=[list(map(int,c)) for c in cells],squeeze={str(tuple(map(int,k))):v for k,v in sq.items()},cell_order=[list(map(int,c)) for c in cl],
    members={str(tuple(map(int,k))):v for k,v in mem.items()},sel=[int(i) for i in sel],log=[str(l) for l in rng.log],arch_X=arch_X.ravel().tolist())
# iteration 2: offspring x=4.6,5.2 added, capacity 4
off=np.array([4.6,5.2]); Fo=np.array([f(x) for x in off])
rng2=Scripted(ints=[0,0])  # crowded_cells choice, member choice
aX,aF,aC=pesa.update_archive(arch_X,arch_F,np.zeros(len(arch_X)),off.reshape(-1,1),Fo,np.zeros(2),max_size=4,n_divs=4,rng=rng2)
comb_F=np.vstack([arch_F,Fo]); c2,_=pesa.compute_grid_indices(comb_F,n_divs=4); sq2,mem2=pesa.compute_squeeze_factors(c2)
nrm2=(comb_F-comb_F.min(0))/(comb_F.max(0)-comb_F.min(0))
OUT['pesa2']=dict(off=off.tolist(),Fo=Fo.tolist(),comb_F=comb_F.tolist(),norm=nrm2.tolist(),cells=[list(map(int,c)) for c in c2],
   squeeze={str(tuple(map(int,k))):v for k,v in sq2.items()},members={str(tuple(map(int,k))):v for k,v in mem2.items()},
   final_x=aX.ravel().tolist(),final_F=aF.tolist(),log=[str(l) for l in rng2.log])

# ================= MOBA =================
class SRng:
    def __init__(s,ch,un,ra): s.ch=list(ch); s.un=list(un); s.ra=list(ra); s.log=[]
    def choice(s,n,p=None):
        v=s.ch.pop(0); v=min(v,n-1); s.log.append(('choice',n,v)); return v
    def uniform(s,a=0,b=1,size=None):
        v=s.un.pop(0); s.log.append(('uniform',v)); return v
    def random(s):
        v=s.ra.pop(0); s.log.append(('random',v)); return v
X=np.array([[2.0],[5.0],[9.0]]); V=np.zeros_like(X)
Fm,CVm=P1.evaluate(X)
ar=moba.update_archive(np.empty((0,1)),np.empty((0,2)),np.empty(0),X,Fm,CVm,100)
OUT['moba0']=dict(X=X.ravel().tolist(),F=Fm.tolist(),arch_X=ar[0].ravel().tolist(),arch_F=ar[1].tolist())
freq=np.zeros(3); loud=np.ones(3); pulse=np.full(3,0.5)
rg=SRng(ch=[1,0,1],un=[0.2,0.5,0.25],ra=[0.3,0.3,0.7,0.3])
Xc,Vc,Fc,CVc=X.copy(),V.copy(),Fm.copy(),CVm.copy()
res=moba.step(P1,Xc,Vc,Fc,CVc,freq,loud,pulse,ar[0],ar[1],ar[2],f_min=0.0,f_max=2.0,alpha=0.9,gamma=0.05,t=0,archive_size=100,rng=rg)
assert not rg.ch and not rg.un and not rg.ra,(rg.ch,rg.un,rg.ra)
Xc,Vc,Fc,CVc,fr,lo,pu,aX,aF,aC=res
OUT['moba1']=dict(X=Xc.ravel().tolist(),V=Vc.ravel().tolist(),F=Fc.tolist(),freq=fr.tolist(),loud=lo.tolist(),pulse=pu.tolist(),arch_X=aX.ravel().tolist(),arch_F=aF.tolist(),log=[str(l) for l in rg.log])
rg2=SRng(ch=[0,1,0],un=[0.2,0.2,0.2],ra=[0.02,0.9,0.02,0.9,0.02,0.9])
res2=moba.step(P1,Xc,Vc,Fc,CVc,fr,lo,pu,aX,aF,aC,f_min=0.0,f_max=2.0,alpha=0.9,gamma=0.05,t=1,archive_size=100,rng=rg2)
print('left',rg2.ch,rg2.un,rg2.ra)
Xd,Vd,Fd,CVd,fr2,lo2,pu2,aX2,aF2,aC2=res2
OUT['moba2']=dict(X=Xd.ravel().tolist(),V=Vd.ravel().tolist(),F=Fd.tolist(),freq=fr2.tolist(),loud=lo2.tolist(),pulse=pu2.tolist(),arch_X=aX2.ravel().tolist(),arch_F=aF2.tolist(),log=[str(l) for l in rg2.log],arch_before=aX.ravel().tolist())
json.dump(OUT,open('/home/claude/report/traces.json','w'),indent=1,default=float)
print(json.dumps(OUT,indent=0,default=float)[:6000])