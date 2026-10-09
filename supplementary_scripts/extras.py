import sys, json, numpy as np
sys.path.insert(0,'/home/claude/OT-exp-7')
from experiments import runner as R
import problems as PM
OUT={}
# ---- P9: smallest feature set within 1 percentage point of all-ones baseline (error 0.0351)
base=0.0351
res={}
for a,n in [(23,'MOGWO'),(30,'MOMA')]:
    best=[]
    for s in range(101,111):
        F=np.loadtxt(f'results/fronts/full/a{a:02d}_P9_s{s}.csv',delimiter=',')
        ok=F[F[:,0]<=base+0.01]
        best.append((int(ok[:,1].min()), float(ok[ok[:,1].argmin(),0])) if len(ok) else None)
    res[n]=best
OUT['p9']=res
# ---- P13: equal weights vs MGDA vs GradNorm, baseline and gradient-scale stress test
class Scaled(PM.ToyTwoTask):
    def __init__(s,k2): super().__init__(); s.k2=k2
    def _evaluate_one(s,x):
        f,c=super()._evaluate_one(x); f=f.copy(); f[1]*=s.k2; return f,c
    def gradients(s,x):
        g=super().gradients(x).copy(); g[1]*=s.k2; return g
mg=R.load_algorithm(36); gn=R.load_algorithm(37)
def equal(p,steps=100,lr=0.01):
    x=p.start.copy(); L=[p.evaluate(x)[0].ravel().tolist()]
    for _ in range(steps):
        x=x-lr*p.gradients(x).sum(0); L.append(p.evaluate(x)[0].ravel().tolist())
    return x,np.array(L)
for k2 in [1,10]:
    p=Scaled(k2) if k2!=1 else PM.ToyTwoTask()
    xe,Le=equal(p)
    th,Lg,W=gn.run(p,n_steps=100,lr_theta=0.01,lr_w=0.025,alpha_asym=0.5,seed=101)
    tx,tf=mg.run(p,max_steps=100,eta=0.1)
    tx=np.array(tx); tf=np.array(tf)
    OUT[f'p13_k{k2}']=dict(equal_final_x=xe.tolist(),equal_final_L=Le[-1].tolist(),equal_L0=Le[0].tolist(),equal_hist=Le[::10].tolist(),
        gn_final_x=np.array(th).tolist(),gn_final_L=Lg[-1].tolist(),gn_w_final=W[-1].tolist(),gn_hist=Lg[::10].tolist(),gn_w_hist=W[::10].tolist(),
        mgda_steps=len(tx),mgda_final_x=tx[-1].tolist(),mgda_final_L=tf[-1].tolist(),mgda_hist=tf[::10].tolist())
    print(k2,json.dumps(OUT[f'p13_k{k2}'])[:900])
json.dump(OUT,open('/home/claude/report/extras.json','w'),indent=1)
print(res)