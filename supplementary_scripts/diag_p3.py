import sys, importlib, json
import numpy as np
sys.path.insert(0,'/home/claude/OT-exp-7')
import problems as PM, operators as ops, moo_utils as U
wm=importlib.import_module('algorithms.04_weight_metric')
P3=PM.make_problem('P3')
rng=np.random.default_rng(101)
# (a) random-pool quality in the full 10-D problem, as used by algorithms 1-4 (1000 random candidates)
res=[]
for s in range(101,111):
    X=P3.random_solutions(1000,np.random.default_rng(s)); F,_=P3.evaluate(X)
    g=1+X[:,1:].sum(1); res.append(g.min())
print('min g over 1000 random candidates, 10 seeds: mean %.3f min %.3f (g=1 on true front)'%(np.mean(res),np.min(res)))
# (b) 1-D slice x2..x10=0 (true-front manifold): weight metric p=1,2,inf with 11 weights, 1001 candidates
x1=np.linspace(0,1,1001); Xs=np.zeros((1001,10)); Xs[:,0]=x1
F,_=P3.evaluate(Xs); z=np.array([0.0,0.0])
W=ops.weight_grid(2,n_weights=11)
out={}
for p in [1,2,np.inf]:
    pts=set()
    for w in W:
        sc=[wm.weight_metric(f,z,w,p=p) for f in F]; k=int(np.argmin(sc)); pts.add(round(float(x1[k]),3))
    out[str(p)]=sorted(pts)
    print('p=',p,'distinct x1 chosen:',len(pts),sorted(pts))
json.dump(dict(min_g_mean=float(np.mean(res)),min_g_min=float(np.min(res)),slice=out),open('/home/claude/report/diag_p3.json','w'))