import sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/home/claude/OT-exp-7')
import moo_utils as U
from experiments import runner as R, config as C
FIG='/home/claude/report/figs/'
runs=pd.read_csv('/home/claude/OT-exp-7/results/raw/runs_full.csv')
runs=runs.drop_duplicates(['task','alg_id','problem','seed','params'],keep='last')
summ=pd.read_csv('/home/claude/OT-exp-7/results/raw/summary_full.csv')
A=summ[summ.task=='A'].reset_index(drop=True)
plt.rcParams.update({'font.size':7,'axes.titlesize':7,'axes.labelsize':6.5,'xtick.labelsize':6,'ytick.labelsize':6,'figure.dpi':200,'axes.grid':True,'grid.alpha':.25})
names={a:v[1] for a,v in C.CATALOGUE.items()}
def load_front(a,pid,s=101,tag='full'):
    return np.atleast_2d(np.loadtxt(f'/home/claude/OT-exp-7/results/fronts/{tag}/a{a:02d}_{pid}_s{s}.csv',delimiter=','))
probs={}
def prob(pid):
    if pid not in probs: probs[pid]=R.load_problem(pid)
    return probs[pid]
# ---------- true-front HV (normalized space) ----------
tf_hv={}
for pid in ['P1','P2','P3','P4','P5','P6','P7','P10_M3','P10_M5','P11','P13','P14']:
    p=prob(pid); T=R.get_true_front(p)
    if T is None: continue
    Tn=p.normalize(T); tf_hv[pid]=float(U.hypervolume(Tn,p.hv_ref(),200000,0))
json.dump(tf_hv,open('/home/claude/report/tf_hv.json','w'))
print(tf_hv)
# ---------- Gallery ----------
combos=[(int(r.alg_id),r.problem) for r in A.itertuples()]
per=20
for pg in range(0,len(combos),per):
    chunk=combos[pg:pg+per]
    fig,axs=plt.subplots(5,4,figsize=(7.4,9.0)); axs=axs.ravel()
    for ax in axs[len(chunk):]: ax.axis('off')
    for ax,(a,pid) in zip(axs,chunk):
        row=A[(A.alg_id==a)&(A.problem==pid)].iloc[0]
        p=prob(pid); T=R.get_true_front(p)
        ttl=f"#{a} {names[a]} | {pid}"
        if a in (8,9):
            ax.axis('off'); 
            rk=R.load_algorithm(a)
            if a==8: r_,Cv=rk.run(p); sc=f"closeness C: "+", ".join(f"{p.suppliers[i]}={Cv[i]:.3f}" for i in r_[:3])
            else: r_,Q,cs=rk.run(p); sc=f"Q (lower=better): "+", ".join(f"{p.suppliers[i]}={Q[i]:.3f}" for i in r_[:3])
            ax.text(.02,.55,"ranking method: no front.\nTop-3: "+", ".join(p.suppliers[i] for i in r_[:3])+"\n"+sc,fontsize=6.5,va='center',transform=ax.transAxes)
            ax.set_title(ttl,fontsize=7); continue
        F=load_front(a,pid)
        i,j=(0,1)
        if T is not None and T.shape[1]>=2:
            o=np.argsort(T[:,0]); ax.plot(T[o,i],T[o,j],'-',color='0.55',lw=1,label='true front',zorder=1)
        ax.scatter(F[:,i],F[:,j],s=7,c='tab:red',alpha=.8,zorder=2,label='obtained')
        if T is not None and len(T)>1:
            lo=T.min(0);hi=T.max(0);pad=.35*(hi-lo+1e-9)
            ax.set_xlim(lo[i]-pad[i],hi[i]+pad[i]);ax.set_ylim(lo[j]-pad[j],hi[j]+pad[j])
        ig='n/a' if pd.isna(row.igd_mean) else f"{row.igd_mean:.3f}"
        ax.set_title(ttl+f"\nHV {row.hv_mean:.3f}±{row.hv_std:.3f} | IGD {ig} | n={row.n_front_mean:.0f}",fontsize=6.3)
        ax.tick_params(labelsize=5)
    fig.tight_layout(); fig.savefig(FIG+f'gallery_{pg//per+1}.png'); plt.close(fig)
# ---------- budget figure ----------
ev=A.groupby('alg_id').evals_used_mean.mean()
fig,ax=plt.subplots(figsize=(7.2,2.6)); ids=list(range(1,38))
ax.bar([str(i) for i in ids],[ev.get(i,0) for i in ids],color=['tab:blue' if ev.get(i,0)>=9500 else 'tab:orange' for i in ids])
ax.axhline(10000,color='k',ls='--',lw=.8); ax.set_ylabel('mean evaluations used'); ax.set_xlabel('algorithm number')
ax.set_title('Evaluations actually used per algorithm (dashed = 10,000 target; orange = below 95% of target)')
fig.tight_layout(); fig.savefig(FIG+'budget.png'); plt.close(fig)
# ---------- D1 bars ----------
D1=summ[summ.task=='D1']
fig,axs=plt.subplots(2,2,figsize=(7.4,6.6))
for r_,pid in enumerate(['P3','P5']):
    d=D1[D1.problem==pid].sort_values('hv_mean')
    axs[r_,0].barh([f"#{a} {n}" for a,n in zip(d.alg_id,d.alg_name)],d.hv_mean,xerr=d.hv_std,color='tab:blue',ecolor='k',error_kw={'lw':.6})
    axs[r_,0].set_title(f'{pid}: hypervolume (higher better), mean±std, 10 seeds')
    if pid in tf_hv: axs[r_,0].axvline(tf_hv[pid],color='r',ls='--',lw=.8)
    d=D1[D1.problem==pid].sort_values('igd_mean',ascending=False)
    axs[r_,1].barh([f"#{a} {n}" for a,n in zip(d.alg_id,d.alg_name)],d.igd_mean,xerr=d.igd_std,color='tab:green',ecolor='k',error_kw={'lw':.6})
    axs[r_,1].set_xscale('log'); axs[r_,1].set_title(f'{pid}: IGD (lower better, log axis)')
fig.tight_layout(); fig.savefig(FIG+'d1_bars.png'); plt.close(fig)
# ---------- D2 ----------
D2=summ[summ.task=='D2'].sort_values('alg_id')
nm=[f"#{a} {n}" for a,n in zip(D2.alg_id,D2.alg_name)]
normstat=[]
for a in D2.alg_id:
    v=[]
    for s in range(101,111):
        F=load_front(int(a),'P10_M5',s); v.append(np.linalg.norm(F,axis=1).mean())
    normstat.append((np.mean(v),np.std(v,ddof=1)))
json.dump({int(a):list(map(float,ns)) for a,ns in zip(D2.alg_id,normstat)},open('/home/claude/report/d2_norms.json','w'))
fig,axs=plt.subplots(1,4,figsize=(7.4,2.5))
axs[0].bar(nm,D2.hv_mean,yerr=D2.hv_std,color='tab:blue',error_kw={'lw':.6}); axs[0].set_title('HV (MC, higher better)')
axs[1].bar(nm,D2.igd_mean,yerr=D2.igd_std,color='tab:green',error_kw={'lw':.6}); axs[1].set_title('IGD (lower better)')
axs[2].bar(nm,[n[0] for n in normstat],yerr=[n[1] for n in normstat],color='tab:purple',error_kw={'lw':.6}); axs[2].axhline(1,color='r',ls='--',lw=.8); axs[2].set_title('mean ||f|| (true front = 1)')
axs[3].bar(nm,D2.evals_used_mean,color='tab:orange'); axs[3].axhline(10000,color='k',ls='--',lw=.8); axs[3].set_title('evaluations used')
for ax in axs: ax.tick_params(axis='x',rotation=60,labelsize=5.5)
fig.tight_layout(); fig.savefig(FIG+'d2.png'); plt.close(fig)
# ---------- D3 ----------
D3=summ[summ.task=='D3'].sort_values('alg_id'); p8=prob('P8')
fig,axs=plt.subplots(1,2,figsize=(7.4,3.0))
cols={14:'tab:blue',20:'tab:red',25:'tab:green',30:'tab:orange'}
for a in [14,20,25,30]:
    F=load_front(a,'P8',101); axs[0].scatter(F[:,0],F[:,1],s=14,c=cols[a],label=f"#{a} {names[a]}",alpha=.8)
san=pd.read_csv('/home/claude/OT-exp-7/results/problem_sanity.csv')
axs[0].set_xlabel('total distance'); axs[0].set_ylabel('total time'); axs[0].legend(fontsize=5.5); axs[0].set_title('P8 fronts, seed 101 (min both)')
axs[1].bar([f"#{a} {n}" for a,n in zip(D3.alg_id,D3.alg_name)],D3.hv_mean,yerr=D3.hv_std,color=[cols[a] for a in D3.alg_id],error_kw={'lw':.6}); axs[1].set_title('P8 hypervolume (normalized, ref 1.1)'); axs[1].tick_params(axis='x',rotation=20)
fig.tight_layout(); fig.savefig(FIG+'d3.png'); plt.close(fig)
# ---------- Task C ----------
Cr=runs[runs.task=='C']
spec={17:('n_divs',[4,8,16],['P3','P5']),29:('alpha',[0.7,0.9,0.99],['P3','P4']),4:('p',[1,2,float('inf')],['P1','P3'])}
check=[]
for a,(pn,vals,pl) in spec.items():
    fig,axs=plt.subplots(2,3,figsize=(7.4,4.6))
    for r_,pid in enumerate(pl):
        p=prob(pid); T=R.get_true_front(p)
        cmap=['tab:blue','tab:orange','tab:red']
        if T is not None:
            o=np.argsort(T[:,0]); axs[r_,0].plot(T[o,0],T[o,1],'-',color='0.6',lw=1)
        for v,cl in zip(vals,cmap):
            row,F=R.run_one(dict(task='C',alg=a,pid=pid,seed=101,params={pn:v}),save=False)
            stored=Cr[(Cr.alg_id==a)&(Cr.problem==pid)&(Cr.seed==101)&(Cr.params==json.dumps({pn:v},default=str))]
            same=bool(len(stored)) and abs(float(stored.hv.iloc[0])-float(row['hv']))<1e-9
            check.append((a,pid,str(v),same))
            axs[r_,0].scatter(F[:,0],F[:,1],s=9,c=cl,alpha=.7,label=f'{pn}={v}')
        if T is not None:
            lo=T.min(0);hi=T.max(0);pad=.35*(hi-lo+1e-9); axs[r_,0].set_xlim(lo[0]-pad[0],hi[0]+pad[0]);axs[r_,0].set_ylim(lo[1]-pad[1],hi[1]+pad[1])
        axs[r_,0].legend(fontsize=5); axs[r_,0].set_title(f'{names[a]} on {pid}: fronts (seed 101)')
        for c_,(m,lab) in enumerate([('hv','HV'),('igd','IGD')],1):
            data=[Cr[(Cr.alg_id==a)&(Cr.problem==pid)&(Cr.params==json.dumps({pn:v},default=str))][m].astype(float).values for v in vals]
            axs[r_,c_].boxplot(data,labels=[str(v) for v in vals]); axs[r_,c_].set_title(f'{pid}: {lab} over 10 seeds'); axs[r_,c_].set_xlabel(pn)
    fig.tight_layout(); fig.savefig(FIG+f'taskc_{a}.png'); plt.close(fig)
json.dump(check,open('/home/claude/report/taskc_repro.json','w'))
print('reproduced stored HV exactly:',sum(c[3] for c in check),'/',len(check))
# ---------- P3 diag figure ----------
import importlib
wm=importlib.import_module('algorithms.04_weight_metric')
d=json.load(open('/home/claude/report/diag_p3.json'))
fig,axs=plt.subplots(1,3,figsize=(7.4,2.4)); x=np.linspace(0,1,200)
for ax,(k,t) in zip(axs,[('1','p=1 (weighted sum)'),('2','p=2'),('inf','p=inf (Tchebycheff)')]):
    ax.plot(x,1-x**2,'-',color='0.6'); xs=np.array(d['slice'][k]); ax.scatter(xs,1-xs**2,c='r',s=12,zorder=3)
    ax.set_title(f'{t}: {len(xs)} distinct points'); ax.set_xlabel('f1'); ax.set_ylabel('f2')
fig.tight_layout(); fig.savefig(FIG+'p3diag.png'); plt.close(fig)
# ---------- P13 ----------
ex=json.load(open('/home/claude/report/extras.json'))
fig,axs=plt.subplots(1,2,figsize=(7.4,2.8))
for ax,k in zip(axs,['p13_k1','p13_k10']):
    e=ex[k]; 
    for lab,key,c in [('equal weights','equal_hist','tab:blue'),('GradNorm','gn_hist','tab:red'),('MGDA','mgda_hist','tab:green')]:
        h=np.array(e[key]); ax.plot(h[:,0],h[:,1],'o-',ms=3,lw=.8,c=c,label=lab)
    ax.set_xlabel('task-1 loss'); ax.set_ylabel('task-2 loss'); ax.set_title('P13a: gradient scale of task 2 = '+('1x' if k=='p13_k1' else '10x'))
    if k=='p13_k10': ax.set_xscale('symlog'); ax.set_yscale('symlog')
    ax.legend(fontsize=5.5)
fig.tight_layout(); fig.savefig(FIG+'p13.png'); plt.close(fig)
print('done')