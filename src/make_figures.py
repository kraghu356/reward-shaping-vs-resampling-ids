import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size':9.5,'font.family':'serif','axes.linewidth':0.8,
                 'savefig.dpi':300,'savefig.bbox':'tight'})
names=["Normal","DoS","Probe","Access","Malware","Other"]
Nc=np.array([266083,1229,2321,4485,757,5074],float); N=Nc.sum(); pi=Nc/N
lam=np.log(N/Nc); K=6

# ---- FIG 1: effective per-class weight under the four configurations ----
fig,ax=plt.subplots(figsize=(7.0,3.6))
cfgs=[("uniform sampler, plain reward",pi*1.0,'#c9ccd1'),
      ("uniform sampler, frequency reward",pi*lam,'#8fb3d9'),
      ("balanced sampler, plain reward",np.full(K,1/K),'#1f3b73'),
      ("balanced sampler, frequency reward",lam/K,'#c0392b')]
x=np.arange(K); w=0.2
for i,(lab,v,c) in enumerate(cfgs):
    ax.bar(x+(i-1.5)*w, v/v.sum(), w, label=lab, color=c, edgecolor='black', linewidth=0.4)
ax.axhline(1/K, ls='--', color='k', lw=0.8)
ax.text(5.42,1/K*1.15,'uniform (1/6)',fontsize=7.5,ha='right')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel('normalised effective\nper-class weight  $\\tilde{w}_c$')
ax.set_title('Effective per-class training weight implied by each configuration (UNSW-NB15, NF-v2)',fontsize=9.5)
ax.legend(fontsize=7.2,loc='lower right',framealpha=0.95); ax.grid(axis='y',ls=':',alpha=0.4)
fig.savefig('figures/fig_weights.png'); plt.close(fig)

# ---- FIG 2: the 2x2 factorial, tuned ----
sup={'nn':[0.6519,0.6445,0.6441,0.6573,0.6541],'ru':[0.6960,0.6942,0.6865,0.6735,0.6699],
     'pb':[0.7877,0.7913,0.7959,0.7955,0.7916],'rb':[0.7882,0.7866,0.7859,0.7905,0.7840]}
det={'nn':[0.9531,0.9513,0.9519,0.9537,0.9529],'ru':[0.9547,0.9530,0.9553,0.9549,0.9544],
     'pb':[0.9431,0.9426,0.9414,0.9431,0.9431],'rb':[0.8893,0.9128,0.8938,0.8909,0.8834]}
fig,axes=plt.subplots(1,2,figsize=(9.4,3.5))
for ax,D,lab,ylim in ((axes[0],sup,'supported macro recall',(0.60,0.83)),
                      (axes[1],det,'binary detection $F_1$',(0.86,0.98))):
    m=lambda k: np.mean(D[k]); s=lambda k: np.std(D[k],ddof=1)
    xs=[0,1]
    ax.errorbar(xs,[m('nn'),m('pb')],yerr=[s('nn'),s('pb')],marker='o',color='#1f3b73',
                capsize=3,lw=1.8,label='plain reward')
    ax.errorbar(xs,[m('ru'),m('rb')],yerr=[s('ru'),s('rb')],marker='s',color='#c0392b',
                capsize=3,lw=1.8,ls='--',label='frequency reward')
    ax.set_xticks(xs); ax.set_xticklabels(['uniform\nsampler','balanced\nsampler'])
    ax.set_ylabel(lab); ax.set_ylim(*ylim); ax.grid(ls=':',alpha=0.45); ax.legend(fontsize=8)
axes[0].set_title('Recall: lines converge (no additive benefit)',fontsize=9.5)
axes[1].set_title('Precision: reward costs detection $F_1$',fontsize=9.5)
fig.suptitle('Reward $\\times$ sampler factorial, validation-tuned, five seeds (UNSW-NB15)',fontsize=10,y=1.03)
fig.savefig('figures/fig_factorial.png'); plt.close(fig)

# ---- FIG 3: untuned vs tuned effect sizes ----
fig,ax=plt.subplots(figsize=(6.4,3.2))
labs=['sampler alone\n(vs neither)','reward alone\n(vs neither)','reward added to\nbalanced sampler']
unt=[0.1810,0.0250,-0.0168]; tun=[0.1420,0.0336,-0.0054]
x=np.arange(3); w=0.34
ax.bar(x-w/2,unt,w,label='fixed 8 epochs, raw features',color='#9aa7b8',edgecolor='black',linewidth=0.4)
ax.bar(x+w/2,tun,w,label='validation-tuned, encoded features',color='#1f3b73',edgecolor='black',linewidth=0.4)
ax.axhline(0,color='k',lw=0.9)
for i,(a,b) in enumerate(zip(unt,tun)):
    ax.text(i-w/2,a+(0.004 if a>0 else -0.012),f'{a:+.3f}',ha='center',fontsize=7.5)
    ax.text(i+w/2,b+(0.004 if b>0 else -0.012),f'{b:+.3f}',ha='center',fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=8.5)
ax.set_ylabel('$\\Delta$ supported macro recall')
ax.set_title('Effect sizes are robust to tuning; only the magnitude shifts',fontsize=9.5)
ax.legend(fontsize=7.8); ax.grid(axis='y',ls=':',alpha=0.45)
fig.savefig('figures/fig_tuning.png'); plt.close(fig)
print("figures:", __import__('os').listdir('figures'))
