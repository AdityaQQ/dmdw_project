import numpy as np, pandas as pd, matplotlib, json, base64, io, warnings, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, r2_score, mean_squared_error,
    roc_curve, auc, classification_report, silhouette_score)
from scipy.cluster.hierarchy import dendrogram, linkage

# ═══════════════════════════════════════════════════════════
# OUTPUT FOLDER — saves beside this script on ANY OS
# ═══════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "output_figures")
os.makedirs(OUT_DIR, exist_ok=True)
FIG_N = [0]

def save(fig, name):
    FIG_N[0] += 1
    path = os.path.join(OUT_DIR, f"fig{FIG_N[0]:02d}_{name}.png")
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
    print(f"  ✅ Saved → {path}")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════
# COLORS & STYLE
# ═══════════════════════════════════════════════════════════
BG="#0d0f14"; BG2="#13161e"; BG3="#1a1d27"
TEAL="#2ec4b6"; RED="#e63946"; GOLD="#f4a261"
BLUE="#457b9d"; PUR="#a78bfa"; GRN="#06d6a0"
MUTED="#7b8099"; TEXT="#e8eaf0"
PAL=[TEAL,RED,GOLD,BLUE,PUR,GRN,"#ff9f1c","#8ecae6"]

plt.rcParams.update({
    "figure.facecolor":BG,"axes.facecolor":BG2,"axes.edgecolor":BG3,
    "axes.labelcolor":TEXT,"xtick.color":MUTED,"ytick.color":MUTED,
    "text.color":TEXT,"grid.color":BG3,"grid.alpha":0.8,
    "axes.titlecolor":TEXT,"axes.titlesize":12,"axes.labelsize":10,
    "legend.facecolor":BG3,"legend.edgecolor":BG3,"legend.fontsize":8,
    "figure.dpi":110,"lines.linewidth":2
})

# ═══════════════════════════════════════════════════════════
# LOAD & PREPROCESS
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  DMDW PROJECT — Commercial Housing ML Analysis")
print("="*60)
print("\n📂 Loading dataset...")

df = pd.read_csv(os.path.join(SCRIPT_DIR, 'housing.csv'))
df['Price_Category'] = df['Price_Category'].astype(str)

le_city = LabelEncoder(); df['City_Code'] = le_city.fit_transform(df['City_Type'])
le_cat  = LabelEncoder(); df['Cat_Code']  = le_cat.fit_transform(df['Price_Category'])
CAT     = le_cat.classes_   # Budget, Luxury, Mid-Range, Premium

NUM_FEATS = ['Area_sqft','Bedrooms','Bathrooms','House_Age','Floors',
             'Garage_Cars','Distance_km','School_Rating','Crime_Rate',
             'Pool','Renovated','City_Code']

X_all  = df[NUM_FEATS].values
y_clf  = df['Cat_Code'].values
y_reg  = df['Price'].values

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_all)

all_idx = np.arange(len(X_sc))
tr_idx, te_idx, yc_tr, yc_te = train_test_split(
    all_idx, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
Xtr, Xte   = X_sc[tr_idx], X_sc[te_idx]
yr_tr, yr_te = y_reg[tr_idx], y_reg[te_idx]

print(f"  Dataset  : {df.shape[0]} rows × {len(NUM_FEATS)} features")
print(f"  Classes  : {CAT.tolist()}")
print(f"  Train    : {len(Xtr)} samples")
print(f"  Test     : {len(Xte)} samples")

pca2 = PCA(n_components=2, random_state=42)
X_2d = pca2.fit_transform(X_sc)

# ═══════════════════════════════════════════════════════════
# HELPER: Confusion Matrix panel
# ═══════════════════════════════════════════════════════════
def draw_cm(ax, y_true, y_pred, title, cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=CAT, yticklabels=CAT, ax=ax,
                linewidths=1, linecolor=BG,
                annot_kws={'size':11,'weight':'bold'})
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.tick_params(labelsize=8)
    return cm

def draw_roc(ax, y_true, y_prob, title):
    yb = label_binarize(y_true, classes=list(range(len(CAT))))
    for i,(c,col) in enumerate(zip(CAT,PAL)):
        fpr,tpr,_ = roc_curve(yb[:,i], y_prob[:,i])
        ax.plot(fpr,tpr,color=col,lw=1.8,label=f"{c} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1],[0,1],'--',color=MUTED)
    ax.set_title(title); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=7)

def metrics_bar(ax, y_true, y_pred, title):
    vals=[accuracy_score(y_true,y_pred),
          precision_score(y_true,y_pred,average='weighted'),
          recall_score(y_true,y_pred,average='weighted'),
          f1_score(y_true,y_pred,average='weighted')]
    bars=ax.bar(['Accuracy','Precision','Recall','F1'],vals,color=PAL[:4],edgecolor=BG)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                f'{v:.4f}',ha='center',fontsize=9,color=TEXT)
    ax.set_title(title); ax.set_ylim(0,1.18); ax.tick_params(axis='x',labelsize=9)
    return vals

def perclass_f1(ax, y_true, y_pred, title):
    rep=classification_report(y_true,y_pred,target_names=CAT,output_dict=True)
    f1s=[rep[c]['f1-score'] for c in CAT]
    bars=ax.bar(CAT,f1s,color=PAL[:4],edgecolor=BG)
    for bar,v in zip(bars,f1s):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                f'{v:.2f}',ha='center',fontsize=9,color=TEXT)
    ax.set_title(title); ax.set_ylim(0,1.2); ax.tick_params(axis='x',labelsize=8)

def lc(ax, model, title):
    sz,tr_s,cv_s=learning_curve(model,X_sc,y_clf,cv=5,
        train_sizes=np.linspace(0.1,1,8),scoring='accuracy',n_jobs=-1)
    ax.fill_between(sz,tr_s.mean(1)-tr_s.std(1),tr_s.mean(1)+tr_s.std(1),alpha=0.2,color=TEAL)
    ax.fill_between(sz,cv_s.mean(1)-cv_s.std(1),cv_s.mean(1)+cv_s.std(1),alpha=0.2,color=RED)
    ax.plot(sz,tr_s.mean(1),'o-',color=TEAL,label='Train')
    ax.plot(sz,cv_s.mean(1),'o-',color=RED,label='CV')
    ax.set_title(title); ax.set_xlabel("Training Size"); ax.set_ylabel("Accuracy"); ax.legend()

# ═══════════════════════════════════════════════════════════════════════════════════
# ███  KNN — 6 FIGURES
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  KNN — K-Nearest Neighbours")
print("="*60)

k_vals=[1,3,5,7,9,11,13,15]
km_res={}
for k in k_vals:
    m=KNeighborsClassifier(n_neighbors=k).fit(Xtr,yc_tr); p=m.predict(Xte)
    km_res[k]=dict(acc=accuracy_score(yc_te,p),f1=f1_score(yc_te,p,average='weighted'),
                   prec=precision_score(yc_te,p,average='weighted'),
                   rec=recall_score(yc_te,p,average='weighted'),model=m,pred=p)
bk = max(km_res,key=lambda k:km_res[k]['f1'])
knn_m   = km_res[bk]['model']
knn_p   = km_res[bk]['pred']
knn_prob= knn_m.predict_proba(Xte)
print(f"  Best K={bk}  Acc={km_res[bk]['acc']:.4f}  F1={km_res[bk]['f1']:.4f}")

# ── KNN Fig 1: Metrics vs K
fig=plt.figure(figsize=(16,6),facecolor=BG)
fig.suptitle(f"KNN — Metrics vs K Value  (Best K={bk})",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.35)
ax=fig.add_subplot(g[0,0])
for key,col,lbl in [('acc',TEAL,'Accuracy'),('f1',RED,'F1'),('prec',GOLD,'Precision'),('rec',BLUE,'Recall')]:
    ax.plot(k_vals,[km_res[k][key] for k in k_vals],'o-',color=col,label=lbl,markersize=7)
ax.axvline(bk,color=PUR,ls='--',lw=2,label=f'Best K={bk}')
ax.set_title("All Metrics vs K"); ax.set_xlabel("K (Neighbours)"); ax.set_ylabel("Score")
ax.set_xticks(k_vals); ax.legend(fontsize=8); ax.set_ylim(0.3,1.05)
ax=fig.add_subplot(g[0,1])
accs=[km_res[k]['acc'] for k in k_vals]
f1s_k=[km_res[k]['f1'] for k in k_vals]
ax.bar(np.array(k_vals)-0.3,[a for a in accs],0.55,color=TEAL,label='Accuracy',edgecolor=BG)
ax.bar(np.array(k_vals)+0.3,[f for f in f1s_k],0.55,color=RED,label='F1',edgecolor=BG)
ax.set_title("Accuracy & F1 Bar per K"); ax.set_xlabel("K"); ax.set_xticks(k_vals); ax.legend()
plt.tight_layout(); save(fig,"KNN_metrics_vs_K")

# ── KNN Fig 2: Confusion Matrix
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle(f"KNN — Confusion Matrix (K={bk})",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
draw_cm(ax,yc_te,knn_p,f"Confusion Matrix  K={bk}  Acc={km_res[bk]['acc']:.4f}",'Blues')
plt.tight_layout(); save(fig,"KNN_confusion_matrix")

# ── KNN Fig 3: ROC Curves
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle(f"KNN — ROC Curves (K={bk})",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
draw_roc(ax,yc_te,knn_prob,f"ROC Curves (One-vs-Rest)  K={bk}")
plt.tight_layout(); save(fig,"KNN_ROC_curves")

# ── KNN Fig 4: Per-Class F1
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle(f"KNN — Per-Class F1 Score (K={bk})",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
perclass_f1(ax,yc_te,knn_p,f"Per-Class F1  (K={bk})")
plt.tight_layout(); save(fig,"KNN_perclass_F1")

# ── KNN Fig 5: Learning Curve
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle(f"KNN — Learning Curve (K={bk})",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
lc(ax,KNeighborsClassifier(n_neighbors=bk),f"Learning Curve  K={bk}")
plt.tight_layout(); save(fig,"KNN_learning_curve")

# ── KNN Fig 6: Final Metrics Summary
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle(f"KNN — Metrics Summary (K={bk})",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
knn_met=metrics_bar(ax,yc_te,knn_p,f"Accuracy={km_res[bk]['acc']:.4f}  F1={km_res[bk]['f1']:.4f}")
plt.tight_layout(); save(fig,"KNN_metrics_summary")

print(f"  KNN: 6 figures saved ✅")

# ═══════════════════════════════════════════════════════════════════════════════════
# ███  LINEAR REGRESSION — 6 FIGURES
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  LINEAR REGRESSION")
print("="*60)

lr_reg = LinearRegression().fit(Xtr, yr_tr)
rid    = Ridge(alpha=1.0).fit(Xtr, yr_tr)
las    = Lasso(alpha=500, max_iter=5000).fit(Xtr, yr_tr)
lr_pred  = lr_reg.predict(Xte)
rid_pred = rid.predict(Xte)
las_pred = las.predict(Xte)
r2_lr    = r2_score(yr_te, lr_pred)
rmse_lr  = np.sqrt(mean_squared_error(yr_te, lr_pred))
res      = yr_te - lr_pred
print(f"  OLS  R²={r2_lr:.4f}  RMSE=${rmse_lr/1000:.1f}K")

# ── Reg Fig 1: Actual vs Predicted
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle("Linear Regression — Actual vs Predicted",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
sp_colors=[PAL[int(yc_te[i])] for i in range(len(yc_te))]
ax.scatter(yr_te/1000,lr_pred/1000,c=sp_colors,s=18,alpha=0.7,edgecolors='none')
mn,mx=yr_te.min()/1000,yr_te.max()/1000
ax.plot([mn,mx],[mn,mx],'--',color=RED,lw=2,label='Perfect fit')
for i,name in enumerate(CAT): ax.scatter([],[],c=PAL[i],label=name,s=50)
ax.set_xlabel("Actual Price ($K)"); ax.set_ylabel("Predicted Price ($K)")
ax.set_title(f"Actual vs Predicted   R²={r2_lr:.4f}   RMSE=${rmse_lr/1000:.1f}K")
ax.legend(fontsize=8)
plt.tight_layout(); save(fig,"REG_actual_vs_predicted")

# ── Reg Fig 2: Residuals Plot
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle("Linear Regression — Residuals Plot",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
ax.scatter(lr_pred/1000,res/1000,c=GOLD,s=15,alpha=0.7,edgecolors='none')
ax.axhline(0,color=RED,lw=2,ls='--',label='Zero line')
ax.set_xlabel("Predicted Price ($K)"); ax.set_ylabel("Residual ($K)")
ax.set_title("Residuals vs Predicted (random scatter = good model)")
ax.legend()
plt.tight_layout(); save(fig,"REG_residuals")

# ── Reg Fig 3: Residual Distribution
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle("Linear Regression — Residual Distribution",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
ax.hist(res/1000,bins=35,color=BLUE,edgecolor=BG,alpha=0.85)
ax.axvline(0,color=RED,lw=2,ls='--',label='Mean=0')
ax.set_xlabel("Residual ($K)"); ax.set_ylabel("Count")
ax.set_title("Residual Histogram (bell shape = good)"); ax.legend()
plt.tight_layout(); save(fig,"REG_residual_distribution")

# ── Reg Fig 4: OLS vs Ridge vs Lasso
fig=plt.figure(figsize=(10,6),facecolor=BG)
fig.suptitle("Linear Regression — OLS vs Ridge vs Lasso",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.35)
ax=fig.add_subplot(g[0,0])
models_r=['OLS','Ridge','Lasso']
r2s=[r2_lr,r2_score(yr_te,rid_pred),r2_score(yr_te,las_pred)]
rmses=[np.sqrt(mean_squared_error(yr_te,p))/1000 for p in [lr_pred,rid_pred,las_pred]]
x3=np.arange(3); ax2b=ax.twinx()
ax.bar(x3-0.2,r2s,0.38,color=TEAL,edgecolor=BG,label='R²')
ax2b.bar(x3+0.2,rmses,0.38,color=RED,edgecolor=BG,label='RMSE($K)',alpha=0.85)
ax.set_xticks(x3); ax.set_xticklabels(models_r)
ax.set_ylabel("R²",color=TEAL); ax2b.set_ylabel("RMSE ($K)",color=RED)
ax.set_title("R² and RMSE Comparison"); ax.set_ylim(0,1.2)
for i,(r,rm) in enumerate(zip(r2s,rmses)):
    ax.text(i-0.2,r+0.02,f'{r:.3f}',ha='center',fontsize=8,color=TEAL)
    ax2b.text(i+0.2,rm+0.5,f'${rm:.1f}K',ha='center',fontsize=8,color=RED)
ax=fig.add_subplot(g[0,1])
alphas=[0.001,0.01,0.1,1,10,100,1000]
r_ridge=[Ridge(alpha=a).fit(Xtr,yr_tr).score(Xte,yr_te) for a in alphas]
r_lasso=[Lasso(alpha=a*100,max_iter=5000).fit(Xtr,yr_tr).score(Xte,yr_te) for a in alphas]
ax.semilogx(alphas,r_ridge,'o-',color=TEAL,label='Ridge',markersize=7)
ax.semilogx(alphas,r_lasso,'s-',color=RED,label='Lasso',markersize=7)
ax.set_title("R² vs Alpha (Regularisation)"); ax.set_xlabel("Alpha"); ax.set_ylabel("R²"); ax.legend()
plt.tight_layout(); save(fig,"REG_model_comparison")

# ── Reg Fig 5: Feature Coefficients
fig=plt.figure(figsize=(10,6),facecolor=BG)
fig.suptitle("Linear Regression — Feature Coefficients",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
coef=pd.Series(lr_reg.coef_,index=NUM_FEATS).sort_values()
colors_c=[RED if v<0 else TEAL for v in coef.values]
bars=ax.barh(coef.index,coef.values/1000,color=colors_c,edgecolor=BG)
ax.axvline(0,color=MUTED,lw=1.5,ls='--')
for bar,v in zip(bars,coef.values/1000):
    ax.text(v+(0.5 if v>=0 else -0.5),bar.get_y()+bar.get_height()/2,
            f'{v:.1f}',va='center',fontsize=8,color=TEXT)
ax.set_xlabel("Coefficient ($K per unit)"); ax.set_title("Feature Impact on Price  (Teal=positive, Red=negative)")
ax.tick_params(labelsize=9)
plt.tight_layout(); save(fig,"REG_coefficients")

# ── Reg Fig 6: Prediction Error Distribution
fig=plt.figure(figsize=(10,6),facecolor=BG)
fig.suptitle("Linear Regression — Prediction Error Analysis",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.35)
ax=fig.add_subplot(g[0,0])
pct_err=(np.abs(res)/yr_te)*100
ax.hist(pct_err,bins=30,color=PUR,edgecolor=BG,alpha=0.85)
ax.axvline(pct_err.mean(),color=RED,lw=2,ls='--',label=f'Mean={pct_err.mean():.1f}%')
ax.set_xlabel("% Error"); ax.set_ylabel("Count")
ax.set_title("Percentage Error Distribution"); ax.legend()
ax=fig.add_subplot(g[0,1])
sorted_actual=np.sort(yr_te/1000)
sorted_pred=np.sort(lr_pred/1000)
ax.plot(sorted_actual,sorted_pred,'.',color=TEAL,markersize=4,alpha=0.6)
mn2,mx2=sorted_actual.min(),sorted_actual.max()
ax.plot([mn2,mx2],[mn2,mx2],'--',color=RED,lw=2,label='Perfect')
ax.set_xlabel("Actual (sorted, $K)"); ax.set_ylabel("Predicted (sorted, $K)")
ax.set_title("Q-Q Style: Sorted Actual vs Predicted"); ax.legend()
plt.tight_layout(); save(fig,"REG_error_analysis")

print(f"  Regression: 6 figures saved ✅")

# ═══════════════════════════════════════════════════════════════════════════════════
# ███  K-MEANS — 6 FIGURES
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  K-MEANS CLUSTERING")
print("="*60)

k_range=range(2,9); inerts=[]; sils_km=[]
for k in k_range:
    m=KMeans(n_clusters=k,random_state=42,n_init=10).fit(X_sc)
    inerts.append(m.inertia_); sils_km.append(silhouette_score(X_sc,m.labels_))
bkm=list(k_range)[np.argmax(sils_km)]
km_best=KMeans(n_clusters=bkm,random_state=42,n_init=10).fit(X_sc)
kml=km_best.labels_; kmc=pca2.transform(km_best.cluster_centers_)
sil_km=silhouette_score(X_sc,kml)
print(f"  Best K={bkm}  Silhouette={sil_km:.4f}")

# ── KM Fig 1: Elbow Curve
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle("K-Means — Elbow Curve (Inertia vs K)",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
ax.plot(list(k_range),inerts,'o-',color=TEAL,markersize=10,lw=2.5)
ax.axvline(bkm,color=RED,ls='--',lw=2,label=f'Best K={bkm}')
for k,v in zip(k_range,inerts):
    ax.annotate(f'{v:.0f}',(k,v),textcoords="offset points",xytext=(0,10),ha='center',fontsize=8,color=MUTED)
ax.set_xlabel("K (Number of Clusters)"); ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax.set_title("Elbow Method — pick K where curve bends"); ax.legend(); ax.set_xticks(list(k_range))
plt.tight_layout(); save(fig,"KM_elbow_curve")

# ── KM Fig 2: Silhouette Score
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle("K-Means — Silhouette Score vs K",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
ax.plot(list(k_range),sils_km,'s-',color=GOLD,markersize=10,lw=2.5)
ax.axvline(bkm,color=RED,ls='--',lw=2,label=f'Best K={bkm} (highest sil)')
for k,v in zip(k_range,sils_km):
    ax.annotate(f'{v:.3f}',(k,v),textcoords="offset points",xytext=(0,10),ha='center',fontsize=8,color=MUTED)
ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score (-1 to 1, higher=better)")
ax.set_title("Silhouette Score — confirms best K"); ax.legend(); ax.set_xticks(list(k_range))
plt.tight_layout(); save(fig,"KM_silhouette")

# ── KM Fig 3: PCA 2D Cluster Scatter
fig=plt.figure(figsize=(10,8),facecolor=BG)
fig.suptitle(f"K-Means — Cluster Scatter (PCA 2D)  K={bkm}",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.3)
ax=fig.add_subplot(g[0,0])
for i in range(bkm):
    mask=kml==i
    ax.scatter(X_2d[mask,0],X_2d[mask,1],c=PAL[i],s=18,alpha=0.75,label=f'Cluster {i}',edgecolors='none')
ax.scatter(kmc[:,0],kmc[:,1],c='white',marker='X',s=200,zorder=5,label='Centroids',edgecolors=BG,lw=0.5)
ax.set_title(f"K-Means Clusters (K={bkm})"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=7)
ax=fig.add_subplot(g[0,1])
for i,name in enumerate(CAT):
    mask=y_clf==i
    ax.scatter(X_2d[mask,0],X_2d[mask,1],c=PAL[i],s=18,alpha=0.75,label=name,edgecolors='none')
ax.set_title("True Price Categories"); ax.set_xlabel("PC1"); ax.legend(fontsize=7)
plt.tight_layout(); save(fig,"KM_cluster_scatter")

# ── KM Fig 4: Cluster Sizes
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle(f"K-Means — Cluster Sizes  (K={bkm})",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
u,cnt=np.unique(kml,return_counts=True)
bars=ax.bar([f'Cluster {i}' for i in u],cnt,color=PAL[:len(u)],edgecolor=BG,width=0.6)
for bar,v in zip(bars,cnt):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,str(v),ha='center',fontsize=12,color=TEXT,fontweight='bold')
ax.set_ylabel("Number of Houses"); ax.set_title("How many houses in each cluster?")
plt.tight_layout(); save(fig,"KM_cluster_sizes")

# ── KM Fig 5: Feature Heatmap per Cluster
fig=plt.figure(figsize=(12,6),facecolor=BG)
fig.suptitle(f"K-Means — Feature Means per Cluster  (K={bkm})",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
cdf=pd.DataFrame(X_sc,columns=NUM_FEATS); cdf['Cluster']=kml
means=cdf.groupby('Cluster').mean()
sns.heatmap(means.T,ax=ax,cmap='RdYlGn',center=0,
            annot=True,fmt='.2f',cbar_kws={'shrink':0.8},
            linewidths=0.5,linecolor=BG,annot_kws={'size':9})
ax.set_title("Average Standardised Feature Value per Cluster\n(Green=high, Red=low)")
ax.tick_params(labelsize=9)
plt.tight_layout(); save(fig,"KM_feature_heatmap")

# ── KM Fig 6: Inertia + Silhouette Combined
fig=plt.figure(figsize=(12,5),facecolor=BG)
fig.suptitle("K-Means — Elbow + Silhouette Combined",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.35)
ax=fig.add_subplot(g[0,0])
ax.plot(list(k_range),inerts,'o-',color=TEAL,markersize=8,lw=2)
ax.axvline(bkm,color=RED,ls='--',lw=1.5,label=f'Best K={bkm}')
ax.set_title("Elbow: Inertia vs K"); ax.set_xlabel("K"); ax.set_ylabel("Inertia"); ax.legend()
ax2=ax.twinx(); ax2.plot(list(k_range),sils_km,'s--',color=GOLD,markersize=6,alpha=0.7,label='Silhouette')
ax2.set_ylabel("Silhouette",color=GOLD); ax2.legend(loc='center right')
ax=fig.add_subplot(g[0,1])
bars=ax.bar([f'K={k}' for k in k_range],inerts,color=[RED if k==bkm else TEAL for k in k_range],edgecolor=BG)
ax.set_title("Inertia per K (red=chosen)"); ax.set_ylabel("Inertia"); ax.tick_params(axis='x',rotation=30)
plt.tight_layout(); save(fig,"KM_elbow_silhouette_combined")

print(f"  K-Means: 6 figures saved ✅")

# ═══════════════════════════════════════════════════════════════════════════════════
# ███  K-MEDOIDS — 6 FIGURES
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  K-MEDOIDS CLUSTERING")
print("="*60)

def kmedoids(X,k=3,rs=42):
    rng=np.random.RandomState(rs); idx=list(rng.choice(len(X),k,replace=False))
    for _ in range(100):
        d=np.array([[np.sum((x-X[m])**2) for m in idx] for x in X])
        labels=np.argmin(d,axis=1); new=[]
        for j in range(k):
            cl=np.where(labels==j)[0]
            if len(cl)==0: new.append(idx[j]); continue
            intra=np.array([np.sum([np.sum((X[a]-X[b])**2) for b in cl]) for a in cl])
            new.append(int(cl[np.argmin(intra)]))
        if set(new)==set(idx): break
        idx=new
    return labels,idx

kmed_ks=range(2,7); kmed_sils=[]
for k in kmed_ks:
    lbl,_=kmedoids(X_sc[:400],k); kmed_sils.append(silhouette_score(X_sc[:400],lbl))
bkmed=list(kmed_ks)[np.argmax(kmed_sils)]
kmed_lbl,kmed_idx=kmedoids(X_sc,bkmed)
kmed_c=pca2.transform(X_sc[kmed_idx])
sil_kmed=silhouette_score(X_sc,kmed_lbl)
print(f"  Best K={bkmed}  Silhouette={sil_kmed:.4f}")

# ── KMed Fig 1: Silhouette vs K
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle("K-Medoids — Silhouette Score vs K",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
ax.plot(list(kmed_ks),kmed_sils,'D-',color=BLUE,markersize=10,lw=2.5)
ax.axvline(bkmed,color=RED,ls='--',lw=2,label=f'Best K={bkmed}')
for k,v in zip(kmed_ks,kmed_sils):
    ax.annotate(f'{v:.3f}',(k,v),textcoords="offset points",xytext=(0,10),ha='center',fontsize=9,color=MUTED)
ax.set_xlabel("K"); ax.set_ylabel("Silhouette Score")
ax.set_title("Best K where Silhouette is highest"); ax.legend(); ax.set_xticks(list(kmed_ks))
plt.tight_layout(); save(fig,"KMED_silhouette_vs_K")

# ── KMed Fig 2: Cluster Scatter PCA
fig=plt.figure(figsize=(10,8),facecolor=BG)
fig.suptitle(f"K-Medoids — Cluster Scatter  (K={bkmed})",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.3)
ax=fig.add_subplot(g[0,0])
for i in range(bkmed):
    mask=kmed_lbl==i
    ax.scatter(X_2d[mask,0],X_2d[mask,1],c=PAL[i],s=18,alpha=0.75,label=f'Cluster {i}',edgecolors='none')
ax.scatter(kmed_c[:,0],kmed_c[:,1],c='white',marker='D',s=180,zorder=5,label='Medoids',edgecolors=BG,lw=0.7)
ax.set_title(f"K-Medoids (K={bkmed})  Sil={sil_kmed:.3f}")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=7)
ax=fig.add_subplot(g[0,1])
for i in range(bkm):
    mask=kml==i
    ax.scatter(X_2d[mask,0],X_2d[mask,1],c=PAL[i],s=18,alpha=0.75,label=f'Cluster {i}',edgecolors='none')
ax.scatter(kmc[:,0],kmc[:,1],c='white',marker='X',s=180,zorder=5,label='Centroids',edgecolors=BG)
ax.set_title(f"K-Means (K={bkm}) for comparison")
ax.set_xlabel("PC1"); ax.legend(fontsize=7)
plt.tight_layout(); save(fig,"KMED_cluster_scatter")

# ── KMed Fig 3: Cluster Sizes
fig=plt.figure(figsize=(8,6),facecolor=BG)
fig.suptitle(f"K-Medoids — Cluster Sizes  (K={bkmed})",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
u2,cnt2=np.unique(kmed_lbl,return_counts=True)
bars=ax.bar([f'Cluster {i}' for i in u2],cnt2,color=PAL[:len(u2)],edgecolor=BG,width=0.6)
for bar,v in zip(bars,cnt2):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+3,str(v),ha='center',fontsize=12,color=TEXT,fontweight='bold')
ax.set_ylabel("Number of Houses"); ax.set_title("Houses per Cluster (K-Medoids)")
plt.tight_layout(); save(fig,"KMED_cluster_sizes")

# ── KMed Fig 4: Medoid house details
fig=plt.figure(figsize=(12,5),facecolor=BG)
fig.suptitle(f"K-Medoids — Medoid House Details  (K={bkmed})",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
medoid_df=pd.DataFrame(X_sc[kmed_idx],columns=NUM_FEATS)
medoid_df.index=[f'Medoid {i}' for i in range(bkmed)]
sns.heatmap(medoid_df.T,ax=ax,cmap='RdYlGn',center=0,
            annot=True,fmt='.2f',cbar_kws={'shrink':0.8},
            linewidths=0.5,linecolor=BG,annot_kws={'size':9})
ax.set_title("Feature Values of Each Medoid House (standardised)\nThese are REAL houses from the dataset")
ax.tick_params(labelsize=9)
plt.tight_layout(); save(fig,"KMED_medoid_details")

# ── KMed Fig 5: Feature Heatmap per Cluster
fig=plt.figure(figsize=(12,6),facecolor=BG)
fig.suptitle(f"K-Medoids — Feature Means per Cluster  (K={bkmed})",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
kmed_df=pd.DataFrame(X_sc,columns=NUM_FEATS); kmed_df['Cluster']=kmed_lbl
means2=kmed_df.groupby('Cluster').mean()
sns.heatmap(means2.T,ax=ax,cmap='coolwarm',center=0,
            annot=True,fmt='.2f',cbar_kws={'shrink':0.8},
            linewidths=0.5,linecolor=BG,annot_kws={'size':9})
ax.set_title("Average Feature per Cluster")
ax.tick_params(labelsize=9)
plt.tight_layout(); save(fig,"KMED_feature_heatmap")

# ── KMed Fig 6: KMeans vs KMedoids comparison
fig=plt.figure(figsize=(12,5),facecolor=BG)
fig.suptitle("K-Means vs K-Medoids — Side by Side",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,3,figure=fig,wspace=0.35)
ax=fig.add_subplot(g[0,0])
comp_names=['K-Means','K-Medoids']
comp_sils=[sil_km,sil_kmed]; comp_k=[bkm,bkmed]
bars=ax.bar(comp_names,comp_sils,color=[TEAL,BLUE],edgecolor=BG,width=0.5)
for bar,v,k in zip(bars,comp_sils,comp_k):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,f'{v:.3f}\n(K={k})',ha='center',fontsize=10,color=TEXT)
ax.set_title("Silhouette Score"); ax.set_ylabel("Score")
ax=fig.add_subplot(g[0,1])
ax.scatter(X_2d[:,0],X_2d[:,1],c=[PAL[l%len(PAL)] for l in kml],s=10,alpha=0.5,edgecolors='none')
ax.scatter(kmc[:,0],kmc[:,1],c='white',marker='X',s=200,zorder=5)
ax.set_title(f"K-Means  K={bkm}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax=fig.add_subplot(g[0,2])
ax.scatter(X_2d[:,0],X_2d[:,1],c=[PAL[l%len(PAL)] for l in kmed_lbl],s=10,alpha=0.5,edgecolors='none')
ax.scatter(kmed_c[:,0],kmed_c[:,1],c='white',marker='D',s=200,zorder=5)
ax.set_title(f"K-Medoids  K={bkmed}"); ax.set_xlabel("PC1")
plt.tight_layout(); save(fig,"KMED_vs_kmeans")

print(f"  K-Medoids: 6 figures saved ✅")

# ═══════════════════════════════════════════════════════════════════════════════════
# ███  DBSCAN — 6 FIGURES
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  DBSCAN CLUSTERING")
print("="*60)

eps_vals=[0.3,0.5,0.8,1.0,1.5,2.0]
db_res={}
for eps in eps_vals:
    m=DBSCAN(eps=eps,min_samples=10).fit(X_sc)
    lbl=m.labels_
    nc=len(set(lbl))-(1 if -1 in lbl else 0)
    nn=(lbl==-1).sum()
    sil=silhouette_score(X_sc[lbl!=-1],lbl[lbl!=-1]) if nc>1 and (lbl!=-1).sum()>nc else 0
    db_res[eps]=dict(labels=lbl,clusters=nc,noise=nn,sil=sil)
    print(f"  eps={eps}  clusters={nc}  noise={nn}  sil={sil:.3f}")

# use eps=0.8 as main
best_eps=0.8
db_lbl=db_res[best_eps]['labels']
ndb=db_res[best_eps]['clusters']
nnz=db_res[best_eps]['noise']

# ── DB Fig 1: eps sweep
fig=plt.figure(figsize=(12,5),facecolor=BG)
fig.suptitle("DBSCAN — Parameter Sweep (eps)",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,3,figure=fig,wspace=0.35)
ax=fig.add_subplot(g[0,0])
ax.plot(eps_vals,[db_res[e]['clusters'] for e in eps_vals],'o-',color=TEAL,markersize=9,lw=2)
ax.axvline(best_eps,color=RED,ls='--',lw=1.5,label=f'eps={best_eps}')
ax.set_title("Clusters Found vs eps"); ax.set_xlabel("eps"); ax.set_ylabel("# Clusters"); ax.legend()
ax=fig.add_subplot(g[0,1])
ax.plot(eps_vals,[db_res[e]['noise'] for e in eps_vals],'s-',color=GOLD,markersize=9,lw=2)
ax.axvline(best_eps,color=RED,ls='--',lw=1.5)
ax.set_title("Noise Points vs eps"); ax.set_xlabel("eps"); ax.set_ylabel("# Noise Points")
ax=fig.add_subplot(g[0,2])
ax.plot(eps_vals,[db_res[e]['sil'] for e in eps_vals],'D-',color=PUR,markersize=9,lw=2)
ax.axvline(best_eps,color=RED,ls='--',lw=1.5)
ax.set_title("Silhouette Score vs eps"); ax.set_xlabel("eps"); ax.set_ylabel("Silhouette")
plt.tight_layout(); save(fig,"DBSCAN_eps_sweep")

# ── DB Fig 2: Cluster Scatter PCA
fig=plt.figure(figsize=(10,8),facecolor=BG)
fig.suptitle(f"DBSCAN — Cluster Scatter (eps={best_eps}, min=10)",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.3)
ax=fig.add_subplot(g[0,0])
for lbl in sorted(set(db_lbl)):
    mask=db_lbl==lbl
    col='#555555' if lbl==-1 else PAL[lbl%len(PAL)]
    nm='Noise ⚠️' if lbl==-1 else f'Cluster {lbl}'
    ax.scatter(X_2d[mask,0],X_2d[mask,1],c=col,s=15 if lbl==-1 else 18,
               alpha=0.5 if lbl==-1 else 0.8,label=nm,edgecolors='none')
ax.set_title(f"DBSCAN  eps={best_eps}\n{ndb} clusters  {nnz} noise points")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=7,ncol=2)
ax=fig.add_subplot(g[0,1])
noise_mask=db_lbl==-1
ax.scatter(X_2d[~noise_mask,0],X_2d[~noise_mask,1],c=TEAL,s=15,alpha=0.6,label=f'Cluster ({(~noise_mask).sum()})',edgecolors='none')
ax.scatter(X_2d[noise_mask,0],X_2d[noise_mask,1],c=RED,s=20,alpha=0.9,label=f'Noise ({noise_mask.sum()})',edgecolors='none')
ax.set_title("Noise Points Highlighted in Red")
ax.set_xlabel("PC1"); ax.legend(fontsize=8)
plt.tight_layout(); save(fig,"DBSCAN_cluster_scatter")

# ── DB Fig 3: Noise house analysis
fig=plt.figure(figsize=(12,6),facecolor=BG)
fig.suptitle("DBSCAN — Noise (Outlier) House Feature Analysis",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.35)
ax=fig.add_subplot(g[0,0])
noise_feat=pd.DataFrame(X_sc[db_lbl==-1],columns=NUM_FEATS).mean()
normal_feat=pd.DataFrame(X_sc[db_lbl!=-1],columns=NUM_FEATS).mean()
comp_df=pd.DataFrame({'Noise':noise_feat,'Normal':normal_feat})
x4=np.arange(len(NUM_FEATS))
ax.bar(x4-0.2,comp_df['Noise'],0.38,label='Noise houses',color=RED,edgecolor=BG,alpha=0.85)
ax.bar(x4+0.2,comp_df['Normal'],0.38,label='Normal houses',color=TEAL,edgecolor=BG,alpha=0.85)
ax.set_xticks(x4); ax.set_xticklabels(NUM_FEATS,rotation=45,ha='right',fontsize=7)
ax.set_title("Avg Feature: Noise vs Normal Houses"); ax.set_ylabel("Scaled Value"); ax.legend()
ax=fig.add_subplot(g[0,1])
noise_prices=y_reg[db_lbl==-1]/1000
normal_prices=y_reg[db_lbl!=-1]/1000
ax.hist(normal_prices,bins=25,alpha=0.7,color=TEAL,label=f'Normal ({len(normal_prices)})',edgecolor=BG)
ax.hist(noise_prices,bins=15,alpha=0.85,color=RED,label=f'Noise ({len(noise_prices)})',edgecolor=BG)
ax.set_xlabel("Price ($K)"); ax.set_ylabel("Count")
ax.set_title("Price Distribution: Normal vs Noise Houses"); ax.legend()
plt.tight_layout(); save(fig,"DBSCAN_noise_analysis")

# ── DB Fig 4: Different eps visual comparison
fig=plt.figure(figsize=(18,5),facecolor=BG)
fig.suptitle("DBSCAN — Cluster Results at Different eps Values",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,4,figure=fig,wspace=0.3)
for pi,(eps) in enumerate([0.3,0.5,0.8,1.5]):
    ax=fig.add_subplot(g[0,pi])
    lbl=db_res[eps]['labels']
    for l in sorted(set(lbl)):
        mask=lbl==l; col='#444' if l==-1 else PAL[l%len(PAL)]
        ax.scatter(X_2d[mask,0],X_2d[mask,1],c=col,s=8,alpha=0.6,edgecolors='none')
    nc=db_res[eps]['clusters']; nn=db_res[eps]['noise']
    ax.set_title(f"eps={eps}\n{nc} clusters, {nn} noise"); ax.set_xlabel("PC1")
    if pi==0: ax.set_ylabel("PC2")
plt.tight_layout(); save(fig,"DBSCAN_eps_comparison")

# ── DB Fig 5: Cluster stats
fig=plt.figure(figsize=(10,6),facecolor=BG)
fig.suptitle(f"DBSCAN — Cluster Summary (eps={best_eps})",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.35)
ax=fig.add_subplot(g[0,0])
unique_lbl=sorted(set(db_lbl)); counts_db=[(db_lbl==l).sum() for l in unique_lbl]
names_db=['Noise' if l==-1 else f'Cluster {l}' for l in unique_lbl]
colors_db=['#555555' if l==-1 else PAL[l%len(PAL)] for l in unique_lbl]
bars=ax.bar(names_db,counts_db,color=colors_db,edgecolor=BG)
for bar,v in zip(bars,counts_db):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,str(v),ha='center',fontsize=10,color=TEXT)
ax.set_title("Cluster Sizes + Noise Count"); ax.set_ylabel("Count")
ax=fig.add_subplot(g[0,1])
prices_by_cluster=[]
labels_by_cluster=[]
for l in sorted(set(db_lbl)):
    prices_by_cluster.append(y_reg[db_lbl==l]/1000)
    labels_by_cluster.append('Noise' if l==-1 else f'Cluster {l}')
bp2=ax.boxplot(prices_by_cluster,patch_artist=True,
               medianprops=dict(color='white',lw=2))
for patch,col in zip(bp2['boxes'],colors_db): patch.set_facecolor(col)
ax.set_xticklabels(labels_by_cluster,fontsize=9)
ax.set_title("Price Distribution per Cluster"); ax.set_ylabel("Price ($K)")
plt.tight_layout(); save(fig,"DBSCAN_cluster_summary")

# ── DB Fig 6: min_samples sweep
fig=plt.figure(figsize=(12,5),facecolor=BG)
fig.suptitle("DBSCAN — min_samples Parameter Sweep (eps=0.8)",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.35)
min_s_vals=[3,5,8,10,15,20,30]
ms_clusters=[]; ms_noise=[]
for ms in min_s_vals:
    m=DBSCAN(eps=0.8,min_samples=ms).fit(X_sc); l=m.labels_
    ms_clusters.append(len(set(l))-(1 if -1 in l else 0))
    ms_noise.append((l==-1).sum())
ax=fig.add_subplot(g[0,0])
ax.plot(min_s_vals,ms_clusters,'o-',color=TEAL,markersize=8,lw=2)
ax.axvline(10,color=RED,ls='--',lw=1.5,label='min=10 used')
ax.set_title("Clusters vs min_samples"); ax.set_xlabel("min_samples"); ax.set_ylabel("# Clusters"); ax.legend()
ax=fig.add_subplot(g[0,1])
ax.plot(min_s_vals,ms_noise,'s-',color=GOLD,markersize=8,lw=2)
ax.axvline(10,color=RED,ls='--',lw=1.5)
ax.set_title("Noise Points vs min_samples"); ax.set_xlabel("min_samples"); ax.set_ylabel("# Noise Points")
plt.tight_layout(); save(fig,"DBSCAN_minsamples_sweep")

print(f"  DBSCAN: 6 figures saved ✅")

# ═══════════════════════════════════════════════════════════════════════════════════
# ███  CONFUSION MATRIX — dedicated full figure for KNN
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  CONFUSION MATRIX — Detailed Analysis")
print("="*60)

# ── CM Fig 1: KNN Confusion Matrix large
fig=plt.figure(figsize=(10,8),facecolor=BG)
fig.suptitle(f"Confusion Matrix — KNN (K={bk})  Detailed",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
cm_knn=confusion_matrix(yc_te,knn_p)
sns.heatmap(cm_knn,annot=True,fmt='d',cmap='Blues',
            xticklabels=CAT,yticklabels=CAT,ax=ax,
            linewidths=2,linecolor=BG,annot_kws={'size':16,'weight':'bold'})
ax.set_xlabel("Predicted Label",fontsize=12); ax.set_ylabel("True Label",fontsize=12)
total=cm_knn.sum(); correct=np.trace(cm_knn)
ax.set_title(f"Total={total}  Correct={correct}  Wrong={total-correct}  Acc={correct/total:.4f}")
plt.tight_layout(); save(fig,"CM_KNN_detailed")

# ── CM Fig 2: Normalised Confusion Matrix
fig=plt.figure(figsize=(10,8),facecolor=BG)
fig.suptitle(f"Confusion Matrix — KNN Normalised (% per row)",fontsize=14,fontweight='bold',color=TEXT)
ax=fig.add_subplot(111)
cm_norm=cm_knn.astype(float)/cm_knn.sum(axis=1,keepdims=True)*100
sns.heatmap(cm_norm,annot=True,fmt='.1f',cmap='Blues',
            xticklabels=CAT,yticklabels=CAT,ax=ax,
            linewidths=2,linecolor=BG,annot_kws={'size':14,'weight':'bold'})
ax.set_xlabel("Predicted Label",fontsize=12); ax.set_ylabel("True Label",fontsize=12)
ax.set_title("Values show % of that row  (diagonal=correctly classified %)")
plt.tight_layout(); save(fig,"CM_KNN_normalised")

# ── CM Fig 3: Per-class breakdown
fig=plt.figure(figsize=(14,8),facecolor=BG)
fig.suptitle("Confusion Matrix — Per-Class Breakdown (KNN)",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(2,2,figure=fig,hspace=0.4,wspace=0.35)
for i,cat in enumerate(CAT):
    ax=fig.add_subplot(g[i//2,i%2])
    tp=cm_knn[i,i]; fn=cm_knn[i,:].sum()-tp
    fp=cm_knn[:,i].sum()-tp; tn=cm_knn.sum()-tp-fp-fn
    binary_cm=np.array([[tp,fn],[fp,tn]])
    sns.heatmap(binary_cm,annot=True,fmt='d',cmap='Blues',ax=ax,
                linewidths=1,linecolor=BG,annot_kws={'size':14,'weight':'bold'},
                xticklabels=[f'Pred {cat}',f'Not {cat}'],
                yticklabels=[f'True {cat}',f'Not {cat}'])
    prec=tp/(tp+fp) if (tp+fp)>0 else 0
    rec=tp/(tp+fn) if (tp+fn)>0 else 0
    ax.set_title(f"{cat}  Prec={prec:.2f}  Rec={rec:.2f}",fontsize=10)
plt.tight_layout(); save(fig,"CM_perclass_breakdown")

# ── CM Fig 4: Error analysis
fig=plt.figure(figsize=(12,6),facecolor=BG)
fig.suptitle("Confusion Matrix — Error Analysis",fontsize=14,fontweight='bold',color=TEXT)
g=gs.GridSpec(1,2,figure=fig,wspace=0.35)
ax=fig.add_subplot(g[0,0])
errors=cm_knn.copy(); np.fill_diagonal(errors,0)
sns.heatmap(errors,annot=True,fmt='d',cmap='Reds',
            xticklabels=CAT,yticklabels=CAT,ax=ax,
            linewidths=1,linecolor=BG,annot_kws={'size':13,'weight':'bold'})
ax.set_title("Error Matrix (diagonal=0, only mistakes shown)")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax=fig.add_subplot(g[0,1])
rep=classification_report(yc_te,knn_p,target_names=CAT,output_dict=True)
cats2=list(CAT); precs=[rep[c]['precision'] for c in cats2]
recs=[rep[c]['recall'] for c in cats2]; f1s2=[rep[c]['f1-score'] for c in cats2]
x5=np.arange(len(cats2))
ax.bar(x5-0.25,precs,0.25,label='Precision',color=TEAL,edgecolor=BG)
ax.bar(x5,recs,0.25,label='Recall',color=RED,edgecolor=BG)
ax.bar(x5+0.25,f1s2,0.25,label='F1',color=GOLD,edgecolor=BG)
ax.set_xticks(x5); ax.set_xticklabels(cats2,fontsize=8)
ax.set_title("Precision / Recall / F1 per Class"); ax.legend(); ax.set_ylim(0,1.2)
plt.tight_layout(); save(fig,"CM_error_analysis")

print(f"  Confusion Matrix: 4 figures saved ✅")

# ═══════════════════════════════════════════════════════════════════════════════════
# ███  FINAL SUMMARY FIGURE
# ═══════════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)

fig=plt.figure(figsize=(16,10),facecolor=BG)
fig.suptitle("DMDW Project — Complete Algorithm Summary",fontsize=15,fontweight='bold',color=TEXT)
g=gs.GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.38)

ax=fig.add_subplot(g[0,0])
alg_names=['KNN','Linear\nReg','K-Means','K-Medoids','DBSCAN']
alg_sils=[None,None,sil_km,sil_kmed,db_res[best_eps]['sil']]
knn_acc=km_res[bk]['acc']; lr_r2=r2_lr
ax.bar(['KNN\nAcc','LR\nR²'],[knn_acc,lr_r2],color=[TEAL,GOLD],edgecolor=BG,width=0.5)
for x,v,lbl in zip([0,1],[knn_acc,lr_r2],['Accuracy','R²']):
    ax.text(x,v+0.01,f'{v:.3f}',ha='center',fontsize=11,color=TEXT,fontweight='bold')
ax.set_title("Supervised: KNN & Regression"); ax.set_ylim(0,1.2)

ax=fig.add_subplot(g[0,1])
cl_names=['K-Means','K-Medoids','DBSCAN*']
db_sil=db_res[best_eps]['sil']
cl_sils=[sil_km,sil_kmed,db_sil if db_sil>0 else 0.05]
bars=ax.bar(cl_names,cl_sils,color=[TEAL,BLUE,PUR],edgecolor=BG,width=0.5)
for bar,v,k in zip(bars,cl_sils,[bkm,bkmed,'eps=0.8']):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,
            f'{v:.3f}\nK={k}',ha='center',fontsize=9,color=TEXT)
ax.set_title("Clustering: Silhouette Scores\n(*DBSCAN non-noise only)"); ax.set_ylabel("Silhouette")

ax=fig.add_subplot(g[0,2])
knn_vals=[km_res[bk]['acc'],km_res[bk]['prec'],km_res[bk]['rec'],km_res[bk]['f1']]
bars=ax.bar(['Acc','Prec','Rec','F1'],knn_vals,color=PAL[:4],edgecolor=BG)
for bar,v in zip(bars,knn_vals):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{v:.3f}',ha='center',fontsize=9)
ax.set_title(f"KNN Best Metrics (K={bk})"); ax.set_ylim(0,1.2)

ax=fig.add_subplot(g[1,0])
ax.scatter(yr_te/1000,lr_pred/1000,c=TEAL,s=10,alpha=0.6,edgecolors='none')
mn3,mx3=yr_te.min()/1000,yr_te.max()/1000
ax.plot([mn3,mx3],[mn3,mx3],'--',color=RED,lw=2)
ax.set_title(f"Regression: Actual vs Predicted\nR²={r2_lr:.4f}  RMSE=${rmse_lr/1000:.1f}K")
ax.set_xlabel("Actual ($K)"); ax.set_ylabel("Predicted ($K)")

ax=fig.add_subplot(g[1,1])
for i in range(bkm):
    mask=kml==i
    ax.scatter(X_2d[mask,0],X_2d[mask,1],c=PAL[i],s=10,alpha=0.7,label=f'C{i}',edgecolors='none')
ax.scatter(kmc[:,0],kmc[:,1],c='white',marker='X',s=150,zorder=5)
ax.set_title(f"K-Means Clusters (K={bkm})"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=7)

ax=fig.add_subplot(g[1,2])
draw_cm(ax,yc_te,knn_p,f"KNN Confusion Matrix  K={bk}",'Blues')

plt.tight_layout(); save(fig,"SUMMARY_all_algorithms")

# ═══════════════════════════════════════════════════════════
# PRINT FINAL REPORT
# ═══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FINAL RESULTS REPORT")
print("="*60)
print(f"""
┌──────────────────────────────────────────────────────┐
│           DMDW PROJECT — RESULTS SUMMARY             │
├─────────────────────────┬────────────────────────────┤
│ KNN  (Best K={bk:<2d})          │ Acc={km_res[bk]['acc']:.4f}  F1={km_res[bk]['f1']:.4f}     │
│ Linear Regression (OLS) │ R²={r2_lr:.4f}  RMSE=${rmse_lr/1000:.1f}K       │
│ K-Means (K={bkm})           │ Silhouette={sil_km:.4f}              │
│ K-Medoids (K={bkmed})        │ Silhouette={sil_kmed:.4f}              │
│ DBSCAN (eps={best_eps})      │ Clusters={ndb}  Noise={nnz}           │
└─────────────────────────┴────────────────────────────┘
""")

total_figs=FIG_N[0]
print(f"  📊 Total figures generated : {total_figs}")
print(f"  📁 Saved to                : {OUT_DIR}")
print(f"\n  ✅ ALL DONE — dmdwAD project complete!")
