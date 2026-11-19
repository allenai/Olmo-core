# dense LoRA actual PF per row, calibrated on memscale (64k 1.836, 128k 4.80) and checked on the Beaker CP4 smoke (252k: 13.3)
def dl(L): x=L/65536; return 1.272*x+0.564*x*x
def dl_split(L): x=L/65536; return 1.272*x, 0.564*x*x
BUDGET=32*3600*0.132   # PF: 32 GPU-h at the measured dense-LoRA 256k rate (101 GPU-s per 252k row = 0.132 PF/GPU-s)
short={"2k":2586,"4k":4149,"8k":8371,"16k":14506}
import sys
def table(longL, longname, gdd_ratio_long, gdd_ratio_short, nshort_per_bin):
    sp=sum(nshort_per_bin*dl(L) for L in short.values())
    nl=int((BUDGET-sp)/dl(longL))
    rows={k:nshort_per_bin for k in short}; rows[longname]=nl
    lens=dict(short); lens[longname]=longL
    R=sum(rows.values()); T=sum(rows[k]*lens[k] for k in rows); F=sum(rows[k]*dl(lens[k]) for k in rows)
    print(f"\n{longname} run, {nshort_per_bin} short rows/bin: budget {BUDGET:.0f} PF")
    print("bin | rows | row share | tokens (M) | token share | dense PF | PF share | gold-drop rows (matched PF)")
    gF=sum(rows[k]*dl(lens[k])*(gdd_ratio_long if k==longname else gdd_ratio_short) for k in rows); kmul=BUDGET/gF
    for k in rows:
        f=rows[k]*dl(lens[k])
        print(f"{k} | {rows[k]} | {rows[k]/R:.1%} | {rows[k]*lens[k]/1e6:.1f} | {rows[k]*lens[k]/T:.1%} | {f:.0f} | {f/F:.1%} | {round(rows[k]*kmul)}")
    print(f"total | {R} | | {T/1e6:.0f} | | {F:.0f} | | {round(R*kmul)}   steps@16: dense {R/16:.0f}, gold-drop {R*kmul/16:.0f}; gdd row multiplier {kmul:.1f}")
for L in (16384*0.885, 58586, 252368, 1039183):
    a,b=dl_split(L); print(f"L={L:.0f}: dense LoRA {a+b:.2f} PF/row (linear {a:.2f}, attn-quadratic {b:.2f} = {b/(a+b):.0%})")
# gold-drop ratios from cost.py (whole 0.15 / keep 0.05-0.5): 16k 0.121, 64k 0.102, 256k 0.070, 1m (whole 0) 0.0245
table(252368,"256k",0.070,0.121,1200)
table(252368,"256k",0.070,0.121,272)
table(1039183,"1m",0.0245,0.121,1200)
