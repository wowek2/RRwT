import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, VBox, Layout, HTML

# Wczytanie danych
df = pd.read_csv("dane_rzeki.csv")
cs_cols = [c for c in df.columns if c.startswith("cs_")]
offsets = np.array([int(c.replace("cs_", "")) for c in cs_cols])
cs_data = df[cs_cols].values

# Figura
fig, (ax_long, ax_cross) = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_alpha(0)
plt.subplots_adjust(wspace=0.3, top=0.85)

fig.suptitle('Wykres profilu oraz przekroju poprzecznego odcinka rzecznego\npomiędzy jez. Modrym a Wrzeszczyńskim', 
             fontsize=12, fontweight='bold', color="#000000", y=0.95)

for ax in [ax_long, ax_cross]:
    ax.set_facecolor('none')
    ax.grid(True, ls=':', alpha=0.5)

# Profil podłużny
ax_long.plot(df['dist_m'], df['z_long_raw'], c='#b0c4de', lw=1, alpha=0.6, label='Dane surowe')
ax_long.plot(df['dist_m'], df['z_long_smooth'], c='#005b96', lw=2.5, label='Wygładzony profil')
ax_long.fill_between(df['dist_m'], df['z_long_smooth'], 290, alpha=0.1)
marker_long, = ax_long.plot([], [], 'o', c='#d9534f', mec='white', ms=10)
ax_long.set(xlabel='Dystans (m)', ylabel='Wysokość (m)', ylim=(290, df['z_long_smooth'].max()+5))
ax_long.legend(loc='upper right')

info_box = ax_long.text(0.02, 0.95, "", transform=ax_long.transAxes, va='top',
                        bbox=dict(boxstyle="round", fc="white", alpha=0.8))

# Przekrój
line_cross, = ax_cross.plot([], [], c='#2e8b57', lw=3, label='Przekrój')
marker_cross, = ax_cross.plot([], [], 'o', c='#d9534f', mec='white', ms=10, label='Oś rzeki (przybliżona)')
ax_cross.fill_between([], [], color='none', edgecolor='red', hatch='///', label='Brak danych')
ax_cross.set(xlabel='Odległość od osi (m)', ylabel='Wysokość (m)',
             xlim=(min(offsets), max(offsets)), ylim=(280, np.nanmax(cs_data)+2))
ax_cross.yaxis.tick_right()
ax_cross.legend(loc='upper right')

fill_poly = None
nan_patches = []

def update(change):
    global fill_poly, nan_patches
    krok = change['new'] if isinstance(change, dict) else change
    d, z = df['dist_m'][krok], df['z_long_smooth'][krok]
    cy = cs_data[krok]
    
    marker_long.set_data([d], [z])
    info_box.set_text(f"Dystans: {d:.1f} m\nWysokość: {z:.2f} m")
    
    line_cross.set_data(offsets, cy)
    marker_cross.set_data([0], [z])
    
    if fill_poly: fill_poly.remove()
    for p in nan_patches: p.remove()
    nan_patches = []
    
    fill_poly = ax_cross.fill_between(offsets, cy, 280, color='#8fbc8f', alpha=0.5)
    
    # Zaznacz NaN
    nan_mask = np.isnan(cy)
    if nan_mask.any():
        ymin, ymax = ax_cross.get_ylim()
        diff = np.diff(np.concatenate([[0], nan_mask.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            x0, x1 = offsets[s], offsets[min(e, len(offsets)-1)]
            if s > 0: x0 = (offsets[s-1] + offsets[s]) / 2
            if e < len(offsets): x1 = (offsets[e-1] + offsets[min(e, len(offsets)-1)]) / 2
            patch = ax_cross.fill_between([x0, x1], ymin, ymax, 
                                          color='none', edgecolor='red', hatch='///', alpha=0.7)
            nan_patches.append(patch)
    
    fig.canvas.draw_idle()

slider = IntSlider(min=0, max=len(df)-1, value=0, description="Krok:", 
                   layout=Layout(width='80%'))
slider.observe(update, names='value')

display(VBox([fig.canvas, slider]))
update(0)
