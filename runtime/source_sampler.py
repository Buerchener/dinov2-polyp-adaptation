"""Matched eight-frame replay plus one C1 normal; auxiliary source is the arm."""
import numpy as np

def draws(rows, seed, steps, arm):
    assert arm in ('old_C1', 'new_C1')
    centers=('C1','C2','C3','C4')
    groups={c:[i for i,r in enumerate(rows) if r['center']==c and not r.get('normal_expansion_candidate')] for c in centers}
    old=[i for i,r in enumerate(rows) if r['center']=='C1' and not r['reference_positive'] and not r.get('normal_expansion_candidate')]
    videos=sorted({r['group'] for r in rows if r.get('normal_expansion_candidate')})
    new={g:[i for i,r in enumerate(rows) if r.get('normal_expansion_candidate') and r['group']==g] for g in videos}
    assert all(len(g)>=2 for g in groups.values()) and old and len(videos)==3
    base_rng=np.random.default_rng(seed)
    aux_rng=np.random.default_rng(seed+456000)
    for _ in range(steps):
        ids=[int(i) for c in centers for i in base_rng.choice(groups[c],2,replace=False)]
        augmentation=[int(i) for i in base_rng.integers(0,2**31-1,8)]
        # Both arms consume the same auxiliary RNG stream. Pick videos equally,
        # then frames uniformly, so the longest video does not dominate.
        video_index=int(aux_rng.integers(0,len(videos)))
        uniform=float(aux_rng.random())
        pool=[i for i in old if i not in ids] if arm=='old_C1' else new[videos[video_index]]
        extra=pool[min(int(uniform*len(pool)),len(pool)-1)]
        ids.append(extra)
        augmentation.append(int(aux_rng.integers(0,2**31-1)))
        yield ids,augmentation
