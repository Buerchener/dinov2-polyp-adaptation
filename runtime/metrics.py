import numpy as np

def summarize(records):
    positives=[r for r in records if r['reference_positive']]
    negatives=[r for r in records if not r['reference_positive']]
    seqgroups=sorted({r['group'] for r in negatives if r['kind']=='negative_sequence'})
    negative_sequences={g:[r for r in negatives if r['group']==g] for g in seqgroups}
    def mean(rs,key):return float(np.mean([r[key] for r in rs])) if rs else None
    strata={label:[r for r in positives if lo<=r['gt_pixels']/r['image_pixels']<hi] for label,lo,hi in [('small',0,.05),('medium',.05,.2),('large',.2,1.01)]}
    return dict(positive_count=len(positives),positive_image_mean_dice=mean(positives,'positive_dice'),
                severe_count=sum(r['severe'] for r in positives),negative_count=len(negatives),
                positive_area_strata={k:dict(n=len(rs),mean_dice=mean(rs,'positive_dice'),severe=sum(r['severe'] for r in rs)) for k,rs in strata.items()},
                negative_image_any_fp_rate=mean(negatives,'negative_any_false_positive'),
                negative_image_mean_fp_fraction=mean(negatives,'negative_fp_fraction'),
                negative_sequence_groups={g:dict(n=len(rs),any_fp_rate=mean(rs,'negative_any_false_positive'),
                    fp_over_0p1pct_rate=mean(rs,'negative_fp_over_0p1pct'),mean_fp_fraction=mean(rs,'negative_fp_fraction')) for g,rs in negative_sequences.items()},
                negative_sequence_macro_over_0p1pct_rate=float(np.mean([mean(rs,'negative_fp_over_0p1pct') for rs in negative_sequences.values()])) if negative_sequences else None)
