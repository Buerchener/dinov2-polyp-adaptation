"""Original metrics plus explicitly requested counts; no changed definitions."""
from metrics import summarize

def summary(records, epoch):
    result = summarize(records)
    positives = [r for r in records if r['reference_positive']]
    normals = [r for r in records if not r['reference_positive']]
    result.update(epoch=epoch,
        empty_positive_predictions=sum(r['prediction_pixels'] == 0 for r in positives),
        normal_any_pixel_fp_count=sum(r['negative_any_false_positive'] for r in normals),
        normal_area_over_0p1pct_fp_count=sum(r['negative_fp_over_0p1pct'] for r in normals))
    return result
