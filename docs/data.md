# Data and weights

No image, mask, prediction archive or weight tensor is included. Obtain data under its source terms; the manifests describe the specific releases used, rather than granting redistribution permission.

## PolypGen

Use the [official project](https://github.com/DebeshJha/PolypGen) and [dataset paper](https://www.nature.com/articles/s41597-023-01981-y), which identifies the [Synapse collection](https://doi.org/10.7303/syn26376615). The recorded archive is `PolypGen2021_MultiCenterData_v3.zip`. Do not substitute a similarly named release without checking hashes.

Either place that archive in `datasets/polypgen/` (members begin `PolypGen2021_MultiCenterData_v3/`) or extract its contents so `datasets/polypgen/data_C1/images_C1/...` etc. match the manifest's relative paths. The original loader verifies image/mask hashes as it reads them. `runtime/crop-boxes.json` provides fixed RGB-only crops. Some canonical pairings/encodings were audited during preparation; if an official download fails a manifest hash, stop and resolve the release or canonicalization discrepancy rather than silently overriding it. This repository does not redistribute the canonicalized pixels or claim clean-download end-to-end data reconstruction was rerun during publication.

## External releases

The evaluated archive came from [Zenodo record 5579392](https://zenodo.org/records/5579392). Recreate `datasets/external/data/<dataset>/images/` and `masks/`, where dataset is `CVC-ColonDB`, `ETIS-LaribPolypDB`, or `CVC-300`. `configs/external/*.jsonl` records member-level SHA256 values. The historical run checked ZIP member CRC and file SHA256; it did not download/verify the full archive checksum. Follow the source and original dataset terms. This particular release has complete CVC-300-to-ColonDB overlap; do not generalize that finding to every archive named CVC-300.

## Encoder weights and final checkpoint

Obtain DINOv2 ViT-S/14 from the [official DINOv2 repository](https://github.com/facebookresearch/dinov2). Historical encoder file SHA256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. Pass the path through `--weights`; there is no automatic download in the public scripts.

The selected trained model is identified by SHA256 `ceb895039f9fccd58c4fb6d64c4628f993dcca62bc7f9e2f4e8629ec7bd5e75a`. It stays local. Public users can retrain using the manifest, but cannot reproduce the exact published checkpoint output from this repository alone. If distribution is later approved, publish it separately with model/data license review and this checksum.
