# SANA-map

<p align="center">
  <strong>Real-time open-vocabulary semantic mapping for agricultural robots</strong>
</p>

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

SANA-map builds a persistent, multichannel bird's-eye-view (BEV) semantic map
from RGB-D observations, robot pose, and user-specified natural-language
prompts. Open-vocabulary instance masks are back-projected into a semantic
point cloud, voxelised, and fused into a global map whose semantic channels
change at deployment time — no detector retraining required when the crop,
cultivar, or task changes.

This repository is the mapping-only reference implementation accompanying
the paper *"From Words to Rows: Prompt-Conditioned Real-Time Semantic
Mapping for Agricultural Robots"* (SATNAC 2026). It reproduces the pipeline
in Sec. III of the paper — RGB-D capture, prompt-conditioned detection,
back-projection, and geocentric fusion (Fig. 1) — on a Stereolabs ZED 2i
camera. It does not include the navigation-policy or evaluation code used
to produce the paper's figures; those live in the group's internal
research repository.

## Quick start

Requires a CUDA GPU and the [Stereolabs ZED SDK](https://www.stereolabs.com/developers/release)
(the ZED SDK ships its own `pyzed` Python API, which is not on PyPI — install
it from the SDK's `get_python_api.py` after installing the SDK itself).

```bash
git clone https://github.com/<org>/sana-map.git
cd sana-map
pip install -e .
```

Run against a recorded `.svo` file (no camera required):

```bash
sana-map --config configs/indoor.yaml \
          --svo path/to/recording.svo \
          --dump_dir results/indoor_run_01
```

Run against a live ZED 2i camera:

```bash
sana-map --config configs/berryfarm.yaml --dump_dir results/field_run_01
```

`--config` loads a YAML file of defaults; any flag still passed on the
command line overrides it. Run `sana-map --help` for the full flag list, or
see [`sana_map/config.py`](sana_map/config.py) — every flag there is
documented at the point where it's added to the parser.

### Prompt-conditioned categories

The mapped semantic categories are set by `--classes`, a comma-separated
list of natural-language prompts (e.g. `"potted plant,flower pot"`). Each
category becomes one channel in the output map. The paper's Sec. IV-A
finding — that a visually related *proxy* prompt (`"potted plant"`) can
recover a usable signal where a direct crop prompt (`"blueberry bush"`)
fails — applies directly here: try a proxy prompt describing a visually
distinctive part of the scene if the direct category name produces no
detections.

### Output

Each run writes to `--dump_dir`:
- `SLAM_MAP.pt` — the final multichannel BEV map (obstacle, explored,
  current/past robot location, and one channel per prompt category).
- `pose_history.npy` — the accumulated trajectory.
- `profiling.csv` — a per-stage timing breakdown (Sec. IV-B in the paper).
- Periodic visualisation frames, saved every `--saving_frequency` frames.

## Repository layout

```
sana_map/
  run.py                       Threaded ZED capture / detect / map pipeline
  config.py                    CLI + YAML configuration (sana-map --help)
  mapping.py                   Semantic_Mapping: the BEV projection and fusion module
  profiler.py                  Per-stage timing (feeds profiling.csv)
  visualisation.py             Map + trajectory rendering
  perception/
    open_vocab_detector.py     Selects one of the four detectors below
    yoloe.py owl.py yoloworld.py dino.py   Detector-specific wrappers
  utils/
    depth_utils.py             Pinhole back-projection, voxel splatting
    rotation_utils.py          Rotation-matrix helpers used by depth_utils
    nn_utils.py                get_grid (geocentric warp) and ChannelPool
configs/
  indoor.yaml                  Sec. IV settings for the indoor test bed
  berryfarm.yaml                Sec. IV settings for the outdoor field deployment
```

## Supported detectors

Selected via `--yolo_or_owl` in any config:

| Value | Detector | Notes |
|---|---|---|
| 0 | YOLO-World | Real-time; benchmarked in Sec. IV-A |
| 1 | OWLv2 | Not real-time on the reported hardware (3.0 fps) |
| 2 | YOLOE | Default; fastest in the reported benchmark (30.3 fps) |
| 3 | Grounding DINO | Not real-time on the reported hardware (0.84 fps) |

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{webb2026sanamap,
  title     = {From Words to Rows: Prompt-Conditioned Real-Time Semantic Mapping for Agricultural Robots},
  author    = {Webb, Travimadox and Amayo, Paul and Nemaangani, Talifhani},
  booktitle = {Southern Africa Telecommunication Networks and Applications Conference (SATNAC)},
  year      = {2026}
}
```

## Acknowledgements

This work is based on research supported in part by the National Research
Foundation of South Africa (NRF) and received funding from the Google.Org
AI Collaborative on Food Security. The mapping formulation builds on
[Active Neural SLAM](https://arxiv.org/abs/2004.05155) and
[SemExp](https://arxiv.org/abs/2007.00643).

## License

[MIT](LICENSE)
