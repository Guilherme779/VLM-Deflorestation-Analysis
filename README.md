# VLM Deforestation Analysis

**Master's Thesis** · Instituto Superior Técnico, Lisbon · 2025–Present  
**Author:** João Guilherme Teixeira  
**Status:** 🟡 In Progress — Dataset pipeline and annotation complete, evaluation upcoming

---

## Overview

This thesis investigates how Vision-Language Models (VLMs) generalise to remote sensing imagery for deforestation analysis in the Amazon biome. It covers zero/one-shot evaluation of state-of-the-art VLMs across image captioning, VQA, object detection, and change detection — followed by supervised fine-tuning of the best-performing model on a purpose-built multimodal dataset. Ultimately, a new from scratch VLM will be developed specifically for this domain.

---

## Research Phases

| Phase | Status |
|---|---|
| Literature review | ✅ Done |
| Dataset pipeline | ✅ Implemented |
| Dataset annotation (captions, QA, grounding) | ✅ Implemented |
| Dataset export & curation | 🟡 In progress |
| Baseline VLM evaluation | ⬜ Planned |
| Model selection & fine-tuning | ⬜ Planned |
| Authoral from scratch VLM creation | ⬜ Planned |
| Final benchmarking | ⬜ Planned |


---

## What's Built

A fully automated pipeline from raw satellite imagery to annotated multimodal patches, orchestrated by `RegionPipeline` and runnable via `run_region_pipeline.py`.

**Data collection**
1. **Export** — pulls Sentinel-2 RGB+NIR imagery and PRODES/MapBiomas label layers from Google Earth Engine

**Dataset construction**

2. **Tiling** — cuts scenes into 256×256 patches, filtering out mostly-empty ones
3. **Label attachment** — reprojects and aligns all label layers to each patch
4. **Class remapping** — simplifies MapBiomas codes into 5 classes (forest, pasture, agriculture, water, urban)
5. **Augmentation** — creates rotated copies (90°, 180°, 270°) per patch, with split-safe assignment

**Annotation**

6. **Index** — builds `dataset_index.jsonl` as the single source of truth for all patch metadata
7. **Captions** — generates structured land-cover descriptions using class fractions and spatial analysis (connected components, elongation, contact ratios)
8. **VQA** — generates presence/absence question-answer pairs per land-cover class
9. **Grounding** — generates normalized bounding boxes for deforested (converted) regions using connected component analysis

---

## Usage

```bash
pip install earthengine-api rasterio numpy scipy
earthengine authenticate
```

**1. Export scene + labels from GEE**
```bash
python scripts/pipeline/Exporter.py --year 2021 --drive-folder my_folder
```

**2–9. Run the full pipeline for a region**
```bash
python scripts/pipeline/run_region_pipeline.py --region santarem --year 2021
```

Or skip augmentation:
```bash
python scripts/pipeline/run_region_pipeline.py --region santarem --year 2021 --no-augment
```

Individual stages can also be run directly — see each script's docstring for options.

---

## Contact

João Guilherme Teixeira · [guilhermeteix2016@gmail.com](mailto:guilhermeteix2016@gmail.com)  
[LinkedIn](https://linkedin.com/in/guilherme-teixeira) · [GitHub](https://github.com/Guilherme779)

> ⚠️ Active research — evaluation code and results will be added progressively. Full results published after thesis submission.
