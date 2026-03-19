# CRC Immune Phenotyping and Purine Pathway Analysis

This started as an extension of independent research I've been doing on HGPRT's role in DIPG (diffuse intrinsic pontine glioma). HGPRT, encoded by *HPRT1*, is the rate-limiting enzyme in the purine salvage pathway and is what makes DIPG cells sensitive to 6-thioguanine. The question I was trying to answer here was whether HPRT1 expression associates with immune phenotype in colorectal cancer — which would have implied that purine salvage activity modulates immune exclusion across cancer types.

It doesn't. At least not in bulk RNA-seq. That's actually the interesting part.

---

## What I found

**In TCGA-COAD (colorectal cancer, n=329):**

HPRT1 is essentially flat across immune phenotypes — median expression is nearly identical whether a tumor is immune-inflamed, excluded, or desert (Kruskal-Wallis p=0.55). No correlation with T-cell infiltration. HPRT1 ranked dead last in random forest feature importance for predicting immune phenotype.

ADA tells a different story. Adenosine deaminase (ADA) is significantly higher in immune-inflamed tumors compared to excluded and desert (KW p<0.001) and correlates positively with T-cell infiltration scores (ρ=0.33, p<0.001). ADA degrades adenosine, which is a well-characterized immunosuppressive metabolite in the tumor microenvironment — the Jabs/Ohta work on CD39/CD73/adenosine signaling is relevant here. So the purine pathway is doing something in relation to immune phenotype, just through ADA-mediated adenosine catabolism rather than HPRT1-mediated salvage.

The random forest classifier predicts immune phenotype with 89.7% balanced accuracy (5-fold CV). Top features are CD8A, FAP, CXCL9, TGFB1 — exactly what you'd expect from the literature on T-cell inflamed vs excluded signatures.

**On HPRT1 being null in bulk:**

The most plausible explanation is spatial restriction. Bulk RNA-seq averages across all cell types in a tumor biopsy — if HPRT1 activity is cell-type-specific (e.g., elevated specifically in tumor cells surrounded by stroma, as opposed to immune-infiltrated regions), that signal gets washed out. Spatial transcriptomics would be the right tool to test this, which is partly why I found Wala et al.'s ongoing work on spatial tumor-immune interactions in CRC relevant.

**In DIPG (GEO GSE50021, Buczkowicz et al. 2014, n=35 tumors + 10 normal brain):**

This dataset is Illumina microarray, not RNA-seq, so the expression values are on a completely different scale — I'm not comparing numbers across datasets. Within the DIPG dataset alone: HPRT1 expression does not significantly predict overall survival in this cohort (Spearman ρ≈-0.15, logrank p≈0.44). Small n (35 patients) limits power substantially. The dataset does have OS in years for each patient which makes survival analysis possible, but I'd want a larger cohort to say anything definitive.

What's clear from the DIPG data is that T-cell marker genes (CD8A, GZMB, IFNG, CXCL9) are generally low in DIPG tumors relative to normal brain — consistent with DIPG being an immunologically cold tumor, which is part of why checkpoint blockade has largely failed in DIPG trials.

---

## Datasets

| Dataset | Source | Access |
|---------|--------|--------|
| TCGA-COAD HiSeqV2 | UCSC Xena public hub | Free, no login |
| TCGA-COAD clinical | UCSC Xena public hub | Free, no login |
| GSE50021 (DIPG) | GEO, Buczkowicz et al. 2014 | Free, GEOparse |

---

## How to run

**Google Colab (easiest):**
```
!pip install GEOparse pandas numpy matplotlib seaborn scipy scikit-learn lifelines requests
!python analysis.py
```

First run downloads ~50 MB of TCGA data and ~22 MB of DIPG data. After that everything is cached locally so subsequent runs are instant.

**Locally:**
```bash
pip install GEOparse pandas numpy matplotlib seaborn scipy scikit-learn lifelines requests
python analysis.py
```

**Outputs:**
- `figures.png` — 12-panel figure
- `tcga_expr.pkl`, `tcga_clin.pkl` — TCGA cache
- `dipg_cache.pkl` — DIPG cache

---

## Methods

**Immune phenotype classification** follows Teng et al. (2015) *Cancer Cell* and Hegde et al. (2016) *Clin Cancer Res*. Tumors are scored by mean Z-score of the T-cell inflamed gene expression profile (Ayers et al. 2017, *JCI Insight*: CD8A, CD8B, GZMB, PRF1, IFNG, CXCL9, CXCL10, IDO1, PDCD1, LAG3, TIGIT) and an immune exclusion signature (TGFB1, VEGFA, MMP9, FAP, ACTA2, TGM2). Tumors above the T-cell inflamed median are classified as Inflamed; below that threshold but above the exclusion median are Excluded; rest are Desert.

Cytolytic activity (CYT) is computed as the log10 geometric mean of GZMA and PRF1 in linear RSEM space, per Rooney et al. (2015) *Cell*.

Random forest: 300 trees, max depth 6, min 10 samples per leaf, class-balanced, 5-fold stratified CV.

DIPG survival: Kaplan-Meier by median HPRT1 split, logrank test, Spearman correlation with continuous OS. All events assumed observed (DIPG is nearly universally fatal within the study follow-up window).

---

## Dependencies

```
pandas numpy matplotlib seaborn scipy scikit-learn lifelines requests GEOparse
```

---

## What I still want to do

- Apply this to spatial transcriptomics data (10x Visium CRC datasets are publicly available) to test whether HPRT1 shows spatial colocalization with excluded immune niches that bulk RNA-seq misses
- Look at ADA more carefully — specifically whether ADA expression varies with CD39/CD73 co-expression, which would place it more precisely in the adenosine immunosuppression axis
- Find a larger DIPG cohort with survival data (the CBTTC dataset through PedcBioPortal has more cases)
