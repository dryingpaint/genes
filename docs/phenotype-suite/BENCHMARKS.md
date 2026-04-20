# Genome-to-Phenotype Benchmark Harness

**Purpose:** A multi-tier evaluation framework that honestly measures genome-to-phenotype prediction performance across trait types, ancestry groups, and prediction modalities — with mandatory strong baselines, anti-leakage protocols, and explicit theoretical ceilings per trait.

**Design principles:**
1. Every benchmark tier has a known theoretical ceiling (heritability or assay accuracy) and the gap between SOTA and ceiling is reported.
2. Every result is reported per ancestry group — aggregate-only metrics are rejected.
3. Every model must beat a strong supervised baseline, not just random or naive predictors.
4. Label leakage is treated as a disqualifying failure, not a caveat.

---

## Table of Contents

1. [Tier Architecture](#tier-architecture)
2. [Tier 0: Sanity Checks](#tier-0-sanity-checks-ceiling-10)
3. [Tier 1: Molecular Phenotypes](#tier-1-molecular-phenotypes-h²-02-08-for-cis)
4. [Tier 2: Complex Traits](#tier-2-complex-traits-h²-01-08)
5. [Tier 3: Model Organism Ceiling Tests](#tier-3-model-organism-ceiling-tests)
6. [Data Splits & Anti-Leakage Protocol](#data-splits--anti-leakage-protocol)
7. [Metrics Specification](#metrics-specification)
8. [Baseline Methods](#baseline-methods)
9. [Datasets & Access](#datasets--access)
10. [Reporting Template](#reporting-template)
11. [Known Failure Modes & Pitfalls](#known-failure-modes--pitfalls)

---

## Tier Architecture

```
Tier 0: Sanity Checks          ceiling ≈ 1.0       "Does the pipeline work at all?"
  │     Mendelian pathogenicity, PGx, HLA
  │
Tier 1: Molecular Phenotypes   ceiling = cis-h²     "Can we predict molecular effects?"
  │     eQTL, caQTL, sQTL, DMS
  │
Tier 2: Complex Traits         ceiling = h²_SNP     "Can we predict clinical phenotypes?"
  │     UKB quantitative traits, disease endpoints
  │
Tier 3: Model Organisms        ceiling ≈ h²_broad   "What's the upper bound under env control?"
        DGRP, 1001 Genomes, CC/DO mice
```

Each tier increases in difficulty and decreases in expected accuracy. A model that fails Tier 0 should not be evaluated on Tier 2. A model that matches baselines on Tier 2 but fails Tier 1 molecular phenotypes is likely learning shortcuts.

---

## Tier 0: Sanity Checks (ceiling ≈ 1.0)

These are near-solved problems. The purpose of Tier 0 is to verify pipeline correctness — if the system can't classify known pathogenic BRCA1 variants or call CYP2D6 diplotypes, nothing downstream is trustworthy.

### 0a. Mendelian Variant Pathogenicity

| Field | Specification |
|---|---|
| **Task** | Binary classification: pathogenic vs benign for known disease-associated variants |
| **Source data** | ClinVar 3-star+ P/LP/B/LB submissions (exclude VUS, 1-star, conflicting) |
| **Split** | Temporal: train on submissions ≤ 2023-12-31, test on 2024-01-01+ submissions. This prevents circularity from tools trained on earlier ClinVar releases. |
| **Variant classes** | Evaluate separately: (a) missense, (b) nonsense/frameshift, (c) splice-site (±2 bp), (d) non-coding regulatory, (e) in-frame indels |
| **N (approximate)** | ~50K–80K test variants (missense dominant); non-coding subset ~2K–5K |
| **Metrics** | AUROC, AUPRC, balanced accuracy, calibrated PPV at clinical thresholds (0.90 sensitivity, 0.99 specificity) |
| **Baselines** | AlphaMissense (missense), GPN-MSA (non-coding), SpliceAI (splice), CADD (legacy comparison) |
| **Expected ceiling** | AUROC ~0.95–0.98 for missense; ~0.85–0.90 for non-coding; ~0.95 for splice |
| **Failure criterion** | AUROC < 0.85 on missense = pipeline broken |

**Anti-circularity notes:**
- CADD, REVEL, BayesDel are trained on ClinVar labels — their AUCs on ClinVar are inflated. Report them as baselines but flag the circularity.
- AlphaMissense and EVE are unsupervised (no ClinVar labels in training) — their AUCs are honest.
- GPN-MSA uses evolutionary conservation (MSA), not clinical labels — honest for non-coding.

### 0b. Pharmacogenomics Diplotyping

| Field | Specification |
|---|---|
| **Task** | Star-allele diplotype calling and metabolizer status classification |
| **Source data** | GeT-RM truth set (CDC Genetic Testing Reference Materials Program): 137+ reference samples with orthogonally validated diplotypes across CPIC genes |
| **Split** | Leave-one-sample-out CV on GeT-RM; additionally test on All of Us pharmacogenomics callset |
| **Genes** | All 34 CPIC genes; CYP2D6 reported separately (hardest due to SVs) |
| **Metrics** | Exact diplotype match accuracy, metabolizer-bin accuracy (PM/IM/NM/RM/UM), per-gene accuracy |
| **Baselines** | PharmCAT v2 (general), Cyrius v1.1.1 (CYP2D6), Stargazer v2.0.2 (other PGx) |
| **Expected ceiling** | >0.97 overall; CYP2D6 specifically 0.965–0.993 (Cyrius on GeT-RM) |
| **Failure criterion** | CYP2D6 accuracy < 0.90 = SV handling broken |

### 0c. HLA Typing

| Field | Specification |
|---|---|
| **Task** | HLA allele calling at 2-field (4-digit) resolution |
| **Source data** | 1000 Genomes Phase 3 samples with validated HLA types (Gourraud 2014, Abi-Rached 2018); IMGTHLA reference |
| **Split** | Hold out by superpopulation (e.g., train on EUR/EAS/SAS, test on AFR/AMR) to test ancestry robustness |
| **Loci** | Class I: HLA-A, -B, -C. Class II: HLA-DRB1, -DQB1, -DQA1, -DPB1 |
| **Metrics** | Per-locus 2-field concordance, overall accuracy, ancestry-stratified accuracy |
| **Baselines** | HLA*LA (DNA), arcasHLA (RNA-seq if available) |
| **Expected ceiling** | >0.99 class I, >0.97 class II (from DNA-seq); >0.994 from RNA-seq |
| **Failure criterion** | Class I < 0.95 = MHC read extraction or graph alignment broken |

### 0d. Variant Calling Accuracy (GIAB)

| Field | Specification |
|---|---|
| **Task** | Germline SNV/indel calling accuracy on NIST truth sets |
| **Source data** | GIAB HG001–HG007, high-confidence regions; CMRG v1.0 (challenging medically-relevant genes) |
| **Split** | Standard GIAB benchmark regions; separately report CMRG difficult regions |
| **Metrics** | SNV F1, indel F1, Ti/Tv ratio, het/hom ratio |
| **Baselines** | DeepVariant v1.6, DRAGEN v4.2, GATK4 HC |
| **Expected ceiling** | SNV F1 > 0.999, indel F1 > 0.995 (high-confidence regions); CMRG indel F1 > 0.98 |
| **Tool** | `hap.py` (Illumina/Google) for benchmarking |
| **Failure criterion** | SNV F1 < 0.99 = caller or alignment broken |

---

## Tier 1: Molecular Phenotypes (h² 0.2–0.8 for cis)

These benchmarks test whether the model can predict specific molecular effects of specific variants — the mechanistic layer between genotype and complex trait.

### 1a. Gene Expression (cis-eQTL Prediction)

| Field | Specification |
|---|---|
| **Task** | Predict per-gene expression level (TPM) for held-out individuals from their genotype |
| **Source data** | GTEx v8: 838 donors, WGS + bulk RNA-seq across 49 tissues |
| **Split** | **Individual hold-out**: 80/20 donor split, stratified by reported ancestry. **Chromosome hold-out** (secondary): train on all donors but hold out chr8+chr21 to test generalization to unseen genomic regions. |
| **Evaluation granularity** | Per-gene, per-tissue. Report distribution of per-gene r across all genes, not just the aggregate. |
| **Metrics** | (1) Cross-individual Pearson r per gene per tissue (the Huang/Sasse test — the hard metric). (2) Proportion of genes with r > 0.1 (practical utility). (3) Cross-gene Pearson r on reference genome (the easy metric — for sanity only). (4) Direction-of-effect accuracy on fine-mapped eQTLs (SuSiE PIP > 0.9). |
| **Baselines** | (1) PrediXcan/FUSION elastic-net cis-eQTL models (the strong baseline). (2) Top eQTL linear model (single-variant). (3) Enformer/Borzoi personalized predictions (the foundation model comparison). (4) Mean expression per gene (null model). |
| **Expected ceiling** | Cis-h² per gene varies: median ~0.15, some genes >0.5. Cross-individual r should asymptotically approach √(cis-h²) — for median gene, r ≈ 0.39 at ceiling. Current SOTA (all models): r ≈ 0–0.1 genome-wide. |
| **Why this is the hardest honest test** | Huang 2023 and Sasse 2023 showed Enformer cross-individual Pearson r ≈ 0 despite cross-gene R ≈ 0.58. If a model can't beat elastic-net PrediXcan on this task, it is not learning cis-regulatory variant effects — it's learning gene-identity features. |
| **Stratification** | Report separately for: high-cis-h² genes (top decile), housekeeping vs tissue-specific genes, genes with vs without known regulatory elements in ENCODE. |

### 1b. Chromatin Accessibility (caQTL Prediction)

| Field | Specification |
|---|---|
| **Task** | Predict variant effect on chromatin accessibility at specific ATAC-seq peaks |
| **Source data** | (1) ENCODE4 donor-matched ATAC-seq (primary). (2) Degner 2012 dsQTL panel (LCLs, 70 individuals). (3) GTEx ATAC-seq (pilot, limited individuals). |
| **Split** | Individual hold-out + chromosome hold-out |
| **Metrics** | (1) Fine-mapped caQTL effect-size Spearman ρ. (2) Direction-of-effect accuracy on caQTLs with PIP > 0.5. (3) Regulatory-vs-control classification AUROC (DART-Eval protocol). |
| **Baselines** | (1) ChromBPNet ISM (in silico mutagenesis). (2) GPN-MSA log-likelihood ratio. (3) Enformer/Borzoi ISM. (4) Allelic-imbalance linear model (direct measurement baseline). |
| **Expected ceiling** | Variable per locus. DART-Eval shows ChromBPNet wins counterfactual VEP. |
| **Key reference** | Patel & Kundaje, DART-Eval (NeurIPS 2024) — use their exact evaluation protocol for comparability. |

### 1c. Splicing (sQTL Prediction)

| Field | Specification |
|---|---|
| **Task** | Predict variant effect on splicing quantitative trait (PSI / intron excision ratio) |
| **Source data** | GTEx v8 sQTLs (LeafCutter/SuSiE fine-mapped); SpliceVarDB (curated splice-affecting variants) |
| **Split** | Individual hold-out for GTEx sQTLs; SpliceVarDB is an independent test set |
| **Metrics** | (1) ΔPSI correlation for held-out sQTLs (Spearman ρ). (2) Binary splice-disruption classification AUROC on SpliceVarDB. (3) Per-tissue accuracy (Pangolin comparison). |
| **Baselines** | (1) SpliceAI Δscore. (2) Pangolin per-tissue scores. (3) Borzoi splice-junction predictions. (4) Evo 2 zero-shot (SpliceVarDB #1 unsupervised). |
| **Expected ceiling** | SpliceAI Δscore > 0.5 captures ~85% of known splice-disrupting variants. Continuous ΔPSI prediction ceiling limited by splicing measurement noise (LeafCutter intron cluster variance). |

### 1d. Protein Variant Effect (DMS)

| Field | Specification |
|---|---|
| **Task** | Zero-shot prediction of variant fitness effects measured by deep mutational scanning |
| **Source data** | ProteinGym benchmark: 217 DMS assays, ~2.7M single amino acid variants across diverse protein families |
| **Split** | Per-assay evaluation (standard ProteinGym protocol). No train/test split needed — models are evaluated zero-shot. |
| **Metrics** | (1) Per-assay Spearman ρ between predicted score and measured fitness. (2) Aggregate (mean/median) across assays. (3) Stratified by protein family, assay type (binding, stability, enzymatic activity), and MSA depth. |
| **Baselines** | (1) SaProt-1.3B (current SOTA, ~0.48). (2) ESM-2 15B (sequence-only pLM). (3) TranceptEVE (retrieval-augmented). (4) EVE (per-family VAE). (5) AlphaMissense (AF2-derived). |
| **Expected ceiling** | Assay noise limits maximum achievable ρ; for high-quality assays, ρ ceiling ~0.7–0.8. Aggregate ceiling across all assays is lower (~0.55–0.65) due to noisy assays. |
| **Key subtlety** | ProteinGym mixes assays of very different quality. A model that's great on well-powered assays (BRCA1 SGE, TEM-1 β-lactamase) but poor on noisy assays may be better than aggregate suggests. Report top-quartile and bottom-quartile assay performance separately. |

### 1e. BRCA1 Saturation Genome Editing (SGE)

| Field | Specification |
|---|---|
| **Task** | Predict functional effect of all possible SNVs in BRCA1 exons as measured by Findlay 2018 SGE |
| **Source data** | 3,893 SNVs in BRCA1 exons 2–5 with binary functional/non-functional classification from SGE |
| **Split** | Entire dataset is test — models are evaluated zero-shot |
| **Metrics** | (1) Binary classification accuracy (functional vs non-functional). (2) AUROC. (3) Stratified by variant type (missense, synonymous, splice-proximal). |
| **Baselines** | (1) Evo 2 (>90% accuracy). (2) AlphaMissense. (3) SpliceAI (splice-proximal subset). (4) CADD (legacy). |
| **Expected ceiling** | SGE assay has its own noise floor; ~95–97% of variants have clear functional classification. |
| **Why include this** | BRCA1 SGE is the gold-standard functional benchmark for coding+splice variant prediction. It's the one locus where we have near-complete ground truth. A model that fails here is not ready for clinical variant interpretation. |

---

## Tier 2: Complex Traits (h² 0.1–0.8)

These benchmarks test end-to-end genome-to-phenotype prediction for clinically relevant traits. The ceiling is SNP heritability (h²_SNP), which is substantially below broad-sense heritability for most traits due to non-additive effects, rare variants, and structural variants not captured by common-variant arrays.

### 2a. Quantitative Traits — UK Biobank

**Primary evaluation cohort:** UK Biobank (~500K participants, WGS + deep phenotyping)

| Trait | UKB Field | h²_SNP (EUR) | Expected PRS R² at SOTA | Clinical utility |
|---|---|---|---|---|
| Standing height | 50 | ~0.80 | 0.30–0.40 | Benchmark standard; best-case |
| BMI | 21001 | ~0.40 | 0.08–0.12 | GxE heavy (diet, activity) |
| LDL cholesterol | 30780 | ~0.50 | 0.12–0.18 | Statin use confounder — exclude treated |
| HDL cholesterol | 30760 | ~0.45 | 0.10–0.15 | |
| Triglycerides | 30870 | ~0.35 | 0.06–0.10 | Fasting status confounder |
| SBP | 4080 | ~0.30 | 0.05–0.08 | Medication confounder — exclude treated |
| HbA1c | 30750 | ~0.40 | 0.08–0.12 | T2D management confounder |
| eGFR (creatinine) | 30700 | ~0.35 | 0.05–0.08 | CKD-EPI formula ancestry issues |
| FEV1 | 3063 | ~0.45 | 0.08–0.12 | Smoking interaction |
| Grip strength | 46/47 | ~0.40 | 0.05–0.08 | Age/sex interaction |
| Heel bone mineral density | 3148 | ~0.50 | 0.10–0.15 | |
| Platelet count | 30080 | ~0.55 | 0.15–0.20 | |
| Mean corpuscular volume | 30040 | ~0.60 | 0.18–0.25 | High h², clean phenotype |

**Confounder handling:**
- Medication adjustment: For SBP and LDL, add 15 mmHg and 1 mmol/L respectively for individuals on antihypertensives/statins (standard UKB adjustment) OR exclude medicated individuals entirely (cleaner but lower N).
- Residualize for age, age², sex, genotyping batch, and top 20 PCs before evaluation.
- Report both raw and adjusted R².

**Evaluation protocol:**
1. Filter to unrelated individuals (KING < 0.0442, ~410K remain)
2. Stratify by genetic ancestry (PCA-based clustering into EUR/AFR/SAS/EAS/mixed)
3. Within each ancestry: 80/10/10 train/validation/test split
4. Train PRS weights on training set (or use external GWAS summary statistics)
5. Evaluate on held-out test set
6. Report per-ancestry and aggregate

### 2b. Disease Endpoints — UK Biobank

| Disease | ICD-10 / definition | Prevalence in UKB | PRS AUC (EUR, PRS only) | PRS AUC (+ clinical covariates) | Key covariates |
|---|---|---|---|---|---|
| Coronary artery disease | I20–I25 | ~8% | 0.62–0.64 | 0.78–0.82 | Age, sex, smoking, SBP, LDL, BMI, T2D, family hx |
| Type 2 diabetes | E11 + HbA1c ≥ 48 | ~6% | 0.60–0.65 | 0.80–0.85 | Age, sex, BMI, waist circ, family hx |
| Breast cancer (female) | C50 | ~7% (of females) | 0.63–0.68 | 0.68–0.72 | Age, family hx, menopausal status, HRT |
| Prostate cancer (male) | C61 | ~9% (of males) | 0.65–0.70 | 0.70–0.75 | Age, PSA, family hx |
| Schizophrenia | F20 | ~0.5% | ~0.72 | ~0.74 | Age, sex (limited clinical predictors) |
| Alzheimer's disease | G30 + F00 | ~1.5% | 0.75–0.80 | 0.80–0.85 | Age, sex, APOE (report with/without APOE) |
| Atrial fibrillation | I48 | ~5% | 0.62–0.65 | 0.75–0.80 | Age, sex, SBP, BMI |
| Asthma | J45 | ~12% | 0.55–0.58 | 0.60–0.65 | Age, sex, smoking, BMI |
| Major depressive disorder | F32–F33 | ~5% | 0.55–0.57 | 0.60–0.65 | Age, sex (limited clinical predictors) |
| Inflammatory bowel disease | K50–K51 | ~1% | 0.63–0.67 | 0.65–0.70 | Age, sex |

**Evaluation protocol:**
1. Same unrelated individual filter and ancestry stratification as quantitative traits
2. Case-control matching: 1:4 or 1:10 case:control ratio, matched on age + sex + ancestry
3. Metrics: (a) AUC for PRS alone, (b) AUC for clinical covariates alone, (c) AUC for PRS + clinical covariates, (d) net reclassification improvement (NRI), (e) calibration slope and intercept
4. **Mandatory: report EUR→AFR portability ratio** = AUC(AFR) / AUC(EUR) for PRS component

**APOE special handling for Alzheimer's:**
- APOE ε4 dominates AD PRS. Report with and without APOE variants to show incremental value of genome-wide PRS beyond the single strongest locus.

### 2c. Externally Visible Characteristics

| Trait | Method | SNPs | Expected AUC/accuracy |
|---|---|---|---|
| Eye color (blue/brown) | HIrisPlex-S | 6 (of 41 total) | >0.90 |
| Eye color (3-category) | HIrisPlex-S | 6 | ~0.85 |
| Hair color (4-category) | HIrisPlex-S | 22 | ~0.80 |
| Skin color (5-category) | HIrisPlex-S | 36 | ~0.75 |

**Why include:** These are the highest-accuracy complex trait predictions from DNA alone. If the pipeline can't reproduce HIrisPlex-S performance, something is fundamentally broken in the PRS calculation. Serves as a Tier 0.5 sanity check between Mendelian certainty and polygenic modesty.

### 2d. Ancestry-Stratified Portability Analysis

This is not a separate benchmark but a **mandatory cross-cutting analysis** applied to every Tier 2 trait:

| Metric | Definition | Expected pattern |
|---|---|---|
| Portability ratio | R²(target ancestry) / R²(EUR) | AFR: 0.25–0.50; SAS: 0.50–0.70; EAS: 0.40–0.60 |
| Absolute AUC drop | AUC(EUR) – AUC(target) | AFR: 0.05–0.15; SAS: 0.03–0.08 |
| PRS-CSx improvement | R²(PRS-CSx) / R²(SBayesRC-EUR-only) in target ancestry | 1.2–1.8× in AFR, 1.1–1.3× in SAS/EAS |

**Reporting requirement:** Any model that reports only aggregate or EUR-only performance is considered incomplete. Ancestry-stratified tables are mandatory.

---

## Tier 3: Model Organism Ceiling Tests

**Rationale:** Human complex trait prediction is bounded by uncontrollable environment (diet, lifestyle, socioeconomic factors, developmental stochasticity). Model organisms with controlled environments and high heritability let us ask: "given maximal genetic signal, how well can genome-to-phenotype prediction work?" This establishes the upper bound for the methods themselves.

### 3a. DGRP (Drosophila Genetic Reference Panel)

| Field | Specification |
|---|---|
| **Organism** | Drosophila melanogaster |
| **N** | 205 inbred lines, fully sequenced (Illumina + long-read for some) |
| **Phenotypes** | ~60 published: starvation resistance, chill coma recovery, startle response, lifespan, ethanol sensitivity, aggression, sleep, climbing speed, body size, wing morphology, metabolic rate, oxidative stress resistance, etc. |
| **Heritability** | Broad-sense H² typically 0.3–0.7 per trait (environment controlled in lab). h²_SNP approaches H² for inbred lines (no within-line segregation). |
| **Split** | Leave-N-lines-out CV (N=20, repeated 10×) |
| **Metrics** | Cross-validated prediction R²; per-trait and aggregate |
| **Baselines** | (1) GBLUP (genomic BLUP — linear mixed model). (2) BayesB (Bayesian variable selection). (3) Elastic-net on all SNPs. (4) Top-GWAS-hit linear model. |
| **Expected ceiling** | R² ≈ 0.3–0.7 per trait (≈ H²). If a model achieves R² > 0.5 on starvation resistance (H² ≈ 0.6), it's working well. |
| **Data access** | DGRP website (dgrp2.gnets.ncsu.edu); phenotype compendium from Mackay lab publications |
| **Why include** | This is your "can the model learn genotype-phenotype at all?" test. If a foundation model can't beat GBLUP on 205 inbred Drosophila lines with controlled environment, it's not learning genomic prediction — it's memorizing. |

### 3b. 1001 Genomes Arabidopsis

| Field | Specification |
|---|---|
| **Organism** | Arabidopsis thaliana |
| **N** | 1,135 natural accessions, fully sequenced |
| **Phenotypes** | 200+ traits from AraPheno database: flowering time (most studied), rosette diameter, leaf number, defense metabolites (glucosinolates), ion content (ionomics), disease resistance, root architecture |
| **Heritability** | Variable: flowering time H² ~0.6–0.8; metabolite traits H² ~0.3–0.6; morphological traits H² ~0.2–0.5 |
| **Split** | Leave-N-accessions-out CV (N=100, repeated 10×); stratified by genetic subpopulation (relict, admixed, etc.) |
| **Metrics** | Cross-validated prediction R²; per-trait and aggregate |
| **Baselines** | Same as DGRP: GBLUP, BayesB, elastic-net, top-GWAS |
| **Expected ceiling** | R² = 0.4–0.7 for flowering time; 0.2–0.5 for metabolites |
| **Data access** | 1001genomes.org; AraPheno (arapheno.1001genomes.org) |
| **Advantages over DGRP** | Larger N (1,135 vs 205), richer phenotyping (200+ traits), natural variation (not lab-selected). Population structure is complex (European expansion, relict populations) — tests whether the model handles stratification. |

### 3c. Collaborative Cross / Diversity Outbred Mice (optional extension)

| Field | Specification |
|---|---|
| **Organism** | Mus musculus |
| **N** | CC: ~70 strains (8 founder genomes fully characterized). DO: ~10,000+ mice with dense genotyping. |
| **Phenotypes** | Body weight, coat color, behavioral traits (anxiety, activity), immunological (cell counts, cytokine levels), metabolic (glucose tolerance, lipids), bone density |
| **Split** | Strain hold-out for CC; individual hold-out for DO |
| **Data access** | Mouse Phenome Database (phenome.jax.org); Churchill lab data portals |
| **Why optional** | More complex genetics (outbred, 8 founders) makes this a harder test than DGRP/Arabidopsis. Include if you want to test models on mammalian genomes with controlled environment, but lower priority than DGRP/Arabidopsis. |

---

## Data Splits & Anti-Leakage Protocol

### Split Types

| Split type | When to use | Implementation |
|---|---|---|
| **Temporal** | ClinVar pathogenicity (Tier 0a) | Train: submissions ≤ 2023-12-31. Test: submissions ≥ 2024-01-01. Prevents tools trained on earlier ClinVar from seeing test labels. |
| **Individual hold-out** | GTEx eQTL (1a), UKB traits (2a/2b) | Random 80/10/10 within each ancestry group, filtered for KING < 0.0442 (no 2nd-degree or closer relatives across splits) |
| **Chromosome hold-out** | GTEx molecular phenotypes (secondary) | Train: all chromosomes except chr8+chr21. Test: chr8+chr21. Prevents LD leakage across chromosomes. |
| **Leave-N-out CV** | Model organisms (Tier 3) | Leave out N lines/accessions per fold; repeat 10× and report mean ± SD |
| **Zero-shot** | ProteinGym DMS (1d), BRCA1 SGE (1e) | No training on target data at all; model evaluated as-is |

### Anti-Leakage Checklist

Every benchmark run must verify the following before reporting results:

- [ ] **Relatedness filter:** No individual in test set has KING > 0.0442 with any individual in training set
- [ ] **Temporal integrity:** For ClinVar-based benchmarks, no test variant's clinical significance was available in any database used during model training
- [ ] **Label leakage:** For supervised models (CADD, REVEL, etc.), verify that test variants were not in the training label set. If they were, report results with and without contaminated variants.
- [ ] **Chromosome leakage:** For chromosome hold-out splits, verify no features from test chromosomes are used (e.g., trans-eQTL mapping, cross-chromosome LD)
- [ ] **Population leakage:** For ancestry-stratified evaluation, verify that ancestry labels come from genetic PCA clustering, not self-reported ethnicity
- [ ] **Summary statistic leakage:** For PRS evaluation, verify that the GWAS summary statistics used for PRS training do not include the test individuals (critical for UKB, where many GWAS were run on the full cohort)
- [ ] **Batch effects:** For molecular phenotypes (GTEx, ENCODE), verify that test individuals are not segregated by sequencing batch

### UKB-Specific Anti-Leakage

The most common leakage in UKB PRS benchmarks is **GWAS summary statistics computed on the full UKB that include test individuals.** Solutions:

1. **External GWAS:** Use summary statistics from non-UKB GWAS (FinnGen, BBJ, CKB, external meta-analyses). Cleanest but often less powerful.
2. **Leave-out GWAS:** Re-run GWAS on training set only (expensive but gold standard). The pan-UKBB group provides pre-computed leave-out summary statistics for some traits.
3. **Kinship-group exclusion:** Identify the kinship group of each test individual and exclude all related individuals from GWAS (the Bycroft/UKB recommended approach).

**Mandatory:** Document which solution is used. Results using full-UKB summary statistics evaluated on UKB individuals are considered leaked and will be flagged.

---

## Metrics Specification

### Classification Tasks (Tier 0, Disease Endpoints)

| Metric | Definition | Why |
|---|---|---|
| **AUROC** | Area under receiver operating characteristic curve | Standard discrimination metric; threshold-independent |
| **AUPRC** | Area under precision-recall curve | Better than AUROC for imbalanced classes (rare diseases) |
| **Balanced accuracy** | (sensitivity + specificity) / 2 | Handles class imbalance; used by AlphaMissense |
| **Calibration slope + intercept** | Logistic regression of outcome on predicted probability | Tests whether predicted probabilities match observed rates |
| **NRI (Net Reclassification Improvement)** | Proportion of cases correctly reclassified up + controls correctly reclassified down | Clinical relevance: does the model change clinical decisions? |
| **PPV at fixed sensitivity** | Precision when sensitivity = 0.90 or 0.95 | Clinical: what fraction of flagged variants are truly pathogenic? |
| **NPV at fixed specificity** | Negative predictive value when specificity = 0.99 | Clinical: can we safely rule out pathogenicity? |

### Continuous Prediction Tasks (Tier 1 Molecular, Tier 2 Quantitative, Tier 3)

| Metric | Definition | Why |
|---|---|---|
| **Incremental R²** | R²(model + covariates) − R²(covariates only) | The PRS-attributable variance explained, beyond what demographics predict |
| **Cross-individual Pearson r** | Per-gene/per-trait correlation across individuals | The Huang/Sasse test: does the model predict inter-individual variation? |
| **Spearman ρ** | Rank correlation | Robust to outliers; used for DMS and eQTL effect sizes |
| **Direction-of-effect accuracy** | Fraction of variants where predicted direction matches observed | Binary: even if magnitude is wrong, is the direction right? |
| **Mean absolute error** | Mean |predicted − observed| | Interpretable in original units (mmHg, mg/dL, TPM) |
| **Ceiling-normalized R²** | R² / h²_SNP | What fraction of the achievable variance is the model capturing? |

### Reporting Requirements

For every benchmark:
1. Report point estimate + 95% CI (bootstrap or cross-validation SE)
2. Report per-ancestry (EUR, AFR, SAS, EAS minimum)
3. Report ceiling-normalized performance (R² / h²_SNP or AUC / theoretical max AUC)
4. Report comparison to every specified baseline
5. Report N (total and per-ancestry) and effective case count for disease endpoints

---

## Baseline Methods

Every tier has mandatory baselines. A result that doesn't compare to these is not publishable.

### Tier 0 Baselines

| Task | Strong baseline | Weak baseline (null) |
|---|---|---|
| Mendelian pathogenicity | AlphaMissense (missense), GPN-MSA (non-coding) | Random classifier; allele frequency only |
| PGx diplotyping | PharmCAT + Cyrius | Most common diplotype per gene |
| HLA typing | HLA*LA | Random allele from frequency-matched distribution |
| Variant calling | DeepVariant / DRAGEN | Reference-allele-only call (always call ref) |

### Tier 1 Baselines

| Task | Strong baseline | Weak baseline (null) |
|---|---|---|
| cis-eQTL | PrediXcan elastic-net | Mean expression per gene (no genetics) |
| caQTL | ChromBPNet ISM | Allele frequency × effect direction |
| sQTL | SpliceAI Δscore | No-change prediction (ΔPSI = 0) |
| DMS fitness | SaProt zero-shot | Random; blosum62 substitution score |
| BRCA1 SGE | Evo 2 zero-shot | Conservation score (phyloP) |

### Tier 2 Baselines

| Task | Strong baseline | Weak baseline (null) |
|---|---|---|
| Quantitative traits | SBayesRC PRS + covariates (age, sex, PCs) | Covariates only (no genetics) |
| Disease endpoints | SBayesRC PRS + clinical risk factors | Clinical risk factors only (Framingham, QRISK, etc.) |
| Cross-ancestry | PRS-CSx | EUR-trained SBayesRC applied to non-EUR (portability baseline) |

### Tier 3 Baselines

| Task | Strong baseline | Weak baseline (null) |
|---|---|---|
| DGRP / Arabidopsis | GBLUP (genomic BLUP) | Phenotype mean; random SNP subset |
| | BayesB / elastic-net | Top GWAS hit linear model |

---

## Datasets & Access

### Primary Datasets

| Dataset | N | Data types | Phenotype depth | Access mechanism | Turnaround |
|---|---|---|---|---|---|
| **UK Biobank** | ~500K | WGS, imputed genotypes, imaging, EHR, biochemistry, accelerometry, proteomics, metabolomics | Very deep (~2000 fields) | Application to UKB Access Committee | 2–8 weeks |
| **GTEx v8** | 838 | WGS + bulk RNA-seq (49 tissues) | Molecular (expression, splicing) | dbGaP (phs000424) | 1–4 weeks |
| **All of Us** | ~413K genomic | WGS, EHR, surveys, Fitbit, SDOH | Moderate (EHR + surveys) | Researcher Workbench (controlled tier) | 1–2 weeks |
| **ClinVar** | 3.6M classifications | Variant classifications | Clinical significance | FTP (public) | Immediate |
| **gnomAD v4.1** | 807K | WES/WGS allele frequencies | Population frequencies | Download / API (public) | Immediate |
| **ProteinGym** | 217 DMS assays | Variant fitness measurements | Functional (DMS) | GitHub / Zenodo (public) | Immediate |
| **GIAB** | 7 reference samples | WGS truth sets | Variant calling accuracy | FTP (public) | Immediate |
| **GeT-RM** | 137+ samples | PGx truth diplotypes | Pharmacogenomics | CDC request | 1–2 weeks |
| **DGRP** | 205 lines | WGS + 60 phenotypes | Broad (model organism) | Public (dgrp2.gnets.ncsu.edu) | Immediate |
| **1001 Genomes** | 1,135 accessions | WGS + 200 phenotypes | Broad (model organism) | Public (1001genomes.org) | Immediate |
| **ENCODE4** | Variable | ATAC/DNase/histone/TF ChIP-seq | Molecular (chromatin) | ENCODE portal (public) | Immediate |
| **COSMIC** | ~27M mutations | Somatic mutation catalog | Somatic | Download (academic license) | 1–2 days |

### Access Priority

For building and testing the pipeline, start with public datasets (no DAC approval needed):

1. **Immediate (public):** ClinVar, gnomAD, ProteinGym, GIAB, DGRP, 1001 Genomes, ENCODE4, PGS Catalog, COSMIC
2. **Quick DAC (1–4 weeks):** GTEx v8, All of Us Researcher Workbench
3. **Standard DAC (2–8 weeks):** UK Biobank
4. **Restricted:** MVP (VA-affiliated only), Genomics England (UK-based only)

**Development strategy:** Build and validate the pipeline on public datasets (Tiers 0 + 3 + partial Tier 1). Apply for UKB/GTEx/AoU access in parallel. Full Tier 2 evaluation requires UKB — plan for this timeline.

---

## Reporting Template

Every benchmark result should be reported in this format:

```
## [Trait/Task Name]

### Setup
- Dataset: [name, version, N]
- Split: [type, train/val/test N per ancestry]
- Anti-leakage: [which checks passed]

### Results

| Model | EUR (N=X) | AFR (N=X) | SAS (N=X) | EAS (N=X) | Aggregate |
|---|---|---|---|---|---|
| [Our model] | metric (95% CI) | metric (95% CI) | ... | ... | ... |
| [Strong baseline] | metric (95% CI) | ... | ... | ... | ... |
| [Null baseline] | metric (95% CI) | ... | ... | ... | ... |

### Ceiling Analysis
- h²_SNP (EUR): [value]
- Ceiling-normalized R²: [R² / h²_SNP]
- Gap to ceiling: [h²_SNP - R²]

### Portability
- EUR→AFR ratio: [R²_AFR / R²_EUR]
- PRS-CSx improvement over EUR-only: [ratio]

### Failure modes
- [Any systematic patterns in prediction errors]
- [Ancestry-specific failures]
- [Trait-specific confounders affecting results]
```

---

## Known Failure Modes & Pitfalls

### Pitfalls that have produced published false-positive results

| Pitfall | How it inflates results | Prevention |
|---|---|---|
| **ClinVar circularity** | CADD/REVEL trained on ClinVar labels; AUC on ClinVar test is inflated | Use only unsupervised predictors (AlphaMissense, EVE, GPN-MSA) OR temporal split |
| **UKB GWAS leakage** | PRS weights derived from full-UKB GWAS, evaluated on UKB individuals | Use external GWAS or leave-out summary statistics |
| **Cryptic relatedness** | Related individuals in train and test splits | KING < 0.0442 filter across splits |
| **Cross-gene R² vs cross-individual r** | Models that predict gene-mean expression look great cross-gene but fail cross-individual | Always report cross-individual Pearson r per gene |
| **Aggregate-only ancestry reporting** | Averaging across ancestries masks portability failure | Mandatory per-ancestry reporting |
| **Batch effects in molecular phenotypes** | GTEx/ENCODE batches correlate with genotype ancestry | Verify batch independence in splits |
| **Winner's curse in GWAS** | Effect sizes overestimated in discovery cohort | Use out-of-sample validation or shrinkage methods (LDpred2-auto, SBayesRC) |
| **Survivorship bias in biobanks** | UKB participants are healthier/wealthier than general population | Note generalizability limitations; do not claim population-level PPV |
| **APOE dominance in AD** | APOE ε4 alone gives AUC ~0.70 for AD; genome-wide PRS adds modest increment | Report with and without APOE |
| **Medication confounding** | SBP/LDL in treated individuals are artificially lowered | Exclude medicated OR apply standard adjustments |
| **Ascertainment bias in disease traits** | UKB disease ascertainment via ICD codes misses undiagnosed cases | Verify case definitions; consider enrichment with primary care/death registry data |

### Known hard problems where low performance is expected, not a bug

| Problem | Why it's hard | Expected performance |
|---|---|---|
| Personalized expression from genome | Models learn gene-identity, not cis-variant effects (Huang/Sasse) | Cross-individual r ≈ 0–0.1 |
| Distal enhancer effects | Models underweight >100 kb regulation (Karollus 2023) | Effect-size ρ < 0.3 for distal caQTLs |
| GxE interactions | Not modeled by any production PRS | Unmeasured; adds noise to R² ceiling |
| Rare variant prediction | Insufficient training data; most rare variants are singletons | AUC drops ~0.10 for rare vs common variants |
| Non-coding structural variants | Balanced translocations essentially unsolved | Classification accuracy < 0.7 for balanced SVs |
| Cross-ancestry complex trait PRS | LD patterns and causal variant frequencies differ | AFR R² = 25–50% of EUR R² |
| Behavioral/psychiatric traits | Low h², high GxE, phenotype measurement noise | PRS R² < 0.05 for most behavioral traits |

### What "beating the baseline" means per tier

| Tier | Baseline to beat | What beating it demonstrates |
|---|---|---|
| 0 | AlphaMissense / PharmCAT / HLA*LA | Pipeline correctness (should match, not necessarily beat) |
| 1 | PrediXcan / ChromBPNet / SpliceAI | Model captures genuine cis-regulatory variant effects beyond supervised baselines |
| 2 | SBayesRC + covariates | Model adds predictive value beyond GWAS-based PRS (extraordinary claim requiring extraordinary evidence) |
| 3 | GBLUP / BayesB | Model captures nonlinear or epistatic effects that linear models miss |

**Note on Tier 2:** Beating SBayesRC + clinical covariates for complex trait prediction from sequence alone would be a major result. Do not expect this initially. The honest goal for Tier 2 is matching SOTA PRS performance while providing interpretable variant-level contributions — not exceeding the PRS ceiling.

---

## Implementation Roadmap

### Phase 1: Public-data benchmarks (weeks 1–4)

Build and validate on datasets requiring no DAC approval:
- Tier 0: ClinVar temporal split + GIAB + GeT-RM (PGx)
- Tier 1d/1e: ProteinGym + BRCA1 SGE (zero-shot, no patient data)
- Tier 3: DGRP + 1001 Genomes Arabidopsis (fully public)

### Phase 2: Molecular phenotype benchmarks (weeks 4–8)

Requires GTEx v8 dbGaP access (apply in week 1):
- Tier 1a: GTEx eQTL prediction (the Huang/Sasse test)
- Tier 1b: ENCODE4 caQTL (public) + GTEx ATAC (dbGaP)
- Tier 1c: GTEx sQTL

### Phase 3: Complex trait benchmarks (weeks 8–16)

Requires UKB access (apply in week 1):
- Tier 2a: UKB quantitative traits
- Tier 2b: UKB disease endpoints
- Tier 2c: HIrisPlex-S (can pilot on 1000 Genomes with self-reported phenotypes)
- Tier 2d: Full ancestry-stratified portability analysis

### Phase 4: Integration and reporting (weeks 16–20)

- Cross-tier consistency analysis
- Publication-ready reporting
- Leaderboard infrastructure (if desired)
