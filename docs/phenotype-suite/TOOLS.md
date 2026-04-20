# Genome-to-Phenotype Tool Suite

**Purpose:** A minimal, non-redundant set of bioinformatics tools that Claude can orchestrate to analyze a genome and predict phenotypes across germline, somatic, cfDNA, and epigenomic modalities.

**Design principle:** One best-in-class tool per task. Redundant legacy tools (CADD, REVEL, PolyPhen-2, SIFT, Enformer, deconstructSigs, etc.) are deliberately excluded. Where two tools cover genuinely different signal sources or input types, both are retained and the distinction is documented.

**Total tool count:** ~26 (down from ~120+ in the broader landscape).

---

## Table of Contents

1. [Variant Calling](#1-variant-calling)
2. [Variant Annotation & Databases](#2-variant-annotation--databases)
3. [Variant Effect Prediction — Missense](#3-variant-effect-prediction--missense)
4. [Variant Effect Prediction — Cross-type Interpretable (EVEE)](#4-variant-effect-prediction--cross-type-interpretable-evee)
5. [Variant Effect Prediction — Non-coding](#5-variant-effect-prediction--non-coding)
6. [Splicing Prediction](#6-splicing-prediction)
7. [Structural Variant Interpretation](#7-structural-variant-interpretation)
8. [Mendelian Disease Prioritization](#8-mendelian-disease-prioritization)
9. [Polygenic Risk Scores](#9-polygenic-risk-scores)
10. [Pharmacogenomics](#10-pharmacogenomics)
11. [HLA Typing](#11-hla-typing)
12. [Ancestry Inference](#12-ancestry-inference)
13. [Somatic Variant Calling](#13-somatic-variant-calling)
14. [Mutational Signatures](#14-mutational-signatures)
15. [Somatic Driver & Biomarker Annotation](#15-somatic-driver--biomarker-annotation)
16. [Immune Deconvolution](#16-immune-deconvolution)
17. [cfDNA Analysis](#17-cfdna-analysis)
18. [Epigenetic Clocks & Methylation](#18-epigenetic-clocks--methylation)
19. [Report Generation](#19-report-generation)

---

## 1. Variant Calling

### 1a. DeepVariant (Germline — Long-read & Short-read)

| Field | Detail |
|---|---|
| **What it does** | Germline SNV/indel calling via CNN pileup image classification |
| **Why this tool** | Wins PrecisionFDA Truth Challenges; best long-read caller (ONT/PacBio HiFi); competitive short-read. Open-source (Google). |
| **Input** | Aligned BAM/CRAM + reference FASTA |
| **Output** | VCF/gVCF |
| **Key parameters** | `--model_type` (WGS, WES, PACBIO, ONT_R104, HYBRID_PACBIO_ILLUMINA); `--num_shards` for parallelism |
| **Runtime** | ~2–4 hours for 30× WGS on 64 CPUs; ~30 min with GPU |
| **Install** | Docker (`google/deepvariant`), Bioconda, or Singularity |
| **Version** | v1.6+ (2024) |
| **Limitations** | No somatic calling (see DeepSomatic). No SV calling (use Sniffles2/Manta). MNPs require postprocessing. |
| **Replaces** | GATK4 HaplotypeCaller (legacy), Sentieon DNAscope (proprietary reimplementation), FreeBayes, Clair3 |

### 1b. DRAGEN (Germline — Clinical Short-read)

| Field | Detail |
|---|---|
| **What it does** | FPGA-accelerated alignment + variant calling pipeline |
| **Why this tool** | Ships on Illumina NovaSeq X; FDA-recognized in multiple CDx contexts; fastest clinical-grade pipeline. Industry standard for short-read clinical WGS/WES. |
| **Input** | FASTQ or BAM + reference |
| **Output** | VCF/gVCF, BAM, QC metrics |
| **Key parameters** | `--enable-variant-caller true`, `--vc-emit-ref-confidence GVCF` |
| **Runtime** | ~22 min for 30× WGS (FPGA); ~2 hours CPU-mode via DRAGEN-GATK |
| **Install** | Illumina DRAGEN server (FPGA hardware) or DRAGEN-GATK (open-source CPU fallback via GATK4) |
| **Version** | v4.2+ (2024) |
| **Limitations** | FPGA version requires Illumina hardware or cloud instances. CPU fallback (DRAGEN-GATK) is slower. Proprietary core. |
| **Replaces** | BWA-MEM2 + GATK4 HC pipeline (DRAGEN integrates alignment + calling) |
| **When to use DeepVariant vs DRAGEN** | DeepVariant for long-read, research flexibility, and open-source reproducibility. DRAGEN for clinical short-read production where speed and regulatory pedigree matter. For this toolkit, default to DeepVariant unless processing Illumina clinical samples. |

---

## 2. Variant Annotation & Databases

### 2a. Ensembl VEP (Variant Effect Predictor)

| Field | Detail |
|---|---|
| **What it does** | Functional consequence annotation: transcript mapping, amino acid change, regulatory region overlap, allele frequency, plugin scores |
| **Why this tool** | De facto standard for variant annotation. Extensible plugin architecture (>50 plugins). GENCODE/RefSeq transcript models. |
| **Input** | VCF, HGVS, or genomic coordinates |
| **Output** | Annotated VCF, TSV, or JSON |
| **Key plugins** | `--plugin AlphaMissense`, `--plugin SpliceAI`, `--plugin CADD` (for legacy comparison only), `--plugin dbNSFP` (aggregates 30+ scores), `--plugin gnomADc` |
| **Key parameters** | `--assembly GRCh38`, `--cache`, `--merged` (GENCODE+RefSeq), `--pick` (one consequence per variant) or `--per_gene` |
| **Runtime** | ~1–2 hours for WGS VCF with cache; minutes for panel VCF |
| **Install** | Perl + Bioconda or Docker (`ensemblorg/ensembl-vep`). Cache files ~18 GB (homo_sapiens_merged). |
| **Version** | v112+ (2024) |
| **Limitations** | Perl dependency. Cache download is large. Plugin ecosystem can be fragile across versions. |
| **Replaces** | SnpEff (simpler but less extensible), ANNOVAR (license restrictions, less maintained) |

### 2b. Reference Databases (not tools — data dependencies)

These are not standalone tools but required data layers that VEP and downstream tools query:

| Database | Current Scale | What It Provides | Access |
|---|---|---|---|
| **ClinVar** | 3.6M classifications, ~2.4M variants | Clinical significance (P/LP/VUS/LB/B), review status (0–4 stars), submitter | FTP download, VEP plugin, API |
| **gnomAD v4.1** | 807,162 individuals (731K WES + 76K WGS) | Population allele frequencies by ancestry, pLoF constraint (LOEUF), SVs | Download, API, Hail Tables |
| **ClinGen** | 55+ VCEPs, ~2,420 gene–disease pairs | Gene–disease validity, dosage sensitivity, variant curation | API, downloads |
| **OncoKB** | ~6,000 alterations, ~800 genes | Therapeutic actionability levels (1–4, R1/R2); FDA-recognized | API (requires license for commercial) |
| **CIViC** | ~4,000+ variant-evidence summaries | Open crowd-curated clinical evidence (5 tiers) | API, downloads, open-source |
| **COSMIC** | ~27M coding mutations | Somatic mutation catalog, signatures, census genes | Download (academic license) |
| **PGS Catalog** | ~5,000 scores, 700 traits | Pre-computed PRS weights with harmonized metadata | FTP, API |
| **CPIC** | 34 genes, 164 drugs | Pharmacogenomic guidelines, gene–drug pairs | API (>80K queries/month), downloads |
| **PharmVar** | All pharmacogenes | Canonical star-allele nomenclature and haplotype definitions | Web, API |
| **GIAB/NIST** | HG001–HG007 + CMRG v1.0 | Truth sets for benchmarking variant callers | FTP |

---

## 3. Variant Effect Prediction — Missense

### 3a. AlphaMissense

| Field | Detail |
|---|---|
| **What it does** | Classifies all possible human missense variants as pathogenic/benign using AF2-derived structural features + protein language model + primate frequency |
| **Why this tool** | AUC ~0.94 on ClinVar; classifies 89% of 71M possible human missense variants; unsupervised (no ClinVar label leakage); used as ACMG PP3/BP4 evidence |
| **Input** | Protein ID + amino acid substitution, or pre-computed lookup (all human missense scores available) |
| **Output** | Pathogenicity score (0–1), classification (likely_benign / ambiguous / likely_pathogenic) |
| **Thresholds** | Likely benign <0.34; Likely pathogenic >0.564 (Cheng 2023 calibrated); recalibrated thresholds per Cubuk 2024 for ACMG evidence strength |
| **Runtime** | Lookup: instant. De novo scoring: requires AF2 inference (~minutes/protein on GPU) |
| **Install** | Pre-computed TSV (3.6 GB) from Google DeepMind; or run via AlphaFold + AlphaMissense code (JAX/GPU) |
| **Version** | v1 (2023) |
| **Limitations** | Not FDA-cleared. Does not handle in-frame indels, nonsense, or non-coding variants. Pre-computed scores cover canonical transcripts only. |
| **Replaces** | CADD, REVEL, BayesDel, PolyPhen-2, SIFT, PROVEAN, MutationTaster, DANN — all of which have ClinVar training circularity and/or lower accuracy |

### 3b. EVE (Evolutionary model of Variant Effect)

| Field | Detail |
|---|---|
| **What it does** | Unsupervised VAE trained per protein family on deep MSAs; predicts variant pathogenicity without any labels |
| **Why this tool** | ClinVar AUC ~0.91; fully label-free (no circularity risk); strong on rare/novel variants where AlphaMissense's primate frequency signal is absent |
| **Input** | Protein family MSA + variant position |
| **Output** | EVE score (continuous); classification at 3 uncertainty thresholds |
| **Runtime** | Pre-computed for ~3,200 human proteins; de novo requires MSA generation + VAE training (~hours/family on GPU) |
| **Install** | GitHub (`OATML-Markslab/EVE`), pre-computed scores downloadable |
| **Version** | v1 (2021) |
| **Limitations** | Coverage: ~3,200 protein families (vs AlphaMissense's all-protein coverage). Requires deep MSAs; fails for orphan genes. |
| **Complementarity with AlphaMissense** | EVE provides an independent, orthogonal signal. When AlphaMissense and EVE agree, confidence is high. When they disagree, the variant warrants manual review. Use both; they are not redundant — different architectures, different training signals. |

### 3c. SaProt (Structure-aware Protein language model)

| Field | Detail |
|---|---|
| **What it does** | ESM-2 backbone augmented with 3Di structural tokens from Foldseek; zero-shot variant effect prediction via masked marginal scoring |
| **Why this tool** | SOTA on ProteinGym DMS benchmark (aggregate Spearman ~0.48 at 1.3B params); captures structural context that sequence-only pLMs miss |
| **Input** | Protein sequence + AF2/ESMFold 3Di string + variant |
| **Output** | Log-likelihood ratio (zero-shot VEP score) |
| **Runtime** | ~seconds/protein on GPU (inference only) |
| **Install** | GitHub (`westlake-repl/SaProt`), HuggingFace weights (650M, 1.3B) |
| **Version** | v1 (2024) |
| **Limitations** | Research tool; not validated for clinical use. Requires 3Di generation (Foldseek + structure). Best for research DMS interpretation, not clinical pathogenicity classification. |
| **When to use** | Deep mutational scan interpretation, protein engineering, understanding fitness landscape. For clinical pathogenicity, prefer AlphaMissense + EVE. |

---

## 4. Variant Effect Prediction — Cross-type Interpretable (EVEE)

### 4a. EVEE (Evo Variant Effect Explorer)

| Field | Detail |
|---|---|
| **What it does** | Cross-variant-type pathogenicity prediction with mechanistic explanations. Trains lightweight probes on frozen Evo 2 embeddings to (1) classify pathogenicity, (2) produce disruption profiles across biological features (splice sites, regulatory elements, structural domains), and (3) synthesize natural-language explanations via a frontier reasoning model. |
| **Why this tool** | **Fills three gaps in the suite:** (1) Indel pathogenicity — AlphaMissense is missense-only, GPN-MSA is SNV-only; EVEE handles SNVs and indels. (2) Mechanistic explanation — no other tool generates structured disruption profiles or natural-language reasoning for why a variant is predicted pathogenic. (3) VUS interpretation — pre-computed predictions + explanations for all 4.2M ClinVar variants including ~2M VUS. |
| **Input** | ClinVar variant ID, or genomic coordinate + ref/alt (for pre-computed lookup). De novo scoring requires Evo 2 inference. |
| **Output** | (1) Pathogenicity score. (2) Disruption profile: per-feature shift scores across splice, regulatory, structural annotations. (3) Natural-language mechanistic explanation. |
| **Architecture** | Pathogenicity probe: covariance-based sequence pooling (not mean pooling) on Evo 2 embedding differences between ref and alt sequences. Annotation probes: trained on reference embeddings to predict known biological features, then applied to quantify variant-induced shifts. Explanation layer: frontier LLM synthesizes disruption profile + pathogenicity score + gene context. |
| **Key results** | AUROC 0.997 on 839K ClinVar SNVs; 0.991 on indels (zero-shot transfer); explanation quality 3.8/5 composite (vs 2.8/5 from metadata alone); transfers to BRCA1/BRCA2/TP53/LDLR DMS datasets |
| **Runtime** | Pre-computed lookup: instant (web interface at evee.goodfire.ai). De novo: requires Evo 2 inference (~seconds/variant on GPU) + probe forward pass. |
| **Install** | Web interface (evee.goodfire.ai); preprint under review (Goodfire, 2025) |
| **Version** | v1 (2025, preprint) |
| **Limitations** | **ClinVar circularity warning:** Evo 2 base model is unsupervised (no human variants in training), but the pathogenicity probe is supervised on ClinVar labels. The 0.997 AUROC is evaluated on the same database used for probe training — without a temporal split, this is likely inflated relative to performance on truly novel variants. Compare to AlphaMissense's 0.94 AUROC which uses zero ClinVar labels. Annotation probes surface only known biological mechanisms — novel pathways are invisible. Explanations are computational hypotheses, not clinical evidence. Preprint status — not yet peer-reviewed. |
| **Role in suite** | **Supplementary, not primary.** Use AlphaMissense (missense) and GPN-MSA (non-coding SNVs) as primary pathogenicity scorers because they are unsupervised and their benchmark numbers are honest. Use EVEE for: (1) indel pathogenicity (unique coverage), (2) mechanistic explanation generation (unique capability), (3) VUS triage where the explanation helps prioritize functional follow-up. In the Claude report layer, present EVEE explanations alongside AlphaMissense/GPN-MSA scores — the score says "how pathogenic," the explanation says "why." |
| **Complementarity** | AlphaMissense: score (unsupervised, missense-only). GPN-MSA: score (unsupervised, non-coding SNVs). EVE: score (unsupervised, protein families). **EVEE: score + explanation (supervised probe, all variant types including indels).** Four tools, four different signals, minimal redundancy. |

---

## 5. Variant Effect Prediction — Non-coding

### 5a. GPN-MSA

| Field | Detail |
|---|---|
| **What it does** | Masked language model trained on 100-way vertebrate MSA (Cactus alignment); predicts per-nucleotide log-likelihood ratios for variant effect |
| **Why this tool** | **SOTA on ClinVar non-coding and OMIM promoter variants** at only ~85M parameters. Beats Evo 2 (40B params), CADD, phyloP, all DNA LMs. Alignment conditioning > raw self-supervised scale. |
| **Input** | Genomic coordinate + ref/alt allele (queries pre-aligned MSA columns) |
| **Output** | Log-likelihood ratio (LLR) per variant; interpretable as evolutionary constraint at single-nucleotide resolution |
| **Runtime** | Seconds per variant; batch scoring of full VCF ~minutes |
| **Install** | GitHub (`songlab-cal/gpn`), pre-computed scores for all possible SNVs available |
| **Version** | v2 (2024/2025) |
| **Limitations** | Context window only 128 nt (relies on MSA, not long-range sequence). Does not predict *which* molecular phenotype is affected — only whether the variant is deleterious. Cannot handle structural variants or indels >1 bp. |
| **Replaces** | Nucleotide Transformer, HyenaDNA, DNABERT-2, GENA-LM, Caduceus — all DNA LMs that fail to beat GPN-MSA on non-coding VEP despite much larger parameter counts |

### 5b. ChromBPNet

| Field | Detail |
|---|---|
| **What it does** | Base-resolution ATAC-seq/DNase-seq profile predictor with explicit Tn5/DNase-I bias factorization; predicts chromatin accessibility from sequence |
| **Why this tool** | **Wins DART-Eval counterfactual VEP** against all DNA LMs. The hard-to-beat supervised baseline for regulatory variant effect prediction. Explicitly models assay bias, giving cleaner signal. |
| **Input** | DNA sequence (1 kb window centered on region of interest) |
| **Output** | Base-resolution predicted accessibility profile; variant effect as profile difference |
| **Key parameters** | Cell-type-specific models (train one per ATAC/DNase experiment) |
| **Runtime** | Training: ~hours per cell type on GPU. Inference: milliseconds per sequence. |
| **Install** | GitHub (`kundajelab/chrombpnet`) |
| **Version** | 2023 |
| **Limitations** | Cell-type-specific (need one model per cell type/condition). Predicts accessibility, not expression. Short context (1 kb) — cannot capture distal enhancer effects. |
| **Complementarity with GPN-MSA** | GPN-MSA answers "is this variant evolutionarily constrained?" (Mendelian/deleterious). ChromBPNet answers "does this variant change chromatin accessibility in cell type X?" (mechanistic). Different questions; use both. |

### 5c. AlphaGenome / Borzoi

| Field | Detail |
|---|---|
| **What it does** | Sequence-to-multi-modal regulatory prediction: gene expression, chromatin accessibility, histone marks, CTCF binding, splicing, 3D contacts |
| **Why this tool** | AlphaGenome: 1 Mb context, single-bp resolution, 11 output modalities, SOTA on 22/24 track tasks and 24–25/26 VEP tasks. Borzoi: 524 kb context, 32 bp resolution, unifies expression + splicing + polyA. These are the best "what does this sequence do?" predictors. |
| **Input** | 1 Mb (AlphaGenome) or 524 kb (Borzoi) DNA sequence |
| **Output** | Multi-track predictions: RNA-seq coverage, CAGE, ATAC, histone ChIP, CTCF, splice junctions, Hi-C contact maps (AlphaGenome) |
| **Runtime** | AlphaGenome: ~4 hours to train on TPUs; inference ~seconds/region on GPU. Borzoi: similar. Flashzoi: 3× faster Borzoi inference. |
| **Install** | AlphaGenome: DeepMind release (JAX/TPU). Borzoi/Flashzoi: GitHub (`calico/borzoi`, `johahi/flashzoi`), PyTorch. |
| **Version** | AlphaGenome (2025), Borzoi (2024), Flashzoi (2025) |
| **Limitations** | **Cross-individual Pearson r ≈ 0–0.1** for personalized expression prediction (Huang 2023, Sasse 2023). Models learn gene-identity features, not cis-variant effects. Systematically underweight distal enhancers (Karollus 2023). Best for reference-genome track prediction and eQTL direction-of-effect (AUROC ~0.75–0.80), not for predicting personal expression levels. |
| **When to use** | Predicting functional impact of a variant on molecular phenotypes (which track, which direction). Not for predicting an individual's expression from their genome. |
| **Replaces** | Enformer (superseded by both AlphaGenome and Borzoi), Basenji2, DeepSEA |

---

## 6. Splicing Prediction

### 6a. SpliceAI

| Field | Detail |
|---|---|
| **What it does** | Predicts splice-altering effects of SNVs and indels within 10 kb of splice sites using dilated residual CNN |
| **Why this tool** | Reference standard for splice variant effect prediction (Jaganathan 2019). Delta scores directly interpretable: acceptor gain/loss, donor gain/loss. Widely validated clinically. |
| **Input** | VCF + reference FASTA (or pre-computed scores for all possible SNVs) |
| **Output** | Four delta scores per variant: DS_AG, DS_AL, DS_DG, DS_DL (0–1 scale); position of max effect |
| **Thresholds** | Δ ≥ 0.2 (increased sensitivity), Δ ≥ 0.5 (high precision, recommended for clinical); Δ ≥ 0.8 (very high confidence) |
| **Runtime** | Pre-computed lookup: instant. De novo: ~4 hours for WGS VCF on GPU |
| **Install** | `pip install spliceai`; pre-computed scores via Illumina Basespace (~60 GB); VEP plugin available |
| **Version** | v1.3+ |
| **Limitations** | No tissue specificity. 10 kb context misses ultra-distal splice regulation. Does not predict absolute PSI, only delta. |

### 6b. Pangolin

| Field | Detail |
|---|---|
| **What it does** | Tissue-specific splice effect prediction using 4-species multi-genome training |
| **Why this tool** | Adds tissue specificity that SpliceAI lacks. Trained on human, macaque, mouse, rat — captures cross-species splice conservation. |
| **Input** | Genomic coordinate + ref/alt + tissue |
| **Output** | Per-tissue splice gain/loss scores |
| **Runtime** | Comparable to SpliceAI |
| **Install** | GitHub (`tkzeng/Pangolin`) |
| **Version** | v1 (2022) |
| **Complementarity with SpliceAI** | SpliceAI gives a single splice-disruption score. Pangolin tells you *in which tissues* the splice effect manifests. For a variant with SpliceAI Δ = 0.4, Pangolin can distinguish brain-specific vs ubiquitous splicing — critical for Mendelian neurological vs systemic disease. Use SpliceAI as primary screen, Pangolin for tissue-specific follow-up. |

---

## 7. Structural Variant Interpretation

### 7a. AnnotSV

| Field | Detail |
|---|---|
| **What it does** | Comprehensive SV annotation: gene overlap, known pathogenic SVs (ClinVar, ClinGen, DECIPHER), population frequency (gnomAD-SV, DGV), regulatory element disruption |
| **Why this tool** | Most complete SV annotation framework. Integrates ACMG/ClinGen SV interpretation guidelines. |
| **Input** | BED or VCF of SVs |
| **Output** | Annotated TSV with pathogenicity classification |
| **Install** | GitHub (`lgmgeo/AnnotSV`), Bioconda |
| **Version** | v3.4+ (2024) |

### 7b. ClassifyCNV

| Field | Detail |
|---|---|
| **What it does** | Automated ACMG/ClinGen-based classification of CNVs (deletions and duplications) |
| **Why this tool** | Implements the 2020 ACMG/ClinGen CNV interpretation framework systematically |
| **Input** | BED or VCF of CNVs |
| **Output** | ACMG classification (Pathogenic → Benign) with evidence codes |
| **Install** | GitHub (`Genotek/ClassifyCNV`) |
| **Limitations** | Deletions well-handled; duplications and balanced rearrangements remain challenging across all tools. |

---

## 8. Mendelian Disease Prioritization

### 8a. Exomiser

| Field | Detail |
|---|---|
| **What it does** | Phenotype-driven variant/gene prioritization for rare Mendelian disease. Combines variant pathogenicity with cross-species phenotype matching (hiPHIVE: human, mouse, fish phenotype ontologies). |
| **Why this tool** | 72–77% top-1 accuracy for known disease-gene recovery. Standard in rare disease diagnostics. Integrates frequency filtering, inheritance mode, variant effect, and phenotype matching in one score. |
| **Input** | VCF + HPO term list (patient phenotypes) + pedigree (optional) |
| **Output** | Ranked gene list with composite scores; per-variant evidence breakdown |
| **Key parameters** | `--hpo-ids` (HPO terms), `--inheritance-modes` (AD, AR, XL, MT), `--frequency-threshold`, `--pathogenicity-sources` (can include AlphaMissense) |
| **Runtime** | ~2–10 minutes per case |
| **Install** | Java JAR + data bundle (~80 GB for full cross-species phenotype + variant databases); Docker available |
| **Version** | v14+ (2024) |
| **Limitations** | Requires well-curated HPO terms — garbage in, garbage out. Cross-species phenotype matching fails for human-specific diseases without animal models. |
| **Replaces** | LIRICAL (same group, similar approach), Phen2Gene (weaker), Moon (commercial/discontinued) |

### 8b. AMELIE 2

| Field | Detail |
|---|---|
| **What it does** | NLP-based literature curation for variant–disease association. Reads PubMed/PMC full text to score gene–phenotype relevance. |
| **Why this tool** | Orthogonal signal to Exomiser: literature-based rather than phenotype-ontology-based. Catches recently published gene discoveries that haven't propagated to OMIM/HPO yet. |
| **Input** | Gene list + HPO terms (or free-text phenotype description) |
| **Output** | Per-gene literature relevance score with supporting paper citations |
| **Runtime** | Seconds (queries pre-indexed literature database) |
| **Install** | Web API (Stanford); local deployment possible but requires literature index |
| **Version** | v2 (2020+) |
| **Complementarity with Exomiser** | Exomiser uses structured phenotype ontologies + cross-species data. AMELIE uses unstructured literature. A gene ranked low by Exomiser (poor animal model phenotype match) but high by AMELIE (recent case reports) is worth investigating. Use Exomiser as primary ranker, AMELIE as second opinion. |

---

## 9. Polygenic Risk Scores

### 9a. SBayesRC

| Field | Detail |
|---|---|
| **What it does** | Bayesian PRS method using GWAS summary statistics + LD reference + functional annotations (regulatory, coding, conserved regions) to estimate per-SNP effect sizes |
| **Why this tool** | Current single-ancestry SOTA for most traits. Functional annotation conditioning improves prediction by 5–15% over annotation-naive methods. |
| **Input** | GWAS summary statistics (COJO format) + LD reference panel + functional annotation matrix |
| **Output** | Per-SNP posterior effect sizes (usable as PRS weights) |
| **Runtime** | ~1–4 hours for genome-wide, depending on annotation complexity |
| **Install** | GCTB software suite (`gctb.pctgark.org`), C++ binary |
| **Version** | v2.05+ (2024) |
| **Limitations** | Single-ancestry. Performance degrades on non-EUR without matched LD reference. Requires pre-computed LD matrices (large: ~50–100 GB per ancestry). |
| **Replaces** | PRS-CS (slightly lower accuracy, no functional annotations), LDpred2-auto (similar tier but less annotation-aware), clumping+thresholding (legacy), MegaPRS, PolyFun+ |

### 9b. PRS-CSx

| Field | Detail |
|---|---|
| **What it does** | Cross-ancestry PRS using coupled continuous shrinkage priors across multiple ancestry-specific GWAS summary statistics |
| **Why this tool** | Best-validated cross-ancestry PRS method. Enables polygenic prediction in non-EUR populations by borrowing information across ancestries. |
| **Input** | GWAS summary statistics from multiple ancestries + ancestry-specific LD reference panels |
| **Output** | Per-SNP posterior effect sizes per ancestry; combined weights via linear combination (optimized on validation set) |
| **Runtime** | ~2–6 hours (runs per-ancestry MCMC in parallel, then combines) |
| **Install** | GitHub (`getian107/PRScsx`), Python |
| **Version** | v1 (2022+) |
| **Limitations** | Requires multi-ancestry GWAS (increasingly available from UKB + BBJ + AoU + FinnGen). Linear combination weights must be tuned on a held-out set from the target ancestry — requires some target-ancestry data. |
| **Complementarity with SBayesRC** | SBayesRC for single-ancestry (EUR) where it's strongest. PRS-CSx when the individual's ancestry is non-EUR or admixed and you need to leverage multi-ancestry GWAS. Run both; report both; flag when they disagree substantially. |

### 9c. plink2 (PRS calculation engine)

| Field | Detail |
|---|---|
| **What it does** | Applies pre-computed PRS weights to individual genotype data to produce per-person risk scores |
| **Why this tool** | De facto standard for PRS calculation. Handles all file formats (VCF, PGEN, BED). Extremely fast. |
| **Input** | Genotype file (VCF/PGEN/BED) + scoring file (PGS Catalog format or custom weights from SBayesRC/PRS-CSx) |
| **Output** | Per-individual PRS values |
| **Key command** | `plink2 --score weights.txt --vcf input.vcf.gz --out prs_output` |
| **Runtime** | Seconds to minutes for individual genomes |
| **Install** | `plink.cog-genomics.org`, Bioconda |

---

## 10. Pharmacogenomics

### 10a. PharmCAT

| Field | Detail |
|---|---|
| **What it does** | End-to-end VCF-to-clinical-recommendation pipeline: extracts pharmacogenomic positions → calls star-allele diplotypes → maps to CPIC/DPWG guidelines → generates prescribing recommendations |
| **Why this tool** | Only tool that goes from VCF to actionable drug recommendations in one step. Covers all 34 CPIC genes. Directly links to clinical guidelines. |
| **Input** | VCF (single-sample, GRCh38) |
| **Output** | JSON/HTML report: diplotypes per gene, metabolizer status, drug-specific recommendations with evidence levels |
| **Key parameters** | `--vcf`, `--output-dir`; v2 adds biobank preprocessor for multi-sample VCFs |
| **Runtime** | ~1–5 minutes per sample |
| **Install** | Java JAR, Docker (`pgkb/pharmcat`) |
| **Version** | v2.13+ (2024) |
| **Limitations** | Requires well-called VCF at pharmacogene positions. Some star alleles require phasing information not present in standard short-read VCFs. CYP2D6 is handled but Cyrius is more accurate for complex CYP2D6 alleles. |
| **Replaces** | Individual star-allele callers (Aldy, Stargazer) for most genes — PharmCAT wraps the full workflow |

### 10b. Cyrius

| Field | Detail |
|---|---|
| **What it does** | Specialized CYP2D6 diplotype caller using paralog-aware (CYP2D7) depth analysis to resolve complex structural variants (deletions, duplications, hybrid alleles) |
| **Why this tool** | CYP2D6 is the hardest pharmacogene (pseudogene CYP2D7, common SVs, hybrid alleles). Cyrius achieves 96.5–99.3% concordance on GeT-RM truth set, outperforming Aldy and Stargazer specifically on SV-containing samples. |
| **Input** | BAM/CRAM (WGS; minimum 20× at CYP2D6 locus) |
| **Output** | CYP2D6 star-allele diplotype |
| **Runtime** | ~1 minute per sample |
| **Install** | GitHub (`Illumina/Cyrius`), Python |
| **Version** | v1.1.1 (used by All of Us) |
| **Complementarity with PharmCAT** | Run Cyrius for CYP2D6 specifically, feed the result into PharmCAT for the full CPIC report. PharmCAT's built-in CYP2D6 calling is adequate for simple alleles but Cyrius handles the hard cases (gene deletions, duplications, CYP2D6/2D7 hybrids). |

---

## 11. HLA Typing

### 11a. HLA*LA

| Field | Detail |
|---|---|
| **What it does** | HLA typing from WGS/WES alignments using graph-based realignment to a population reference graph of HLA alleles |
| **Why this tool** | Best class-II accuracy from DNA sequencing (WGS/WES). Handles the extreme polymorphism of HLA region via graph alignment rather than read mapping to a linear reference. |
| **Input** | BAM/CRAM (WGS or WES with MHC coverage) |
| **Output** | HLA alleles at 2-field (4-digit) or higher resolution for class I (A, B, C) and class II (DRB1, DQB1, DQA1, DPB1, etc.) |
| **Runtime** | ~15–60 minutes per sample |
| **Install** | GitHub (`DiltheyLab/HLA-LA`), requires IMGT/HLA reference graph |
| **Version** | v1.0+ |
| **Limitations** | Slower than OptiType (class-I-only, faster but less accurate for class II). Requires aligned BAM with MHC reads. |
| **Replaces** | OptiType (class I only, older), xHLA, HISAT-genotype, HLA-HD |

### 11b. arcasHLA (RNA-seq)

| Field | Detail |
|---|---|
| **What it does** | HLA typing from RNA-seq data using selective alignment and expectation-maximization |
| **Why this tool** | 99.4% accuracy for class I and II from RNA-seq. Essential when DNA data is unavailable but RNA-seq exists (tumor transcriptomes, GTEx). |
| **Input** | FASTQ or BAM from RNA-seq |
| **Output** | HLA alleles at 2-field resolution |
| **Runtime** | ~5–15 minutes per sample |
| **Install** | GitHub (`RabadanLab/arcasHLA`), Python + Bioconda |
| **When to use** | When you have RNA-seq but not WGS/WES, or when you need expression-informed HLA typing (e.g., checking which HLA alleles are expressed in tumor for neoantigen prediction). For DNA-only data, use HLA\*LA. |

---

## 12. Ancestry Inference

### 12a. flashPCA2 + ADMIXTURE

| Field | Detail |
|---|---|
| **What it does** | flashPCA2: fast PCA on genotype matrices for population structure visualization and stratification. ADMIXTURE: maximum-likelihood estimation of global ancestry proportions. |
| **Why these tools** | PCA is the universal first step for population structure (required for PRS stratification, confounding control). ADMIXTURE provides interpretable ancestry fractions (e.g., 60% EUR, 30% AFR, 10% NAT). |
| **Input** | PLINK BED/BIM/FAM + reference panel (1000 Genomes, HGDP) |
| **Output** | PC coordinates (flashPCA2); ancestry proportions per K clusters (ADMIXTURE) |
| **Runtime** | flashPCA2: minutes for biobank-scale. ADMIXTURE: ~hours for K=5–7 at N>10K. |
| **Install** | flashPCA2: GitHub, standalone binary. ADMIXTURE: download from genetics.ucla.edu. |
| **Limitations** | PCA and ADMIXTURE give global ancestry only. For local ancestry (which chromosomal segments come from which ancestral population), use RFMix v2 or FLARE — but these are substantially more complex and only needed for cross-ancestry PRS adjustment or admixture mapping. |

---

## 13. Somatic Variant Calling

### 13a. Mutect2 (GATK4)

| Field | Detail |
|---|---|
| **What it does** | Somatic SNV/indel calling from tumor-normal or tumor-only short-read sequencing |
| **Why this tool** | GATK4 standard; most validated somatic caller; extensive filtering pipeline (FilterMutectCalls, orientation bias, contamination estimation). Community standard for TCGA/PCAWG reanalysis. |
| **Input** | Tumor BAM + matched normal BAM (or Panel of Normals for tumor-only) + reference |
| **Output** | VCF with somatic variants, filtering annotations |
| **Key parameters** | `--panel-of-normals`, `--germline-resource` (gnomAD), `--f1r2-tar-gz` (orientation bias) |
| **Runtime** | ~4–8 hours for WGS tumor-normal on 16 cores |
| **Install** | GATK4 (`broadinstitute/gatk`), Java/Docker |
| **Version** | GATK 4.5+ |
| **Replaces** | Strelka2 (discontinued), VarDict, VarScan2, MuTect (v1, legacy) |

### 13b. DeepSomatic (Long-read somatic)

| Field | Detail |
|---|---|
| **What it does** | Deep learning somatic caller for long-read (PacBio HiFi, ONT) tumor-normal data, extending DeepVariant's pileup image approach |
| **Why this tool** | Beats Strelka2 and ClairS on SEQC2 benchmark for long-read somatic calling. Google-maintained, active development. |
| **Input** | Tumor BAM + normal BAM (long-read aligned) + reference |
| **Output** | VCF |
| **Runtime** | ~2–4 hours for 30× long-read WGS on GPU |
| **Install** | Docker (`google/deepsomatic`), or build from source |
| **Version** | v1 (2024) |
| **Limitations** | Requires tumor-normal pair. For tumor-only long-read, use ClairS-TO (HKU-BAL, 2025). |

---

## 14. Mutational Signatures

### 14a. SigProfiler

| Field | Detail |
|---|---|
| **What it does** | De novo extraction and assignment of mutational signatures (SBS, DBS, ID) from somatic mutation catalogs |
| **Why this tool** | Reference standard. Curated the COSMIC v3.4 signature catalog. Three components: SigProfilerMatrixGenerator (catalog→matrix), SigProfilerExtractor (de novo NMF), SigProfilerAssignment (fit known sigs to sample). |
| **Input** | VCF or MAF of somatic mutations + reference genome |
| **Output** | Signature activity matrix (samples × signatures); per-sample signature attributions; plots |
| **Key parameters** | Minimum mutations per sample (~50 for reliable SBS extraction); choice of SBS-96/192/288/1536 context |
| **Runtime** | Minutes for assignment; hours for de novo extraction on large cohorts |
| **Install** | `pip install SigProfilerMatrixGenerator SigProfilerExtractor SigProfilerAssignment` |
| **Version** | 2023+ |
| **Clinically relevant signatures** | SBS3 (HRD → PARP inhibitor eligibility), SBS6/14/15/20/21/26/44 (MMR deficiency → immunotherapy), SBS4 (tobacco), SBS7a/b (UV), SBS2/13 (APOBEC), SBS31/35 (platinum exposure) |
| **Replaces** | deconstructSigs (biased against low-activity sigs), MutationalPatterns (lighter but less rigorous), SignatureAnalyzer (Bayesian NMF, orthogonal but less adopted), MuSiCal |

---

## 15. Somatic Driver & Biomarker Annotation

### 15a. OncoKB + CIViC Lookup

| Field | Detail |
|---|---|
| **What it does** | Maps somatic variants to therapeutic actionability levels |
| **Why these tools** | OncoKB: FDA-recognized (first such designation, 2021), Levels 1–4 + R1/R2. CIViC: open-source crowd-curated, 5 evidence tiers. Together they provide the most comprehensive actionability annotation. |
| **Input** | Gene + variant (HGVS or protein change) |
| **Output** | Actionability level, associated drugs, evidence summaries, clinical trial references |
| **Access** | OncoKB: REST API (free for research, license for commercial). CIViC: GraphQL API, fully open. |
| **Complementarity** | OncoKB is curated by MSK with strict evidence thresholds — fewer entries, higher confidence. CIViC is community-curated — broader coverage, variable evidence quality. Query both; present OncoKB level as primary, CIViC as supplementary. |

### 15b. MSIsensor-pro

| Field | Detail |
|---|---|
| **What it does** | Microsatellite instability detection from tumor-normal or tumor-only NGS data |
| **Why this tool** | Fast, accurate (97.7% concordance with PCR-based MSI), works with panels and WES/WGS. |
| **Input** | Tumor BAM (+ optional normal BAM) + microsatellite locus list |
| **Output** | MSI score (% unstable loci); MSI-H / MSS classification |
| **Threshold** | ≥20% unstable loci (tumor-normal) or model-based (tumor-only) |
| **Clinical relevance** | MSI-H → pembrolizumab (tumor-agnostic, 2017), dostarlimab (endometrial) |
| **Install** | GitHub (`xjtu-omics/msisensor-pro`), C++ |
| **Replaces** | MSIsensor (v1), MANTIS (slower) |

---

## 16. Immune Deconvolution

### 16a. CIBERSORTx

| Field | Detail |
|---|---|
| **What it does** | Estimates immune cell type fractions from bulk RNA-seq or microarray using constrained support vector regression against a signature matrix, with batch correction and single-cell imputation |
| **Why this tool** | Gives absolute fractions (not just relative). Batch correction mode handles technical variation across datasets. S-mode imputes cell-type-specific gene expression from bulk data. |
| **Input** | Bulk RNA-seq expression matrix (TPM/FPKM) or microarray |
| **Output** | Per-sample cell type fractions (22 immune cell types with LM22 signature; custom signatures supported) |
| **Runtime** | Minutes per sample |
| **Access** | Web portal (cibersortx.stanford.edu) + Docker for local runs; free for academic use |
| **Limitations** | LM22 reference is blood-derived — may not capture tissue-resident immune populations well. Deconvolution is fundamentally ill-posed for rare cell types at low fractions. |
| **Replaces** | xCell (relative scores only), MCP-counter (relative), TIMER2.0 (web-only) |

---

## 17. cfDNA Analysis

### 17a. ichorCNA

| Field | Detail |
|---|---|
| **What it does** | Estimates tumor fraction from ultra-low-pass WGS (ULP-WGS, ≥0.1×) of cfDNA using copy-number signal |
| **Why this tool** | Only widely-validated cfDNA tumor fraction estimator. LOD ~3% tumor fraction. Enables triage: if TF >3%, proceed to deeper analysis; if <3%, standard targeted panels may miss signal. |
| **Input** | BAM from ULP-WGS of cfDNA |
| **Output** | Tumor fraction estimate, genome-wide copy-number profile |
| **Runtime** | ~10–30 minutes per sample |
| **Install** | GitHub (`broadinstitute/ichorCNA`), R + HMMcopy |
| **Version** | v0.3+ |
| **Limitations** | LOD ~3% — misses low-TF samples (early-stage cancer, MRD). Not a variant caller. |

### 17b. UXM (Loyfer atlas deconvolution)

| Field | Detail |
|---|---|
| **What it does** | Fragment-level methylation deconvolution using the Loyfer 2023 atlas of ~100 sorted cell types at 30× WGBS. Identifies cell-of-origin from cfDNA methylation patterns. |
| **Why this tool** | Detection limit 0.03–0.1% — orders of magnitude better than ichorCNA for cell-type identification. The reference atlas (Nature 613:355) is the most comprehensive sorted-cell-type methylation reference available. |
| **Input** | WGBS or targeted bisulfite sequencing BAM of cfDNA |
| **Output** | Cell-type fractions (liver, colon, lung, immune subtypes, etc.) |
| **Runtime** | ~30 minutes per sample |
| **Install** | GitHub (`nloyfer/wgbs_tools`), Python |
| **Applications** | Cancer cell-of-origin (which organ?), transplant rejection monitoring (donor cell fraction), tissue injury detection |
| **Limitations** | Requires bisulfite sequencing (not standard WGS). Atlas coverage gaps for some rare cell types. |

### 17c. Griffin

| Field | Detail |
|---|---|
| **What it does** | Extracts nucleosome footprint profiles at transcription factor binding sites from cfDNA fragment data, with GC bias correction |
| **Why this tool** | Works at ULP-WGS (0.1× coverage). Captures transcription factor activity from cfDNA fragment positioning — a signal orthogonal to both mutations and methylation. |
| **Input** | BAM from cfDNA WGS (any depth, including ULP) + TFBS site lists |
| **Output** | Composite nucleosome profiles per TFBS cluster; tumor-type classification features |
| **Runtime** | ~1–2 hours per sample |
| **Install** | GitHub (`adoebley/Griffin`) |
| **Version** | v1 (2022) |
| **Complementarity** | ichorCNA (copy number / tumor fraction) + UXM (methylation / cell-of-origin) + Griffin (nucleosome / TF activity) = three orthogonal cfDNA signal layers with zero redundancy. |

---

## 18. Epigenetic Clocks & Methylation

### 18a. Biolearn

| Field | Detail |
|---|---|
| **What it does** | Harmonized implementation of major epigenetic clocks and methylation-based biomarkers: Horvath, Hannum, PhenoAge, GrimAge2, DunedinPACE, PC-Clocks, CausAge, DamAge, AdaptAge |
| **Why this tool** | Single interface for all validated clocks with cross-platform harmonization (450K → EPIC v1 → EPIC v2 → MSA probe remapping). Eliminates the need to maintain separate R scripts per clock. |
| **Input** | Methylation beta-value matrix (from Illumina array or WGBS) + sample metadata (age, sex for some clocks) |
| **Output** | Per-sample clock estimates: chronological age prediction (Horvath/Hannum), phenotypic age (PhenoAge), mortality risk (GrimAge2), aging pace (DunedinPACE), causal age components (CausAge/DamAge/AdaptAge) |
| **Key clocks** | |

| Clock | CpGs | What it measures | Key validation |
|---|---|---|---|
| Horvath 2013 | 353 | Pan-tissue chronological age | MAE ~3.6 years |
| PhenoAge | 513 | Phenotypic age (morbidity-weighted) | Predicts disease onset, healthspan |
| GrimAge2 | DNAm surrogates of plasma proteins + smoking pack-years | Mortality risk | HR/SD = 1.3–1.6; P = 3.6×10⁻¹⁶⁷ |
| DunedinPACE | 173 | Pace of biological aging (years of aging per calendar year) | Validated in CALERIE caloric restriction trial |
| PC-Clocks | Principal components of CpGs | Noise-reduced versions of above | ICC >0.97, replicate disagreement <1.5 years |
| CausAge/DamAge | Epigenome-wide MR-selected CpGs | Causally interpretable age acceleration | Separates damage vs adaptation |

| **Runtime** | Minutes per sample batch |
| **Install** | `pip install biolearn`, Python |
| **Limitations** | Requires Illumina methylation array data (450K/EPIC/EPIC v2) or high-coverage WGBS. Not applicable to targeted panels or low-coverage bisulfite data. Array platform transitions require probe remapping — Biolearn handles this but accuracy degrades slightly across platforms. |

---

## 19. Report Generation

### 19a. Claude Synthesis Layer

This is not an external tool but the integration layer where Claude combines outputs from all upstream modules into a coherent interpretation.

| Component | What Claude does |
|---|---|
| **Variant triage** | Prioritizes variants by combining VEP annotation + AlphaMissense/GPN-MSA/SpliceAI scores + ClinVar status + gnomAD frequency + Exomiser/AMELIE phenotype match |
| **ACMG classification** | Applies ACMG/AMP 28-code framework to candidate variants using computed evidence (PM2 from gnomAD, PP3 from AlphaMissense, PS3 from functional data if available, etc.) |
| **PRS contextualization** | Reports per-trait PRS percentiles with ancestry-matched reference distributions; flags ancestry mismatch when PRS-CSx and SBayesRC disagree |
| **PGx summary** | Translates PharmCAT/Cyrius diplotypes into plain-language drug recommendations with CPIC evidence levels |
| **Uncertainty communication** | Explicitly states confidence intervals, heritability ceilings, ancestry limitations, and model failure modes per prediction |
| **Literature integration** | Uses AMELIE and direct PubMed/bioRxiv search to contextualize novel or VUS-classified variants |

**Design principle for the synthesis layer:** Claude should never present a prediction without its uncertainty bound and the strongest known limitation. A PRS for height in EUR ancestry should state the R² (~0.35), the ancestry it was derived in, the expected portability loss for non-EUR, and that environment explains >50% of variance. A VUS should be labeled as a VUS with the specific evidence codes that are missing for P/LP classification.

---

## Dependency Graph

```
                    ┌─────────────┐
                    │ FASTQ/BAM   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │DeepVariant│ │ Mutect2  │ │ichorCNA  │
        │ / DRAGEN │ │/DeepSomat│ │UXM/Griffin│
        │(germline)│ │(somatic) │ │ (cfDNA)  │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
             ▼             ▼             │
        ┌──────────┐ ┌──────────┐        │
        │   VEP    │ │SigProfile│        │
        │(annotate)│ │MSIsensor │        │
        └────┬─────┘ │OncoKB   │        │
             │       └────┬─────┘        │
    ┌────────┼────────┐   │              │
    ▼        ▼        ▼   │              │
┌───────┐┌──────┐┌──────┐ │              │
│AlphaM ││GPN-  ││Splice│ │              │
│issense││MSA   ││AI +  │ │              │
│+ EVE  ││ChromB││Pango-│ │              │
│+SaProt││PNet  ││lin   │ │              │
└───┬───┘└──┬───┘└──┬───┘ │              │
    │       │       │     │              │
    ▼       ▼       ▼     ▼              ▼
┌─────────────────────────────────────────────┐
│              Integration Layer              │
│  Exomiser + AMELIE (Mendelian)              │
│  SBayesRC + PRS-CSx + plink2 (PRS)         │
│  PharmCAT + Cyrius (PGx)                   │
│  HLA*LA / arcasHLA (HLA)                   │
│  flashPCA2 + ADMIXTURE (Ancestry)          │
│  CIBERSORTx (Immune)                       │
│  Biolearn (Epigenetic clocks)              │
│  AnnotSV + ClassifyCNV (SVs)               │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ Claude Report  │
          │ Generation     │
          └────────────────┘
```

---

## Installation & Environment Summary

| Tool | Language | GPU? | Disk (data) | Docker? |
|---|---|---|---|---|
| DeepVariant | C++/Python | Yes (optional) | ~30 GB (model) | Yes |
| DRAGEN | C++/FPGA | FPGA | ~30 GB | No (hardware) |
| VEP | Perl | No | ~18 GB (cache) | Yes |
| AlphaMissense | Python/JAX | Yes (for de novo) | ~3.6 GB (pre-computed) | No |
| EVE | Python/PyTorch | Yes | ~2 GB (pre-computed) | No |
| SaProt | Python/PyTorch | Yes | ~5 GB (weights) | No |
| GPN-MSA | Python/PyTorch | Optional | ~10 GB (MSA + scores) | No |
| ChromBPNet | Python/TF | Yes | ~1 GB/model | No |
| AlphaGenome | Python/JAX | TPU/GPU | ~20 GB | No |
| Borzoi/Flashzoi | Python/PyTorch | Yes | ~15 GB | No |
| EVEE | Python/Web | Yes (de novo) | ~3.6 GB (pre-computed ClinVar) | No |
| SpliceAI | Python/TF | Yes (optional) | ~60 GB (pre-computed) | No |
| Pangolin | Python/PyTorch | Yes | ~2 GB | No |
| AnnotSV | Tcl | No | ~5 GB | No |
| ClassifyCNV | Python | No | ~1 GB | No |
| Exomiser | Java | No | ~80 GB (data) | Yes |
| AMELIE | Web API | No | N/A | No |
| SBayesRC | C++ | No | ~50–100 GB (LD) | No |
| PRS-CSx | Python | No | ~50 GB (LD) | No |
| plink2 | C++ | No | Minimal | No |
| PharmCAT | Java | No | ~500 MB | Yes |
| Cyrius | Python | No | Minimal | No |
| HLA*LA | C++/Perl | No | ~30 GB (graph) | Yes |
| arcasHLA | Python | No | ~1 GB | No |
| flashPCA2 | C++ | No | Minimal | No |
| ADMIXTURE | C++ | No | Minimal | No |
| Mutect2 | Java | No | ~5 GB | Yes (GATK) |
| DeepSomatic | C++/Python | Yes | ~30 GB | Yes |
| SigProfiler | Python | No | ~5 GB | No |
| MSIsensor-pro | C++ | No | Minimal | No |
| CIBERSORTx | R/Docker | No | ~1 GB | Yes |
| ichorCNA | R | No | ~1 GB | No |
| UXM | Python | No | ~50 GB (atlas) | No |
| Griffin | Python | No | ~5 GB | No |
| Biolearn | Python | No | Minimal | No |

**Total disk footprint (all pre-computed data + references):** ~500–600 GB

**GPU requirement:** AlphaMissense (de novo), SaProt, ChromBPNet, AlphaGenome/Borzoi, DeepVariant (optional), DeepSomatic, SpliceAI (de novo). Most tools can run CPU-only with pre-computed scores.

---

## What Is NOT Included (and Why)

| Excluded Tool | Reason |
|---|---|
| CADD, REVEL, BayesDel, PolyPhen-2, SIFT | ClinVar training circularity; superseded by AlphaMissense |
| Enformer | Superseded by AlphaGenome/Borzoi on all benchmarks |
| Nucleotide Transformer, HyenaDNA, DNABERT-2, Caduceus | DNA LMs fail to beat GPN-MSA/ChromBPNet on VEP tasks |
| Strelka2 | Discontinued; replaced by Mutect2 |
| deconstructSigs | Biased against low-activity signatures; replaced by SigProfiler |
| Geneformer, scGPT, scFoundation | Do not beat linear baselines on annotation/integration/perturbation |
| LIRICAL, Phen2Gene | Weaker than Exomiser for Mendelian prioritization |
| PRS-CS (vanilla) | Replaced by SBayesRC (better with annotations) |
| OptiType | Class I only; replaced by HLA*LA |
| xCell, MCP-counter, TIMER2.0 | Relative scores only; replaced by CIBERSORTx |
| MANTIS | Replaced by MSIsensor-pro (faster, more accurate) |
