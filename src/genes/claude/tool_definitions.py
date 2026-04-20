"""Claude tool-use schemas for the genome-to-phenotype toolkit.

Each tool definition maps to an orchestrator pipeline. These schemas are
passed to the Claude API's `tools` parameter to enable structured tool use.
"""

from __future__ import annotations

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "analyze_germline_vcf",
        "description": (
            "Analyze a germline VCF file. Runs VEP annotation followed by parallel "
            "variant effect prediction (AlphaMissense, SpliceAI, GPN-MSA), "
            "pharmacogenomics (PharmCAT), polygenic risk scores, and ancestry inference. "
            "Optionally runs Exomiser for Mendelian disease prioritization if HPO terms "
            "are provided."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Unique identifier for this analysis run.",
                },
                "vcf_path": {
                    "type": "string",
                    "description": "Path to the germline VCF file (GRCh38).",
                },
                "hpo_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "HPO term IDs (e.g., ['HP:0001250', 'HP:0001263']) for "
                        "Mendelian disease prioritization via Exomiser. Optional."
                    ),
                },
                "prs_traits": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Trait names for polygenic risk score calculation "
                        "(e.g., ['height', 'CAD', 'T2D']). Optional."
                    ),
                },
                "reported_ancestry": {
                    "type": "string",
                    "description": (
                        "Self-reported ancestry for PRS calibration and ancestry "
                        "mismatch detection. Optional."
                    ),
                },
            },
            "required": ["run_id", "vcf_path"],
        },
    },
    {
        "name": "analyze_germline_bam",
        "description": (
            "Analyze a germline BAM file end-to-end. Calls variants with DeepVariant, "
            "then runs the full germline VCF pipeline (VEP, AlphaMissense, SpliceAI, "
            "GPN-MSA, PharmCAT, PRS, ancestry). Also runs BAM-specific tools: Cyrius "
            "(CYP2D6 star allele calling) and HLA typing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Unique identifier for this analysis run.",
                },
                "bam_path": {
                    "type": "string",
                    "description": "Path to the aligned BAM file (indexed, GRCh38).",
                },
                "sequencing_platform": {
                    "type": "string",
                    "enum": ["illumina", "pacbio", "ont"],
                    "description": "Sequencing platform used. Default: illumina.",
                },
                "hpo_terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "HPO term IDs for Exomiser prioritization. Optional.",
                },
                "prs_traits": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Traits for polygenic risk score calculation. Optional.",
                },
                "reported_ancestry": {
                    "type": "string",
                    "description": "Self-reported ancestry. Optional.",
                },
            },
            "required": ["run_id", "bam_path"],
        },
    },
    {
        "name": "analyze_tumor",
        "description": (
            "Analyze a tumor sample (tumor-normal BAM pair or tumor-only BAM). "
            "Calls somatic variants with Mutect2, annotates with VEP, then runs "
            "OncoKB/CIViC actionability lookup, mutational signature analysis "
            "(SigProfiler), and microsatellite instability scoring (MSIsensor)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Unique identifier for this analysis run.",
                },
                "tumor_bam_path": {
                    "type": "string",
                    "description": "Path to the tumor BAM file (indexed, GRCh38).",
                },
                "normal_bam_path": {
                    "type": "string",
                    "description": (
                        "Path to the matched normal BAM file. Optional; "
                        "tumor-only mode if omitted."
                    ),
                },
                "tumor_type": {
                    "type": "string",
                    "description": (
                        "Tumor type for OncoKB context and signature analysis "
                        "(e.g., 'LUAD', 'BRCA', 'COAD'). Optional."
                    ),
                },
            },
            "required": ["run_id", "tumor_bam_path"],
        },
    },
    {
        "name": "analyze_cfdna",
        "description": (
            "Analyze cell-free DNA (cfDNA) from a liquid biopsy BAM. Runs ichorCNA "
            "for tumor fraction estimation and Griffin for nucleosome footprinting "
            "in parallel. If a bisulfite-sequenced BAM is also provided, runs UXM "
            "cell-of-origin deconvolution using the Loyfer methylation atlas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Unique identifier for this analysis run.",
                },
                "cfdna_bam_path": {
                    "type": "string",
                    "description": "Path to the cfDNA BAM file (indexed).",
                },
                "bisulfite_bam_path": {
                    "type": "string",
                    "description": (
                        "Path to a bisulfite-sequenced cfDNA BAM for UXM "
                        "cell-of-origin analysis. Optional."
                    ),
                },
            },
            "required": ["run_id", "cfdna_bam_path"],
        },
    },
    {
        "name": "analyze_methylation",
        "description": (
            "Compute epigenetic aging clocks from a DNA methylation beta-value matrix "
            "(e.g., Illumina 450K or EPIC array). Runs Biolearn to calculate Horvath, "
            "PhenoAge, GrimAge2, DunedinPACE, and PC-Clocks. Returns predicted "
            "biological age and age acceleration for each clock."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Unique identifier for this analysis run.",
                },
                "methylation_path": {
                    "type": "string",
                    "description": (
                        "Path to methylation beta-value matrix (CSV/TSV). "
                        "Rows = CpG probes, columns = samples."
                    ),
                },
                "chronological_age": {
                    "type": "number",
                    "description": "Known chronological age for age-acceleration calculation. Optional.",
                },
                "sex": {
                    "type": "string",
                    "enum": ["M", "F"],
                    "description": "Biological sex for sex-adjusted clocks. Optional.",
                },
            },
            "required": ["run_id", "methylation_path"],
        },
    },
    {
        "name": "lookup_variant",
        "description": (
            "Look up pre-computed effect scores for one or more specific variants. "
            "Returns AlphaMissense pathogenicity, SpliceAI splice impact, GPN-MSA "
            "conservation score, ClinVar classification, and any available EVE/EVEE "
            "scores. Use this for quick single-variant queries without running a "
            "full pipeline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Unique identifier for this lookup.",
                },
                "variants": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "chrom": {"type": "string", "description": "Chromosome (e.g., 'chr1')."},
                            "pos": {"type": "integer", "description": "Genomic position (1-based)."},
                            "ref": {"type": "string", "description": "Reference allele."},
                            "alt": {"type": "string", "description": "Alternate allele."},
                        },
                        "required": ["chrom", "pos", "ref", "alt"],
                    },
                    "description": "List of variants to look up.",
                },
            },
            "required": ["run_id", "variants"],
        },
    },
]
