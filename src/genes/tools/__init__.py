"""Importing this package triggers ToolSpec registration for every tool.

The scheduler relies on `genes.tools` being importable to populate the
spec registry. Add new tools here when you add them to the package.
"""

from genes.tools import (  # noqa: F401
    alphamissense,
    ancestry,
    annotsv,
    biolearn,
    cfdna,
    classifycnv,
    cyrius,
    deepvariant,
    evee,
    exomiser,
    gpn_msa,
    hla,
    msisensor,
    mutect2,
    oncokb_civic,
    pharmcat,
    prs,
    sigprofiler,
    spliceai,
    traits,
    vep,
)
