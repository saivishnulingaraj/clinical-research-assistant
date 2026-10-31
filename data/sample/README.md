# Synthetic sample corpus

**⚠️ ALL DOCUMENTS IN THIS DIRECTORY ARE SYNTHETIC AND FICTIONAL.**

These ~15 clinical-trial summaries were written by hand for this portfolio
project. They do **not** describe real trials, real interventions, real people,
or real results, and must **never** be used for any clinical, medical, or
regulatory purpose. Trial IDs (e.g. `SYN-0001`) are invented and share no
namespace with real registries such as ClinicalTrials.gov.

Each file follows a consistent section structure so the section-aware chunker
has headings to split on:

- `# <Title>`
- `## Summary`
- `## Condition`
- `## Intervention`
- `## Eligibility Criteria`
- `## Primary Endpoints`
- `## Secondary Endpoints`
- `## Results`
- `## Status`

Front-matter at the top of each file carries metadata (`doc_id`, `title`,
`source_url`). `source_url` points at a fictional `https://synthetic.example/...`
host to make clear the data is not real.
