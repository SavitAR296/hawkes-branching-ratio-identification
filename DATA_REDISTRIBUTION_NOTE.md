# Data provenance and redistribution note

## Euro-Bund futures event snapshot (`data/bund.npz`)

- Source: the public Euro-Bund futures tick dataset distributed with the research workspace of Bacry, Jaisson and Muzy (X-DataInitiative/tick-datasets; see Bacry, Jaisson, Muzy, *Quantitative Finance* 16(8), 2016, 1179–1201).
- Coverage: 20 trading sessions in April 2014, 859,467 events with microsecond timestamps (pooled up/down mid-price changes and buyer/seller-initiated trades).

## Redistribution check before making the repository public

Third-party market data are not owned by the author. Before making this repository public:

1. Re-read the licence/terms of use attached to the original tick-datasets release and confirm whether redistribution of the raw snapshot is permitted.
2. If redistribution is **not** permitted, replace `data/bund.npz` in the public repository with:
   - a `data/README.md` giving the exact source URL, download steps and checksum of the required snapshot;
   - a small loader script that verifies the downloaded file against the recorded checksum;
   - keep all frozen **derived** outputs in `tables/` and `tables_csf/` (processed results, not the raw database), which is what the manuscript tables and figures consume.
3. The journal submission package may still contain the snapshot as a confidential reproducibility archive for reviewers; that is separate from public redistribution.

Do not push data whose licence forbids redistribution to a public repository.
