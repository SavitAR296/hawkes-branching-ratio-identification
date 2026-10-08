# Data provenance and redistribution note

## Euro-Bund futures event snapshot (`data/bund.npz`)

- Source: the public Euro-Bund futures tick dataset distributed with the research workspace of Bacry, Jaisson and Muzy (X-DataInitiative/tick-datasets; see Bacry, Jaisson, Muzy, *Quantitative Finance* 16(8), 2016, 1179–1201).
- Coverage: 20 trading sessions in April 2014, 859,467 events with microsecond timestamps (pooled up/down mid-price changes and buyer/seller-initiated trades).

## Licence and redistribution (resolved)

The snapshot is the file `hawkes/bund/bund.npz` of X-DataInitiative/tick-datasets, released under the BSD 3-Clause License (copyright 2016, X-DataInitiative). Redistribution in source or binary form is permitted provided the copyright notice, the list of conditions and the disclaimer are retained; they are included verbatim in `data/LICENSE_tick-datasets.txt`. The archived file is byte-identical to the upstream file (SHA-256 `fa1d4679fff7f6e2ee7819cbc461df14056bf1786ad0c98cf9e9019feff4bd8c`).

The per-day event counts in the upstream description page (7,009 / 6,998 / 257,678 / 261,424) differ from the counts in the distributed file (means 5,965 / 5,967 / 14,403 / 16,638 for up / down / buy / sell); the analysis uses the file, and the per-session counts are listed in Appendix A of the revised manuscript.
