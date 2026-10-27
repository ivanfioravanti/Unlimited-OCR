| pages | output tokens | ended on | `<PAGE>` out (must = N) | per-page similarity vs PDF | page-3 first sentence (verbatim, truncated) | tok/s |
|---|---|---|---|---|---|---|
| 3 | 1669 | EOS | 3 | 0.926 1.000 1.000 | Document parsing has undergone a major paradigm shift, transitioning f | 38.95 |
| 4 | 2623 | EOS | 4 | 0.926 1.000 0.998 0.979 | Document parsing is having undergone a major paradigm shift, transitio | 38.15 |
| 5 | 3231 | EOS | 5 | 0.926 1.000 0.998 0.979 0.925 | Document parsing is having undergone a major paradigm shift, transitio | 36.5 |
| 6 | 3934 | EOS | 6 | 0.926 1.000 0.998 0.979 0.925 0.898 | Document parsing is having undergone a major paradigm shift, transitio | 35.08 |

SGLang (wheel from `wheel/`, README flags, CUDA graphs, base mode, greedy, 1 run each, same JPEGs; n-gram processor on / off):

| pages | n-gram | output tokens | ended on | `<PAGE>` out | per-page similarity vs PDF | page-3 first sentence | tok/s (end-to-end) | decode tok/s | prefill s |
|---|---|---|---|---|---|---|---|---|---|
| 3 | on | 1670 | EOS | 3 | 0.926 1.000 1.000 | Document parsing has undergone a major paradigm shift, transi | 128.83 | 134.17 | 0.52 |
| 3 | off | 1670 | EOS | 3 | 0.926 1.000 1.000 | Document parsing has undergone a major paradigm shift, transi | 140.37 | 144.78 | 0.36 |
| 4 | on | 2624 | EOS | 4 | 0.926 1.000 0.998 0.979 | Document parsing is having undergone a major paradigm shift,  | 124.67 | 133.79 | 1.43 |
| 4 | off | 2624 | EOS | 4 | 0.926 1.000 0.998 0.979 | Document parsing is having undergone a major paradigm shift,  | 139.16 | 144.63 | 0.71 |
| 5 | on | 3230 | EOS | 5 | 0.926 1.000 0.998 0.979 0.925 | Document parsing is having undergone a major paradigm shift,  | 128.68 | 133.36 | 0.88 |
| 5 | off | 3230 | EOS | 5 | 0.926 1.000 0.998 0.979 0.925 | Document parsing is having undergone a major paradigm shift,  | 139.87 | 144.49 | 0.74 |
| 6 | on | 3933 | EOS | 6 | 0.926 1.000 0.998 0.979 0.925 0.900 | Document parsing is having undergone a major paradigm shift,  | 129.22 | 133.17 | 0.9 |
| 6 | off | 3933 | EOS | 6 | 0.926 1.000 0.998 0.979 0.925 0.900 | Document parsing is having undergone a major paradigm shift,  | 140.94 | 144.48 | 0.68 |

Answers: the GB10 paths (HF and SGLang) never looped in 3-6 pages (all ended on EOS, `<PAGE>` count = N). The page-3 glitch ("is having undergone" instead of "has undergone") is present at 4, 5 and 6 pages and absent at 3 pages, on both HF and SGLang, with and without the n-gram processor; it is deterministic, not more frequent beyond 4 pages (it is the same wording at 4, 5 and 6).
