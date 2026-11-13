# CoSHE-Eval subset scores

RTF = processing time / audio time (lower is faster; <1 is faster than real time).

| model | clips | WER % | CER % | WER mixed | WER hindi | WER english | RTF |
|---|---|---|---|---|---|---|---|
| maya-cpu | 20 | 18.4 | 10.0 | 19.1 | 7.3 | 19.0 | 0.75 |
| tara-cuda | 30 | 20.0 | 13.2 | 20.5 | 12.8 | 20.1 | 0.07 |
| tara-ov-gpu | 30 | 20.1 | 13.2 | 20.6 | 12.8 | 20.1 | 0.52 |
| maya-cuda | 30 | 22.8 | 13.1 | 22.4 | 11.6 | 27.2 | 0.08 |
| maya-ov-gpu | 30 | 28.6 | 17.3 | 27.1 | 15.7 | 36.8 | 0.69 |
| maya-ov-gpu-f32 | 30 | 28.7 | 17.4 | 27.2 | 15.7 | 37.2 | 1.62 |
| minutes-superwhisper | 30 | 44.6 | 33.2 | 52.2 | 49.2 | 16.2 | 0.05 |
