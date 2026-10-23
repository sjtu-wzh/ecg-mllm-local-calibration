| Dataset | Origin | Relation to PULSE training | Task / prompt | Labels | Calibration pool | Test |
|---|---|---|---|---|---|---|
| CPSC 2018 (ECGBench) | China (multi-centre) | out-of-domain | multi-label classification, official ECGBench prompt | 9 | 600 | 1461 |
| CSN (ECGBench, multi-label version) | China (Shaoxing/Ningbo) | out-of-domain | multi-label, CPSC-style prompt (this work) | 11 | 600 | 979 |
| G12EC (ECGBench, multi-label version) | USA (Georgia) | out-of-domain | multi-label, CPSC-style prompt (this work) | 24 | 600 | 1396 |
| SPH (rendered images; pre-registered) | China (Shandong Provincial Hospital) | out-of-domain, new site | multi-label, CPSC-style prompt (this work) | 12 | 600 | 2312 |
| CODE-15% (ECGBench) | Brazil | in-domain (training source) | binary + multi-label, official prompts | 1+6 | 600 | 800 |
| PTB-XL fold 10 (ECGBench images) | Germany | in-domain (training source) | CPSC 9-label prompt (domain analysis) | 8 | 600 | ≈757 (20 patient-grouped splits) |

Pool = records from which n labelled ECGs are drawn; the test split is fixed and never used for fitting.
