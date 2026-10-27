| Dataset | Abbreviation | Label (as in prompt) | Source of gold label | Positives in pool | Positives in test | Mapping note |
|---|---|---|---|---|---|---|
| CPSC 2018 | NORM | Normal ECG | ECGBench CPSC label | 77 | 187 | PTB-XL SCP: NORM |
| CPSC 2018 | AF | Atrial fibrillation | ECGBench CPSC label | 103 | 255 | PTB-XL SCP: AFIB |
| CPSC 2018 | I-AVB | First-degree atrioventricular block | ECGBench CPSC label | 63 | 154 | PTB-XL SCP: 1AVB |
| CPSC 2018 | LBBB | Left bundle branch block | ECGBench CPSC label | 19 | 43 | PTB-XL SCP: CLBBB |
| CPSC 2018 | RBBB | Right bundle branch block | ECGBench CPSC label | 165 | 402 | PTB-XL SCP: CRBBB |
| CPSC 2018 | PAC | Premature atrial contraction | ECGBench CPSC label | 53 | 126 | PTB-XL SCP: PAC |
| CPSC 2018 | PVC | Premature ventricular contraction | ECGBench CPSC label | 67 | 164 | PTB-XL SCP: PVC |
| CPSC 2018 | STD | ST-segment depression | ECGBench CPSC label | 78 | 193 | PTB-XL SCP: STD_ |
| CPSC 2018 | STE | ST-segment elevated | ECGBench CPSC label | 17 | 42 | PTB-XL SCP: STE_ + INJAS/INJAL/INJIN/INJLA/INJIL (not in primary 8-label domain analysis) |
| CSN | SB | sinus bradycardia | ECGBench MCQ gold option | 247 | 378 |  |
| CSN | NSR | sinus rhythm | ECGBench MCQ gold option | 133 | 239 |  |
| CSN | STach | sinus tachycardia | ECGBench MCQ gold option | 99 | 166 |  |
| CSN | TAb | t wave abnormal | ECGBench MCQ gold option | 83 | 138 |  |
| CSN | AFL | atrial flutter | ECGBench MCQ gold option | 60 | 112 |  |
| CSN | SA | sinus arrhythmia | ECGBench MCQ gold option | 48 | 58 |  |
| CSN | AF | atrial fibrillation | ECGBench MCQ gold option | 29 | 45 |  |
| CSN | LAD | left axis deviation | ECGBench MCQ gold option | 29 | 36 |  |
| CSN | TInv | t wave inversion | ECGBench MCQ gold option | 21 | 37 |  |
| CSN | LQRSV | low qrs voltages | ECGBench MCQ gold option | 21 | 33 |  |
| CSN | NSSTTA | nonspecific st t abnormality | ECGBench MCQ gold option | 14 | 30 |  |
| G12EC | TAb | t wave abnormal | ECGBench MCQ gold option | 135 | 319 |  |
| G12EC | NSSTTA | nonspecific st t abnormality | ECGBench MCQ gold option | 110 | 270 |  |
| G12EC | NSR | sinus rhythm | ECGBench MCQ gold option | 106 | 247 |  |
| G12EC | SB | sinus bradycardia | ECGBench MCQ gold option | 83 | 225 |  |
| G12EC | LQT | prolonged qt interval | ECGBench MCQ gold option | 79 | 181 |  |
| G12EC | STach | sinus tachycardia | ECGBench MCQ gold option | 73 | 186 |  |
| G12EC | LVH | left ventricular hypertrophy | ECGBench MCQ gold option | 70 | 166 |  |
| G12EC | LIs | lateral ischemia | ECGBench MCQ gold option | 62 | 132 |  |
| G12EC | LAD | left axis deviation | ECGBench MCQ gold option | 64 | 122 |  |
| G12EC | STIAb | st interval abnormal | ECGBench MCQ gold option | 55 | 123 |  |
| G12EC | LAE | left atrial enlargement | ECGBench MCQ gold option | 59 | 113 |  |
| G12EC | IAVB | 1st degree av block | ECGBench MCQ gold option | 49 | 121 |  |
| G12EC | TInv | t wave inversion | ECGBench MCQ gold option | 48 | 119 |  |
| G12EC | PAC | premature atrial contraction | ECGBench MCQ gold option | 37 | 81 |  |
| G12EC | RBBB | right bundle branch block | ECGBench MCQ gold option | 35 | 82 |  |
| G12EC | AF | atrial fibrillation | ECGBench MCQ gold option | 29 | 79 |  |
| G12EC | SA | sinus arrhythmia | ECGBench MCQ gold option | 27 | 68 |  |
| G12EC | IIs | inferior ischemia | ECGBench MCQ gold option | 34 | 59 |  |
| G12EC | QAb | qwave abnormal | ECGBench MCQ gold option | 28 | 56 |  |
| G12EC | IRBBB | incomplete right bundle branch block | ECGBench MCQ gold option | 26 | 56 |  |
| G12EC | LQRSV | low qrs voltages | ECGBench MCQ gold option | 25 | 57 |  |
| G12EC | PVC | premature ventricular contractions | ECGBench MCQ gold option | 17 | 54 |  |
| G12EC | AnMIs | anterior ischemia | ECGBench MCQ gold option | 14 | 32 |  |
| G12EC | LBBB | left bundle branch block | ECGBench MCQ gold option | 9 | 34 |  |
| SPH | NORM | Normal ECG | AHA statement code 1 | 334 | 1283 | no modifiers |
| SPH | SB | Sinus bradycardia | AHA statement code 22 | 57 | 273 | no modifiers |
| SPH | TAb | T-wave abnormality | AHA statement code 147 | 48 | 196 | merged: 147 unmodified (224); 147+367 inversion (20) |
| SPH | STDEV | ST deviation | AHA statement code 145 | 53 | 174 | merged: 145+362 depression (134); 145 unmodified (84); 145+363 elevation (9) |
| SPH | SA | Sinus arrhythmia | AHA statement code 23 | 48 | 147 | no modifiers |
| SPH | IRBBB | Incomplete right bundle-branch block | AHA statement code 105 | 23 | 114 | no modifiers |
| SPH | PVC | Ventricular premature complex(es) | AHA statement code 60 | 25 | 105 | merged: 60 unmodified (57); 60+310 frequent (33); 60+308 occasional (25); 60+341 in a bigeminal pattern (10); 60+342 in a trigeminal pattern (8) |
| SPH | STTC | ST deviation with T-wave change | AHA statement code 146 | 21 | 108 | no modifiers |
| SPH | AF | Atrial fibrillation | AHA statement code 50 | 22 | 59 | merged: 50 unmodified (54); 50+346 with a rapid ventricular response (27) |
| SPH | STach | Sinus tachycardia | AHA statement code 21 | 19 | 61 | no modifiers |
| SPH | RBBB | Right bundle-branch block | AHA statement code 106 | 14 | 58 | no modifiers |
| SPH | PAC | Atrial premature complex(es) | AHA statement code 30 | 17 | 49 | merged: 30 unmodified (31); 30+308 occasional (17); 30+310 frequent (15); 30+349 with aberrancy (3); 30+340 couplets (2); 30+341 in a bigeminal pattern (1) |

Pool = the 600-record development pool; test = fixed held-out split. Label names are given in American spelling (the G12EC prompts read 'ischaemia'). CSN/G12EC multi-label versions: gold = union of the correct MCQ options of each record, labels kept if they have at least 40 positive records, and records with none of them removed (CSN 1611 → 1579; G12EC 2026 → 1996; data/derived/*_ml.json). SPH: labels are the SPH primary statements of the AHA/ACC/HRS standard (Liu et al., Sci Data 2022), defined before any model output (docs/prereg_sph.md). Rules applied (clarity/build_sph.py): (1) one record per patient (lowest ECG_ID), random sample of 3000 (seed 42); (2) each statement is reduced to its primary code by dropping secondary modifiers after '+' (e.g. 145+362 'ST deviation, depression' and 145+363 'ST deviation, elevation' both count as 145); the merged statements are listed in the last column, with the number of records carrying each statement (a record can carry more than one, so these counts can sum to more than the number of positive records); (3) primary codes with ≥ 50 positives are kept; (4) records with none of them are excluded (2912 remain). No other merging was done, and no SPH label was mapped to a CPSC, CSN or G12EC label. Abbreviations are ours; STDEV (code 145, ST deviation, either direction), STTC (code 146, ST deviation with T-wave change) and TAb (code 147) keep their AHA meaning and are not equivalent to CPSC STD/STE or to the CSN/G12EC T-wave labels despite similar names. Results are therefore compared across datasets only as within-dataset contrasts (e.g. S1 − B0 on the same test split), never label-by-label. PTB-XL (fold 10) SCP codes were mapped to CPSC labels only for the domain analysis and the transfer experiment, by the rules in the last column (clarity/ptbxl_labels.py).
