# Tokenizer Evaluation Results — flores_core / lines

_Last updated: 2026-03-12 20:07:12_

| Tokenizer | Vocab Size | Fertility ↓ | Compression Rate ↑ | Vocab Util. ↑ | Avg Token Rank ↓ | Gini ↓ | 3-Digit Align. F1 ↑ | Op. Isolation ↑ | Bound. Cross ↓ | Char Split ↓ | CER ↓ | WS Fidelity ↑ | Languages | Dataset | User | Date |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BPE ByteLevel (default) [128k] | 128,000 | 39.881 | 0.025 | 0.292 | 4652.2 | 0.134 | 0.775 | 0.992 | 0.0039 | 0.0091 | **0.0000** | **1.000** | 13 | flores_core | cmeister747 | 2026-03-12 |
| PA BPE ByteLevel (default) [128k] | 127,808 | 53.783 | 0.019 | 0.152 | 1556.1 | 0.170 | 0.768 | 0.992 | 0.0155 | 0.0554 | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| BPE Punctuation + ByteLevel [128k] | 128,000 | 40.179 | 0.025 | **0.295** | 4614.5 | 0.136 | 0.775 | **1.000** | 0.0038 | 0.0089 | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| PA BPE Punctuation + ByteLevel [128k] | 127,840 | 54.220 | 0.018 | 0.154 | **1529.1** | 0.174 | 0.768 | 1.000 | 0.0154 | 0.0559 | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| BPE GPT-4o regex + ByteLevel [128k] | 128,000 | 37.555 | **0.027** | 0.290 | 5156.4 | 0.070 | **0.797** | 0.966 | 0.0044 | 0.0102 | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| PA BPE GPT-4o regex + ByteLevel [128k] | 127,808 | 42.626 | 0.023 | 0.220 | 3290.0 | 0.093 | 0.797 | 0.966 | **0.0029** | **0.0074** | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| BPE Mistral-Nemo regex + ByteLevel [128k] | 128,000 | 37.947 | 0.026 | 0.290 | 5062.2 | 0.069 | 0.450 | 0.966 | 0.0043 | 0.0101 | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| PA BPE Mistral-Nemo regex + ByteLevel [128k] | 127,818 | 43.704 | 0.023 | 0.211 | 3035.5 | 0.089 | 0.450 | 0.966 | 0.0035 | 0.0091 | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| BPE Qwen 3.5 regex + ByteLevel [128k] | 128,000 | 37.993 | 0.026 | 0.289 | 5038.2 | **0.068** | 0.450 | 0.966 | 0.0044 | 0.0102 | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| PA BPE Qwen 3.5 regex + ByteLevel [128k] | 127,818 | 43.725 | 0.023 | 0.211 | 3032.0 | 0.089 | 0.450 | 0.966 | 0.0035 | 0.0091 | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| BPE GPT-4o regex (no contractions) + ByteLevel [128k] | 128,000 | 37.540 | 0.027 | 0.290 | 5161.5 | 0.070 | 0.797 | 0.966 | 0.0044 | 0.0102 | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| PA BPE GPT-4o regex (no contractions) + ByteLevel [128k] | 127,808 | 42.603 | 0.023 | 0.220 | 3293.1 | 0.093 | 0.797 | 0.966 | 0.0030 | 0.0074 | 0.0000 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| BPE NFC + ByteLevel [128k] | 128,000 | 39.872 | 0.025 | 0.292 | 4652.9 | 0.134 | 0.775 | 0.992 | 0.0038 | 0.0095 | 0.0001 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| PA BPE NFC + ByteLevel [128k] | 127,808 | 53.772 | 0.019 | 0.152 | 1556.6 | 0.170 | 0.768 | 0.992 | 0.0155 | 0.0557 | 0.0001 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| BPE NFC + GPT-4o regex + ByteLevel [128k] | 128,000 | **37.539** | 0.027 | 0.290 | 5157.8 | 0.069 | 0.797 | 0.966 | 0.0044 | 0.0107 | 0.0001 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| PA BPE NFC + GPT-4o regex + ByteLevel [128k] | 127,808 | 42.610 | 0.023 | 0.220 | 3290.3 | 0.093 | 0.797 | 0.966 | 0.0029 | 0.0079 | 0.0001 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| BPE NFC + Mistral-Nemo regex + ByteLevel [128k] | 128,000 | 37.930 | 0.026 | 0.290 | 5063.6 | 0.068 | 0.450 | 0.966 | 0.0043 | 0.0106 | 0.0001 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| PA BPE NFC + Mistral-Nemo regex + ByteLevel [128k] | 127,818 | 43.690 | 0.023 | 0.211 | 3035.8 | 0.089 | 0.450 | 0.966 | 0.0035 | 0.0096 | 0.0001 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| BPE NFC + Qwen 3.5 regex + ByteLevel [128k] | 128,000 | 37.977 | 0.026 | 0.289 | 5039.6 | 0.068 | 0.450 | 0.966 | 0.0044 | 0.0107 | 0.0001 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
| PA BPE NFC + Qwen 3.5 regex + ByteLevel [128k] | 127,818 | 43.711 | 0.023 | 0.211 | 3032.2 | 0.089 | 0.450 | 0.966 | 0.0035 | 0.0096 | 0.0001 | 1.000 | 13 | flores_core | cmeister747 | 2026-03-12 |
