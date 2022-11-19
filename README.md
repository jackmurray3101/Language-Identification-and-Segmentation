# Language Identification and Segmentation System
This codebase provides an implementation for a language segmentation system, implemented using self-supervised speech representation frameworks.

Frameworks used include:
 - wav2vec2
 - HuBERT
 - data2vec
 - XLSR

The frameworks were compared for LID on VoxLingua107, against a benchmark system https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa.
The structure used should enable other models to be compared easily in the future - see the LID folder.

For the segmentation stage, a cosine similarity was computed from outputs of the LID stage. To create multilingual data with time-stamped transition times, monolingual language files were spliced together at random intervals within some time range, depending on the desired data characteristics. See datasets folder for corresponding code.

