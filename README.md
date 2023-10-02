# Experiments with S4/Sashimi models

What is Sashmi: [https://arxiv.org/abs/2202.09729](https://arxiv.org/abs/2202.09729)

This repository is a fork of : [https://github.com/albertfgu/diffwave-sashimi](https://github.com/albertfgu/diffwave-sashimi)

Each branch is a different experiment

Please view the branches for more information.

- cold_diffusion2:  Using [Cold Diffusion](https://arxiv.org/abs/2208.09392) to transform midi inputs to audio waveforms using Sashimi
- freq_diffusion2: Diffusion in the frequency domain using Sashimi
- no_diffusion: Using Sashmi without diffusion
- proll_as_mel: Using piano roll for conditional diffusion

Other branches are not in use. 

## Generating waveforms without diffusion

Trying to answer the question: Do we even need diffusion? Predicitng the raw waveform directly from midi inputs using the sashimi model. 
Results were not very promising. 





