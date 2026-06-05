# NES Music Generation: Text-to-Music Research

This repository contains the source code, models, and evaluation frameworks for our research on **Text-to-Music Generation** specifically tailored for the **Nintendo Entertainment System (NES)** architecture. 

The goal of this project is to explore and evaluate different methodologies for synthesizing authentic 8-bit chiptune audio from natural language descriptions.

## Project Architecture

We explore three distinct pipelines for converting textual prompts into NES-compliant audio:

1. **Text-to-Audio (Direct)**
   * End-to-end generation where text descriptions are directly translated into raw or synthesized 8-bit audio waveform outputs.

2. **Text-to-MIDI-to-Audio**
   * A two-stage pipeline where natural language is first translated into standard MIDI sequences, which are subsequently rendered into audio using specialized NES soundfonts/synthesizers.

3. **Text-to-NLM-to-Audio**
   * Our primary research focus. Text is translated into **NLM (NES Language Model format)**, a domain-specific representation that precisely captures the native constraints of the NES APU (Audio Processing Unit), including its specific pulse, triangle, noise, and DPCM channels. This NLM sequence is then deterministically compiled into native 8-bit audio.

## Repository Structure

Saída de código

README.md generated successfully.

```text
├── data/               # Datasets, text-prompt pairings, and preprocessed audio/MIDI/NLM files
├── src/
│   ├── audio/  # Modules for the direct text-to-audio baseline
│   ├── midi/   # Modules for text-to-MIDI conversion and MIDI synthesis
│   └── nlm/    # Modules for NLM parsing, text-to-NLM models, and NLM rendering
├── models/             # Model checkpoints and architecture configurations
├── notebooks/          # Exploratory data analysis and synthesis demonstrations
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

## Authors
- Igor Varejão
- Felipe Marra
- João Santana
- Lucas Ferreira Nascimento
