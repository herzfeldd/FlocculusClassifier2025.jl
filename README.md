# Neuron-type Classification in the Cerebellar Flocculus

[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2025.01.29.634860-blue)](https://doi.org/10.1101/2025.01.29.634860)

This repository contains code for classifying neuron types in the cerebellar flocculus using supervised deep-learning methods. 
The classifier is described in detail in our BioRxiv manuscript available [here](https://doi.org/10.1101/2025.01.29.634860).

## Overview
In our study, we recorded extracellularly from the floccular complex of the cerebellum while 
animals performed smooth pursuit eye movements to visual targets. While extracellular recordings provide excellent t
emporal resolution at the single-spike level in behaving animals, they have the significant drawback that the neuron 
identity of recorded units remains unknown.

To address this limitation, David Herzfeld, Nate Hall, and Steve Lisberger developed methods to identify neuron types for a 
subset of recorded neurons. These identifications were made by leveraging knowledge of cerebellar circuitry, laminar information, 
functional discharge properties, and waveform characteristics. While only a fraction of neurons could be identified using 
these strict criteria, the analysis revealed that many features available from extracellular recordings contained information about neuronal identity.

This observation led to the development of supervised deep-learning classifiers that were cross-validated against 
expert-assigned labels. These classifiers can assign labels to neurons in the cerebellar cortex that don't possess 
the same level of evidence required for expert-level identification.

## What This Repository Provides
The code in this repository implements the classifier detailed in the manuscript. It takes as input:
- Raw waveforms
- Spike-triggered local field potentials (LFPs)
- Auto-correlograms
And outputs predictions (with confidence scores) about the identity of the supplied neuron.

## Requirements
To use this repository on your own data, you will need:
1. **Neurophysiological recording**: Single or multi-channel wideband data
2. **Spike times**: A series of spike times associated with neural units, typically the output of a spike sorting procedure

## Getting Started

### Follow Along with Examples

This repository contains two Jupyter notebooks to help you get started:
1. **Step-by-step Example**: Goes through each step of the classification process in detail
2. **High-level Example**: Contains high-level code for rapid classification

### Installation
The code is written in [Julia](https://julialang.org/) and requires a 
working installation of Julia (version > 1.0). Please refer to the Julia documentation 
for installation instructions for your operating system.

Once Julia is installed, you can install this package using:
```julia
import Pkg
Pkg.add(url="https://github.com/herzfeldd/FlocculusClassifier2025.jl.git")
```

These commands will download all necessary software packages as well as the pre-trained classifier models. 
The models can also be examined in their raw form in the Open Science Framework 
[repository](https://osf.io/kcx39/?view_only=d24c8ab9f2c341a8bdda3affb4a9edd4).

**Note**: These installation commands only need to be run once after installing Julia.

To load the package in each new Julia session:
```julia
using FlocculusClassifier2025
```

## Quick Start
The high-level workflow is straightforward:
1. Load your wideband recording using the `FlatRecording` structure
2. Define one or more `Neuron` objects containing spike indices
3. Call the `predict` function to get predicted labels

That's it! The sections below provide detailed information about each step.

## Detailed Usage

### Data Quality Requirements
**Important**: The classifier relies on waveform and rate-based metrics to identify cerebellar units. I
nput data quality is critical (garbage in = garbage out). Ensure your preprocessing pipeline identifies only well-isolated single units by:
- Removing units with significant refractory violations (indicating contamination)
- Excluding units with low signal-to-noise ratios
- Eliminates any units with evidence of multi-unit activity
Units that are barely visible in raw voltage traces or show evidence of amplitude 
clipping should not be classified, as they likely represent multi-unit activity or have missed spikes due to falling below the noise threshold.

### Loading Recordings
This package uses a `FlatRecording` structure to manage voltage data across time and channels, 
along with sampling rate information. The recording format assumes data are stored as 16-bit signed 
integers in channel-contiguous format.

Load an example recording:
```julia
wideband_recording = FlocculusClassifier2025.FlatRecording(
    "https://osf.io/download/8k59f/",  # Path to file or URL
    1,                                  # Number of channels
    40000.0                            # Sampling rate (Hz)
)
```

### Preprocessing and Filtering
Proper filtering is crucial because waveform shapes and spike-triggered LFPs 
are direct inputs to the classifier. Changes in filtering properties will distort 
waveforms and degrade classification performance.

The package requires two filtered versions of your data:

#### 1. Spike Band
High-pass filtered at 300 Hz using a first-order Butterworth causal filter (equivalent to Neuropixels on-board hardware filtering):
```julia
recording_spike_band = FlocculusClassifier2025.preprocess_spikeband(wideband_recording)
```

#### 2. LFP Band
Band-pass filtered between 5-600 Hz using a second-order causal Butterworth filter, then downsampled to 2,500 Hz:

```julia
recording_lfp_band = FlocculusClassifier2025.preprocess_lfpband(wideband_recording)
```

**Note**: The 5 Hz high-pass filter prevents low-frequency baseline drift, making this slightly 
different from standard Neuropixels LFP processing.

### Defining Neurons
Create `Neuron` objects containing spike indices and sampling rate:

```julia
example_neuron = Neuron(spike_indices, sampling_rate)
```

Where:
- `spike_indices`: Array of integer time samples since recording start
- `sampling_rate`: Sampling rate in samples/second

Load an example neuron:

```julia
neuron = FlocculusClassifier2025.Neuron("https://osf.io/download/g6hq3/")
```

### Feature Extraction

#### Waveform Features
The classifier uses two waveform features:

1. **Mean action potential waveform** (spike template)
2. **Mean spike-triggered LFP waveform**

Extract the primary spike template using a drift-shift algorithm for optimal alignment:

```julia
_, primary_template, primary_channel = FlocculusClassifier2025.drift_shift_match(
    recording_spike_band, neuron, verbose=false
)
```

Preprocess the template to match classifier requirements (40 kHz sampling, negative deflection):

```julia
primary_template = FlocculusClassifier2025.preprocess_template(
    primary_template, 
    original_sampling_rate=FlocculusClassifier2025.sampling_rate(recording_spike_band), 
    output_sampling_rate=40000.0, 
    peak_sign=:negative, 
    normalize=true, 
    clip_width=(1e-3, 3e-3)  # 1ms before to 3ms after peak
)
```

Generate the spike-triggered LFP template using similar steps, but with 2.5 kHz sampling 
and a window from 5ms before to 10ms after the peak deflection.

#### Rate-based Features

Extract auto-correlograms (ACGs):

```julia
time_axis = 1e-3:1e-3:250e-3  # 1ms bins up to 250ms
acg_2d = FlocculusClassifier2025.get_auto_correlogram(neuron, time_axis=time_axis)
_, acg_3d = FlocculusClassifier2025.get_auto_correlogram_vs_firing_rate(neuron, time_axis=time_axis)

# Normalize for classifier input
acg_2d = Float32.(acg_2d ./ (100 * step(time_axis)))
acg_3d = Float32.(acg_3d ./ (100 * step(time_axis)))
```

**Note**: The final classifier uses only 3D ACGs, as 2D ACGs don't provide additional classification information. Only the positive half of ACGs is used due to symmetry.

### Making Predictions

#### High-level Prediction

The simplest approach is to provide a recording and vector of neurons:

```julia
predictions = FlocculusClassifier2025.predict(recording, neurons)
```
This code will generate all of the intermediate information (in the appropriate format)
for input into the classifer.

#### Low-level Prediction
If you have pre-extracted features:

```julia
# Concatenate template and ACG features
mlp_input = vcat(primary_template, lfp_template, acg_2d)

# Load trained models
trained_models, classes = FlocculusClassifier2025.get_trained_models()

# Make prediction
predicted_label, confidence_ratio, label_probabilities = FlocculusClassifier2025.predict(
    trained_models, classes, acg_3d, mlp_input)
```

#### Output Interpretation
The prediction function returns:
- **predicted_label**: Most likely neuron type
- **confidence_ratio**: Confidence score for the prediction
- **label_probabilities**: Full probability distribution across all neuron types

## Data and Models
Pre-trained models and example data are available through our 
Open Science Framework repository: https://osf.io/kcx39/?view_only=d24c8ab9f2c341a8bdda3affb4a9edd4

## Support
For questions about the classifier or issues with the code, please open an issue on this 
repository or contact the authors through the manuscript correspondence information.

