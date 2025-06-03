module FlocculusClassifier2025

# Import our global modules
using Statistics
using StatsBase
using DSP # Needed for filtering code
using Random
using JLD2
using PyPlot

# Flux Deep Learning Modules
using Flux

# Extras (for convenience)
import HDF5
import Downloads
import Pkg
import ProgressMeter

# Includes
include("recording.jl")
include("neuron.jl")
include("clips.jl")
include("correlograms.jl")
include("classifier.jl")
include("high_level_api.jl")

end # module FlocculusClassifier2025
