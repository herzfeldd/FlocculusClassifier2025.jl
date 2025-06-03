export Neuron
export spike_indices

"""
    Neuron(spike_indices, sampling_rate)

A structure representing the spikes of a particular (sorted) neuron.
The spikes indices should be represented as integers and are always
unique and sorted. 

We use this structure in this library to avoid passing different 
sampling rates - which is important when recordings may be sampled
at different rates (e.g., LFP recordings with a lower sampling rate).
"""
mutable struct Neuron
    spike_indices::AbstractVector{<:Real}
    sampling_rate::Real
    function Neuron(spike_indices::AbstractVector{<:Real}, sampling_rate::Real)
        @assert sampling_rate > 0
        return new(sort(unique(spike_indices)), sampling_rate)
    end
end

function Base.show(io::IO, x::Neuron)
    print(io, "Neuron with $(length(x)) spikes @ $(sampling_rate(x)) Hz.")
end

"""
    Neuron(spiketimes, [assumed_sampling_rate])

Construct a neuron from a vector of spike times. We convert these into indices
implicity with an assumed sampling rate.
"""
function Neuron(spiketimes::AbstractVector{<:AbstractFloat}; assumed_sampling_rate::Real=40000.0)
    spike_indices = convert.(Int64, ceil.(spiketimes * assumed_sampling_rate))
    return Neuron(spike_indices, assumed_sampling_rate)
end

sampling_rate(neuron::Neuron) = neuron.sampling_rate
dt(neuron::Neuron) = 1.0 / sampling_rate(neuron)
spike_indices(neuron::Neuron) = neuron.spike_indices
spiketimes(neuron::Neuron) = spike_indices(neuron) .* dt(neuron)
Base.length(neuron::Neuron) = length(spike_indices(neuron))
Base.size(neuron::Neuron, args...) = Base.size(neuron.spike_indices, args...)
Base.getindex(neuron::Neuron, inds...) = Base.getindex(neuron.spike_indices, inds...)

"""
    Neuron(url)

Simple accessor function to load a neuron from a URL. Prefix must be
http/https. The contents of the file are assumed to be a JLD2
"""
function Neuron(url::AbstractString)
    if ! occursin(r"^http", url)
        error("URL must begin with http or https")
    end
    filename = Downloads.download(url)
    f = JLD2.jldopen(filename, "r")
    neuron = Neuron(f["neuron"][:spike_indices], f["neuron"][:sampling_rate])
    Base.Filesystem.rm(filename)
    return neuron
end