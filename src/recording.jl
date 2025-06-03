export FlatRecording
export sampling_rate
export dt
export preprocess_spikeband, preprocess_lfpband

# Define abstract types (and abstract recording has samples and a sampling rate)
abstract type AbstractRecordingTimeseries end
sampling_rate(::AbstractRecordingTimeseries) = error("Not implemented")
dt(x::AbstractRecordingTimeseries) = 1.0 / sampling_rate(x)
Base.size(::AbstractRecordingTimeseries) = error("Not implemented")
Base.getindex(::AbstractRecordingTimeseries, inds...) = error("Not implemented")
Base.setindex!(::AbstractRecordingTimeseries, X, inds...) = error("Not implemented")

"""
    FlatRecording

A wrapper around a binary recording that is really just a matrix of samples
with dimensions channels x timepoints along with an associated sampling rate.

The access functions are passed directly to the underlying samples matrix.
"""
mutable struct FlatRecording <: AbstractRecordingTimeseries
    samples::AbstractMatrix{<:Real} # A 2D matrix of channels x samples
    sampling_rate::Real
end
sampling_rate(x::FlatRecording) = x.sampling_rate
Base.size(x::FlatRecording, args...) = Base.size(x.samples, args...)
Base.getindex(x::FlatRecording, inds...) = Base.getindex(x.samples, inds...)
Base.setindex!(recording::FlatRecording, X, inds...) = Base.setindex!(recording.samples, X, inds...)

function Base.show(io::IO, x::FlatRecording)
    print(io, "Recording @ $(sampling_rate(x)) Hz. $(size(x, 1)) channel(s) x $(size(x, 2)) timepoints\n")
end

"""
    FlatRecording(stream, num_channels, sampling_rate::Real, T::Type{<:Integer}, [offset=0, interleaved])

Read a flat recording data structure with an (optional) header. The data are assumed
to be stored after an optional header as binary integer data of type `T'. The type defaults
to Int16 - which is the common format used by many recording systems.

By default, we assume that the voltage timeseries for each channel are stored in
order. That is, the binary data for channel 2 immediately follow the binary data
from channel 1 within the file (non-interleaved data). By passing the boolean variable
`interleaved=true`, the data will be read in timeseries order (such that channel 2
timepoint 0 immediately follows channel 1 timepoint 0).
"""
function FlatRecording(fp::IO,
        num_channels::Integer, sampling_rate::Real;
        offset::Integer=0, T::Type{<:Integer}=Int16,
        interleaved::Bool=false)
    @assert offset >= 0
    @assert num_channels >= 0
    # Determine the size of the file with the header removed
    seekend(fp)
    data_size = position(fp) - offset
    # Ensure that the data size evenly divides by the number of channels
    num_points_per_channel = Integer(floor(data_size / sizeof(T) / num_channels))
    @assert (data_size % (num_points_per_channel * sizeof(T))) == 0

    # Seek to the start of the binary data
    seek(fp, offset)

    
    if interleaved
        samples = read!(fp, Matrix{T}(undef, num_channels, num_points_per_channel));
    else
        samples = read!(fp, Matrix{T}(undef, num_points_per_channel, num_channels));
        samples = samples'
    end

    return FlatRecording(samples, sampling_rate)
end

function FlatRecording(filename::AbstractString, args...; kwargs...)
    recording = nothing
    if occursin("http", filename) # Assume this is a url
        buffer = IOBuffer()
        _ = Downloads.download(filename, buffer)
        recording = FlatRecording(buffer, args...; kwargs...)
    else # Assume this is a file
        fp = open(filename, "r")
        recording = FlatRecording(fp, args...; kwargs...)
        close(fp)
    end
    return recording
end


"""
    preprocess_spikeband(recording)

Preprocess a wideband recording to return a new recording object
that this highpass filtered above 300 Hz with a single-order 
buttworth (causal) filter. This filter mimics the filtering 
of Neuropixels recordings.
"""
function preprocess_spikeband(recording::AbstractRecordingTimeseries; cutoff::Real=300, order::Integer=1)
    ff = DSP.digitalfilter(DSP.Highpass(cutoff), fs=sampling_rate(recording), Butterworth(order))
    processed_recording = deepcopy(recording)
    T = typeof(recording[1])
    for i = 1:size(recording, 1)
        processed_recording[i, :] = convert.(T, clamp.(round.(filt(ff, recording[i, :])), T))
    end
    return processed_recording
end

"""
    preprocess_lfpband(recording)

Preprocess a wideband recording to return a new recording object
that this lowpass filtered and decimated to 2500 Hz.
"""
function preprocess_lfpband(recording::AbstractRecordingTimeseries; cutoff::Tuple{Real, Real}=(5.0, 500.0), order::Integer=2, output_sampling_rate::Real=2500.0)
    ff = DSP.digitalfilter(DSP.Bandpass(cutoff[1], cutoff[2]), fs=sampling_rate(recording), Butterworth(order))
    processed_recording = deepcopy(recording)
    T = typeof(recording[1])
    for i = 1:size(recording, 1)
        processed_recording[i, :] = convert.(T, clamp.(round.(filt(ff, recording[i, :])), T))
    end
    # Decimate
    voltage_resampled = DSP.resample(processed_recording[:, :], output_sampling_rate / sampling_rate(processed_recording), dims=2)
    voltage_resampled = convert.(T, clamp.(round.(voltage_resampled), T))
    return FlatRecording(voltage_resampled, output_sampling_rate)
end