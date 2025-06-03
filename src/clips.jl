export get_clips, get_template

"""
    get_clips(recording, threshold_crossings, channel; clip_size)

Given a recording and the threshold croessings, return a matrix of clips for a
given set of threshold crossings. We center the clip on the threshold crossing
index. The width of the segment is passed to the function in units of seconds.
"""
function get_clips(recording::AbstractRecordingTimeseries, neuron::Neuron; kwargs...)
    indices = spike_indices(neuron)
    # Convert the timestamp of neuron into the timestamps of recording
    indices = convert.(Integer, round.(indices .* (sampling_rate(recording) / sampling_rate(neuron))))
    return get_clips(recording, indices; kwargs...)
end

function get_clips(recording::AbstractRecordingTimeseries, indices::AbstractVector{<:Integer}; clip_width::Tuple{Real, Real}=(2.5e-3, 2.5e-3), channels::AbstractVector{<:Integer}=1:size(recording, 1))
    @assert all(channels .> 0) .&& all(channels .<= size(recording, 1))

    pre_indices, post_indices = _clip_width_to_clip_indices(recording, clip_width)
    post_indices =  Integer(ceil(clip_width[2] * sampling_rate(recording)))
    indices = indices[(indices .> pre_indices) .&& (indices .+ post_indices) .<= size(recording, 2)]

    num_elements = pre_indices + post_indices

    # Create our output
    clips = zeros(typeof(recording[1]), length(indices), num_elements .* length(channels))
    for i = 1:length(indices)
        start = indices[i] - pre_indices
        stop = start + num_elements - 1
        for j = 1:length(channels)
            clips[i, ((j-1)*num_elements+1):(j*num_elements)] .= recording[channels[j], start:stop]
        end
    end

    return clips
end

"""
    get_template(recording, neuron)

A quick accessor function to get a template based on the median/mean
of the clips array. 
"""
function get_template(args...; robust::Bool=true, kwargs..., )
    clips = get_clips(args...; kwargs...)
    if robust
        return dropdims(median(clips, dims=1), dims=1)
    else
        return dropdims(mean(clips, dims=1), dims=1)
    end
end


function _clip_width_to_clip_indices(recording::AbstractRecordingTimeseries, clip_width::Tuple{Real, Real})
    pre_indices = Integer(ceil(clip_width[1] * sampling_rate(recording)))
    post_indices =  Integer(ceil(clip_width[2] * sampling_rate(recording)))
    return (pre_indices, post_indices)
end

"""
    peak_channel(clips)

Returns the index of the channel with the largest peak to peak amplitude
"""
function peak_channel(clips::AbstractArray{<:Integer, 2}, num_channels::Integer; robust::Bool=false)
    @assert (size(clips, 2) % num_channels) == 0
    # Compute the mean template
    mean_waveform = robust ? median(clips, dims=1) : mean(clips, dims=1)

    # Reshape so that they are across channels
    mean_waveform_across_channels = reshape(mean_waveform, :, num_channels)'
    largest_channel = 0
    largest_range = -Inf
    for i = 1:num_channels
        min_max = extrema(mean_waveform_across_channels[i, :])
        current_range = min_max[2] - min_max[1]
        if current_range > largest_range
            largest_range = current_range
            largest_channel = i
        end
    end
    return largest_channel
end

"""
    align_clips_to_peak_trough(clips, alignment_point, [peak, max_distance])

Given a matrix of clips, align all clips to their respective peak/trough.

Given an `NxNt` matrix of `N` clips, our goal is to align the peak (or the trough)
of each clip to the same index (`alignment_point`) in the matrix. The
returned matrix is an estimate of the true clips with the appropriate
alignment. That is, rather than going out to the recording and fetching
the complete clip when the alignment point changes, this function merely
extends the contents of the clip in the beginning or end (dependong on the direction
of the shift). The purpose of this function is to
quickly be able to estimate something near the optimal template for
peak and trough alignment and then use this estimate to optimally
align all remaining clips

This function accepts the following optional parameters:
 - `peak` (Bool): false if we should align clips to the trough
 - `max_distance` (Integer): Specifies a limit regarding how much we
    are willing to move the alignment point of the clip relative
    to the current alignment point. A negative value, the default,
    means ignore the distance

This function returns a copy of the original clips shifted
based on the requested peak/trough as well as the number
of indices shifted on a clip by clip basis.
"""
function align_clip_to_peak_trough!(clip::AbstractVector{<:Real},
                                    alignment_point::Integer;
                                    peak::Bool=true,
                                    max_distance::Integer=-1)
    # A for-loop is going to be faster than slicing
    indices_to_check = 1:length(clip)
    if max_distance >= 0
        indices_to_check = max(1, alignment_point-max_distance):min(length(clip), alignment_point+max_distance)
    elseif max_distance == 0
        return clip # Nothing to do
    end
    peak_index = indices_to_check[1]
    peak_value = clip[peak_index]
    for i = indices_to_check
        if ((peak == true) && (clip[i] > peak_value)) || ((peak == false) && (clip[i] < peak_value))
            peak_index = i
            peak_value = clip[i]
        end
    end
    # Wrap the clip if the alignment point changed
    lag = alignment_point - peak_index
    if lag < 0
        temp = clip[end]
        clip[1:end+lag] .= clip[-lag+1:end]
        clip[end+lag+1:end] .= temp
    elseif lag > 0
        temp = clip[1]
        clip[lag+1:end] .= clip[1:end-lag]
        clip[1:lag] .= temp
    end
    return clip, lag
end
function align_clips_to_peak_trough!(clips::AbstractArray{<:Real, 2},
                                     args...;
                                     kwargs...)
    lags = zeros(Int16, size(clips, 1))
    for i = 1:size(clips, 1)
        # We want to pass the actual clip not a slice, which is why we use a view
        _, lags[i] = align_clip_to_peak_trough!(view(clips, i, :), args...; kwargs...)
    end
    return clips, lags
end
align_clips_to_peak_trough(clips, args...; kwargs...) = align_clips_to_peak_trough!(deepcopy(clips), args...; kwargs...)

"""
    optimal_clip_delay(template, clip, [max_lag])

Estimate the delay between the passed clip and the template.

Using a cross-correlation analysis, estimate the delay associated with
the passed `clip` so that it is optimally aligned with `template`.

This function returns an integer corresponding to the number of indices
that clip is delayed relative to the passed template. Positive values
indicate that clip must be shifted forwards (to the right) in time whereas
negative values indicate that the clip lags the template and must be shifted
to the left. A value of zero indicates that the clip is already optimally
aligned with the passed template.

The optional parameter `max_lag` specifies the maximum lead or lag allowed.
By default this is equal to half the template length.
"""
function optimal_clip_delay(template::AbstractVector{<:Real}, clip::AbstractVector{<:Real}; max_lag::Integer=div(length(template), 2))
    @assert length(template) == length(clip)
    lags = (-min(max_lag+1, length(template)-1)):(min(max_lag-1, length(template)-1))
    I = argmax(StatsBase.crosscor(template, clip, lags))
    return lags[I]
end
function optimal_clip_delay(template::AbstractVector{<:Real},
                            clips::AbstractArray{<:Real, 2}; kwargs...)
    return [optimal_clip_delay(template, clips[i, :]; kwargs...) for i = 1:size(clips, 1)]
end


"""
    optimally_align_clips!(recording, spike_indices, [clip_size])

Given a set of spike indices, attempt to optimally align each of the clips

This function attempts to optimally align each clip to the optimal template.
It is assumed that spike_indices refers to the spikes for a single neuron.
We then compute an estimate of a semi-optimal template by aligning each
clip first by it's peak and then by it's trough. We say the appropriate
alignment template is either the peak or trough template whose range is
the largest (maximizes the SNR). Then, given this sub-optimal estimate of
the template, we use cross correlation to determine the optimal lag for each
spike index. The spike indices are modified in place within this function as
well as returned.

Note: This function assumes that the recording is stationary (i.e., has
minimal drift) and therefore the template for this neuron is unchanged.

Optional parameters:
 - clip_size (Real/Vector): The pre and post time to align the indices to
"""
function optimally_align_clips(recording::AbstractRecordingTimeseries,
                               spike_indices::AbstractVector{<:Integer},
                               channel::Integer;
                               clip_width::Tuple{Real, Real}=(2.5e-3, 2.5e-3))
    # Get our original (non-aligned clips)
    clips = get_clips(recording, spike_indices, channels=[channel], clip_width=clip_width)
    clip_size_indices = clip_width_to_clip_indices(recording, clip_width)

    # Align first to peak and determine the template's range
    peak_aligned_clips, _ = align_clips_to_peak_trough(clips,
                                                    clip_size_indices[1],
                                                    peak=true,
                                                    max_distance=min(clip_size_indices[1], clip_size_indices[2]))
    # NOTE: We use the median here rather than the mean to
    # compute the sub-optimal template. It is possible that
    # more than one spike exists in a neuron window (e.g., a small
    # amplitude spike that we want the template for and a larger
    # spike that is not the neuron of interest). Alignment to
    # the peak/trough will *always* align to the spike in the
    # clip window with the largest peak (even if your original spike
    # indices correspond to the small amplitude spike). If there
    # are enough of these overlapping spikes, the large amplitude spike
    # will significantly bias the mean and we will end up with a template
    # that is some combination of the two (or more) units. However, by
    # using the median, we essentially ask for the template that is
    # most common.
    peak_aligned_template = dropdims(median(peak_aligned_clips, dims=1), dims=1)
    peak_extrema = extrema(peak_aligned_template)
    peak_range = peak_extrema[2] - peak_extrema[1]

    # Do same thing aligning to trough
    trough_aligned_clips, _ = align_clips_to_peak_trough(clips,
                                                      clip_size_indices[1], # Alignment point is the offset
                                                      peak=false, # align to trough
                                                      max_distance=min(clip_size_indices[1], clip_size_indices[2]))
    trough_aligned_template = dropdims(median(trough_aligned_clips, dims=1), dims=1)
    trough_extrema = extrema(trough_aligned_template)
    trough_range = trough_extrema[2] - trough_extrema[1]
    optimal_template = trough_aligned_template
    if peak_range > trough_range
        optimal_template = peak_aligned_template
    end
    spike_indices = deepcopy(spike_indices)
    # Now that we have our optimal template, we can do our "optimal" alignment
    # NOTE that the cross correlation will not take into account overlapping
    # spikes. Therefore, if there is a higher SNR unit on this channel,
    # it is possible that it will "optimally" align to this large SNR unit
    # rather than the unit of interest.
    for i = 1:size(clips, 1)
        current_lag = optimal_clip_delay(optimal_template, clips[i, :];
            max_lag=min(clip_size_indices[1], clip_size_indices[2]))
        spike_indices[i] = spike_indices[i] + current_lag
    end
    return spike_indices
end


"""
    drift_shift_match()

Implementation of Maxime Beau's drift-shift matching template algorithm for Neuropixels
taken from the NeuroPyxels library available at
https://github.com/m-beau/NeuroPyxels/blob/b331ad33e8c95a8e931f67268d4779eef23c994f/npyx/spk_wvf.py#L192

The algorithm works as follows:
Take all spike clips/waveforms and average them in small "spike batches".
Because these batches are relatively small, Maxime assumes they have the same
"drift state" (semi-isolated temporally.)

Drift matching:
 - Select batches where the spikes peak on the same channel (modal). These batches
 are assumed to be in the same drift state (i.e. are close to the same channel)
 - From these batches on a common channel, select a subset of waveforms to average.
 These waveforms correspond to the largest amplitude (taking care to avoid artifacts),
 and therefore represent the template at it's highest SNR on this modal channel.

Shift matching:
 - Define a template from dift matched batches with the highest amplitude
 - Use cross correlation to optimally align the batches with the template
 - Exclude batches where the shift is larger than some allowed shift value

Note that this version of the algorithm is a simplified version used in the NeuroPyxles
library, and only computed the optimal template and peak channel.
"""
function drift_shift_match(recording::AbstractRecordingTimeseries,
    neuron::Neuron;
    clip_width::Tuple{Real, Real}=(2.5e-3, 2.5e-3),
    num_waveforms_per_batch::Integer=10,
    peak_channel_allowed_range::Integer=10,
    max_allowed_amplitude::Real=Inf,
    use_average_peak_channel::Bool=false,
    num_waveforms_to_average::Integer=5000,
    verbose::Bool=false)

    spike_crossings = spike_indices(neuron)
    spike_crossings = convert.(Integer, round.(spike_crossings .* (sampling_rate(recording) / sampling_rate(neuron))))

    # Split into consecutive batches of num_waveforms_per_batch
    spike_batches = Vector{AbstractVector{<:Integer}}(undef, Integer(ceil(length(spike_crossings) / num_waveforms_per_batch)))
    for i = 1:length(spike_batches)
        spike_batches[i] = spike_crossings[((i-1)*num_waveforms_per_batch+1):min(length(spike_crossings), (i*num_waveforms_per_batch))]
    end

    clips = get_clips(recording, neuron, clip_width=clip_width)
    original_peak_channel = peak_channel(clips, size(recording, 1))
    verbose && println("Starting with original peak channel as $original_peak_channel.")

    # Output variables
    batch_peak_channel = zeros(UInt64, length(spike_batches)) # The peak channel for each batch
    batch_max_amplitude = zeros(length(spike_batches)) # peak to peak amplitudes

    pre_indices = Integer(ceil(clip_width[1] * sampling_rate(recording)))
    post_indices =  Integer(ceil(clip_width[2] * sampling_rate(recording)))
    clip_size_indices = pre_indices + post_indices
    two_ms_indices = Integer(ceil(2e-3 / dt(recording)))

    # Collect spikes in each of the batches (given the clip size)
    # and compute the peak channel
    for i = 1:length(spike_batches)
        select = ((i-1)*num_waveforms_per_batch+1):min(length(spike_crossings), (i*num_waveforms_per_batch))
        mean_waveform = mean(clips[select, :], dims=1)
        # Reshape the mean waveform so that the first dimension is channel
        mean_waveform = reshape(mean_waveform, :, size(recording, 1))'
        channel_ranges = zeros(size(recording, 1))
        for channel_index = 1:size(recording, 1)
            min_max = extrema(mean_waveform[channel_index, :])
            channel_ranges[channel_index] = min_max[2] - min_max[1]
        end

        # Calculate peak to peak only within the channel range +/- peak_channel_allowed_range
        # and only within +/- 2ms of the spike index
        start_index = max(1, clip_size_indices[1] - two_ms_indices)
        stop_index = min(size(mean_waveform, 2), clip_size_indices[1] + two_ms_indices)
        start_channel = max(1, original_peak_channel - peak_channel_allowed_range)
        stop_channel = min(size(recording, 1), original_peak_channel + peak_channel_allowed_range)

        channel_ranges = zeros(stop_channel - start_channel + 1)
        for (j, channel_index) = enumerate(start_channel:stop_channel)
            min_max = extrema(mean_waveform[channel_index, start_index:stop_index])
            channel_ranges[j] = min_max[2] - min_max[1]
        end
        batch_peak_channel[i] = start_channel + argmax(channel_ranges) - 1
        batch_max_amplitude[i] = maximum(channel_ranges)
    end

    # Filter out batches that are beyond the bounds (artifacts)
    to_delete = batch_max_amplitude .> max_allowed_amplitude
    deleteat!(batch_peak_channel, to_delete)
    deleteat!(batch_max_amplitude, to_delete)
    spike_batches = spike_batches[(.! to_delete), :]

    # Z-drift matching (choose channel at the mode of the peak channel distribution)
    final_peak_channel = original_peak_channel
    if use_average_peak_channel == false
        # Use mode of peak channel distribution across spikes
        final_peak_channel = StatsBase.mode(batch_peak_channel)
        verbose && println("Continuing with modal peak channel as $final_peak_channel.")
    end

    # Select only batches that have the selected peak channel
    batch_select = (batch_peak_channel .== final_peak_channel)
    verbose && println("Selecting $(sum(batch_select)) of $(length(batch_select)) batches where batch peak channel is $final_peak_channel")

    # XY Drift Matching
    # Subselect batches with similar amplitude (i.e., similar distance to the probe)
    # and in particular, close to the largest amplitude (close to the probe,
    # but not the overall max to avoid artifacts)
    num_batches_to_use = Integer(ceil(num_waveforms_to_average / num_waveforms_per_batch))
    # Sort the  batches on the final peak channel by amplitude
    I = sortperm(batch_max_amplitude[batch_select], rev=false)
    if verbose
        figure()
        hist(batch_max_amplitude[batch_select])
        xlabel("Batch amplitude")
        ylabel("Count")
    end
    percent_95_index = Integer(round(0.95 * length(I)))
    if percent_95_index > num_batches_to_use
        # Select only up to the 95th percentile
        #batch_select[I[batch_select][(percent_95_index-num_batches_to_use):percent_95_index]] .= tru
        select = falses(length(I))
        select[1:percent_95_index - num_batches_to_use] .= true
        select[percent_95_index:end] .= true
        view(view(batch_select, batch_select), I)[select] .= false

        #batch_select[I[batch_select][.! select]] .= false
        if verbose
            axvline(minimum(batch_max_amplitude[batch_select]), color="r")
            axvline(maximum(batch_max_amplitude[batch_select]), color="r")
            println("Selecting only $(sum(batch_select)) batches up to the 95%.")
        end
    end

    # Given our original clips, we can pull the clips only for the selected batches
    clip_select = falses(size(clips, 1))
    offset = 1
    verbose
    for i = 1:length(batch_select)
        clip_select[offset:offset+length(spike_batches[i])-1] .= batch_select[i]
        offset = offset + length(spike_batches[i])
    end

    # Use the largest 50 clips to compute our template
    # The median here avoids contamination by erroneous spikes
    pre_aligned_clips = clips[clip_select, :]
    final_crossings = spike_crossings[clip_select]
    sub_optimal_template = dropdims(median(pre_aligned_clips, dims=1), dims=1) # Use median to avoid outliers
    two_ms_indices = Integer(ceil(2e-3 / dt(recording)))
    # Do shift matching (align to optimal template)
    verbose && println("Realigning clips.")
    for i = 1:size(pre_aligned_clips, 1)
        current_lag = optimal_clip_delay(sub_optimal_template, pre_aligned_clips[i, :];
            max_lag=two_ms_indices)
        final_crossings[i] = min(max(1, final_crossings[i] + current_lag), size(recording, 2))
    end
    final_clips = get_clips(recording, final_crossings, clip_width=clip_width, channels=[final_peak_channel])
    final_template = dropdims(median(final_clips, dims=1), dims=1) # Use median to avoid outliers
    return final_crossings, final_template, final_peak_channel
end

"""
    preprocess_template(template)

Performs normalization (scaling), realignment (shifting), and potentially
inversion (flipping) to ensure that templates are maximally similar across
recording preparations.

The output of this function is a scaled/flipped/shifted template such 
that the new template has it's maximal peak located at index I, where index
I is specified by clip size. 

The optional input `exemplar_template` allows alignment to an example
template (e.g., the mean template) based on the point that maximizes the
cross-correlation between the current example and the exemplar.
"""
function preprocess_template(template::AbstractVector{<:Real}; 
        original_sampling_rate::Real=30000,
        output_sampling_rate::Real=original_sampling_rate,
        clip_width::Tuple{Real, Real}=(1e-3, 3e-3),
        peak_sign::Union{Nothing, Symbol}=:negative,
        peak_range::Union{Nothing, Tuple{Real, Real}}=nothing, # Time is defined relative to the start of the template
        remove_baseline::Bool=true,
        normalize::Bool=true,
        exemplar_template::Union{Nothing, AbstractVector{<:Real}}=nothing)
    #@assert original_sampling_rate >= output_sampling_rate
    if original_sampling_rate != output_sampling_rate
        # Resample the original template to the new sampling rate
        template = DSP.resample(template, output_sampling_rate / original_sampling_rate)
    end
    
    # Remove any baseline offset
    if remove_baseline
        template .= template .- template[1]
    end

    # Given our clip size, determine the desired index of our
    # peak
    I_peak = Integer(round(abs(clip_width[1]) * output_sampling_rate))
    
    local I
    @assert exemplar_template == nothing || length(exemplar_template) == length(template)
    if exemplar_template !== nothing
        lag_axis = -div(length(template), 2):1:div(length(template), 2)
        I = I_peak - lag_axis[argmax(abs.(crosscor(template, exemplar_template, lag_axis)))]
    else
        # Search through our template to find our desired alignment point
        # We only align to peaks, so our goal is to find a set of local peaks
        # first and then choose the optimal one
        peaks = Vector{Integer}(undef, 0)
        for i = 2:length(template)-1
            if peak_range !== nothing && (((i/output_sampling_rate) < peak_range[1]) || ((i/output_sampling_rate) > peak_range[2]))
                continue
            end

            if (template[i] > template[i-1]) && (template[i] >= template[i+1]) && (template[i] > 0)
                push!(peaks, i) # Positive peak
            elseif (template[i] < template[i-1]) && (template[i] <= template[i+1]) && (template[i] < 0)
                push!(peaks, i) # Negative peak
            end
        end
        
        # Given our list of peaks, our goal is to find the optimal peak,
        # typically this will be the maximum value, but we align to the first
        # peak that is at least 75% of the maximum value
        peak_values = abs.(template[peaks])
        if length(peak_values) >= 1
            current_peak = maximum(peak_values)
            I = peaks[findfirst(peak_values .> 0.75 * current_peak)]
        else
            I = argmax(abs.(template))
        end
    end
    peak_val = abs.(template[I])
    # Determine if we need to flip our template based on the value of the peak
    # ensuring that the peak is negative
    if peak_sign != nothing && peak_sign == :negative && template[I] > 0
        template = template * -1
    elseif peak_sign != nothing && peak_sign == :positive && template[I] < 0
        template = template * -1
    end

    # Construct our output template based on our desired clip_size
    num_indices = Integer(round((abs(clip_width[1]) + abs(clip_width[2])) * output_sampling_rate))
    if I < I_peak
        template = vcat(ones(I_peak - I) * template[1], template) # Extend the beginning of the template
    elseif I > I_peak
        template = template[(I - I_peak + 1):end]
    end
    @assert abs.(template[I_peak]) == peak_val
    
    if length(template) > num_indices
        template = template[1:num_indices]
    elseif length(template) < num_indices
        template = vcat(template, template[end] * ones(num_indices - length(template)))
    end
    @assert length(template) == num_indices
    
    # Remove any (noisy) offset
    num_indices = Integer(round(abs(clip_width[1]) .* output_sampling_rate))
    template = template .- median(template[1:num_indices])
    
    # Normalize the result
    if normalize
        template = template ./ abs(template[I_peak])
        #template = template ./ maximum(abs.(template))
    end
    
    return template 
end