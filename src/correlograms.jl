"""
    get_cross_correlogram(spike_indices_1, spike_train_2, n_bins)

Internal function that is the work-horse for generating cross correlograms.
This function takes a set of spike indices for neuron 1 (the reference neuron)
and returns the probability of observing a spike in neuron 2 (from its passed
spiketrain) at each time where neuron 1 fires. This function takes the spike
train of neuron 2, to allow spike trains that included NaNs. We take care to
track the occurrence of these NaNs to ensure that we they do not bias the resulting
CCG.

The parameter n_bins specifies the number of pre- and post-bins relative to the
reference spike. This should be a tuple of positive integers (defaults to (100,100)).

This function returns the probability of observing a spike in the associated
spike_train_2 at the reference indices passed in the first argument.
"""
function get_cross_correlogram(spike_indices_1::AbstractVector{<:Real}, spike_train_2::AbstractVector{<:Real}, n_bins::Tuple{Integer, Integer}=(100, 100))
    # This is essentially a histogram, were we count the number of events,
    # but we want to allow missing data (NaNs) in either of initial spike indices (trigger)
    # or spike train
    @assert n_bins[1] >= 0 && n_bins[2] >= 0
    total_bins = n_bins[1] + n_bins[2] # Total number of bins
    counts = zeros(total_bins + 1) # number of times we see a spike in neuron 2 at a particular point
    divisor = zeros(total_bins + 1) # number of times we actually have data in for neuron 2 at a particular point (allows NaNs)

    for (i, spike_index) = enumerate(spike_indices_1)
        if isinf(spike_index) || isnan(spike_index)
            continue
        end
        spike_index = Integer(spike_index)
       # Construct a timeseries for neuron 2 based on the current spike index as the
       # zero point
        start = max(1, spike_index - n_bins[1])
        stop = min(spike_index + n_bins[2], length(spike_train_2))
        for j = start:stop
            if isnan(spike_train_2[j])
                continue
            end
            index = j - start + 1
            #counts[index] += (spike_train_2[j] > 0)
            counts[index] += spike_train_2[j]
            divisor[index] += 1
        end
    end
    return counts ./ divisor
end

"""
    get__correlogram(spike_indices_1, dt_1, spike_indices_2, dt_2, [time_axis, set_zero, santize])

Generate a cross correlogram between the spike_indices passed for neuron 1 (the
reference neuron) and the spike indices passed for neuron 2.

Optional arguments:
 - set_zero (Bool): Sets the time point nearest zero to zero (good for ACGs)
 - santize (Bool): Remove any spike indices with ISI violations (defined as 1ms)
   from the reference neuron as well as the second neuron. In the second neuron,
   these indices are replaced as NaNs so as to not bias the resulting CCG.
"""
function get_cross_correlogram(spike_indices_1::AbstractVector{<:Real}, dt_1::Real, spike_indices_2::AbstractVector{<:Real}, dt_2::Real; time_axis::AbstractRange=-100e-3:1e-3:100e-3, set_zero::Bool=false, sanitize::Bool=false)
    # Round the new dt to the nearest microsecond
    dt_new = Integer(round(step(time_axis) * 1E6)) ./ 1E6

    # Convert the spike indices to units of ms
    spike_indices_1 = ceil.(spike_indices_1 ./ (dt_new ./ dt_1))
    spike_indices_2 = ceil.(spike_indices_2 ./ (dt_new ./ dt_2))
    deleteat!(spike_indices_1, spike_indices_1 .< 1)
    deleteat!(spike_indices_2, spike_indices_2 .< 1)

    # Convert to spike trains
    len = Integer(ceil(max(maximum(spike_indices_1[.! isnan.(spike_indices_1)]), maximum(spike_indices_2[.! isnan.(spike_indices_2)]))))

    #spike_trains_2 = falses(len)
    #spike_trains_2[spike_indices_2] .= true
    # The above two lines do not permit non-uniqe elements in spike_indices_2,
    # the below lines correct this.
    spike_train_2 = zeros(Float32, len)
    for i = 1:length(spike_indices_2)
        if isnan(spike_indices_2[i]) || isinf(spike_indices_2[i])
            # Nan from previous non-nan spike to next nan-nan spike
            start_index = findlast((.! isnan.(spike_indices_2)) .& (1:length(spike_indices_2) .< i))
            stop_index = findfirst((.! isnan.(spike_indices_2)) .& (1:length(spike_indices_2) .> i))
            start_index = start_index === nothing ? 1 : Integer(spike_indices_2[start_index]) + 1
            stop_index = stop_index === nothing ? length(spike_train_2) : Integer(spike_indices_2[stop_index]) - 1
            spike_train_2[start_index:stop_index] .= NaN
        else
            spike_train_2[Integer(spike_indices_2[i])] += 1
        end
    end
    if sanitize
        absolute_isi = 1e-3
        # Given an isi violation, we remove both the first and second spike
        isi_violations_1 = vcat(diff(spike_indices_1) .< (absolute_isi / dt_new), false)
        isi_violations_1 = isi_violations_1 .| vcat(false, isi_violations_1[1:end-1])

        deleteat!(spike_indices_1, isi_violations_1)
        # Do the same thing for neuron 2 expect fill the spike train with NaNs
        isi_violations_2 = vcat(diff(spike_indices_2) .< (absolute_isi / dt_new), false)
        isi_violations_2 = isi_violations_2 .| vcat(false, isi_violations_2[1:end-1])
        I = findall(isi_violations_2)
        for i = I
            start = i > 1 ? spike_indices_2[i-1] + 1 : 1
            stop = i < length(spike_indices_2) ? spike_indices_2[i+1] - 1 : length(spike_train_2)
            spike_train_2[start:stop] .= NaN
        end
    end

    pre_bins = 0
    if any(time_axis .<= 0)
        pre_bins = -1 * Integer(ceil(minimum(time_axis[time_axis .<= 0]) * (1.0/dt_new)))
    end
    post_bins = Integer(ceil(maximum(time_axis[time_axis .>= 0]) * (1.0/dt_new)))
    counts = get_cross_correlogram(spike_indices_1, spike_train_2, (pre_bins, post_bins))

    zero_index = findfirst(time_axis .== 0e-3)
    if set_zero && zero_index != nothing # This is correlation with ourself, set to 0
        counts[zero_index] = 0
    end
    return counts
end
get_cross_correlogram(neuron_1::Neuron, neuron_2::Neuron; kwargs...) = get_cross_correlogram(spike_indices(neuron_1), dt(neuron_1), spike_indices(neuron_2), dt(neuron_2); kwargs...)
get_auto_correlogram(neuron::Neuron; args...) = get_cross_correlogram(neuron, neuron; set_zero=true, args...)

"""
    plot_cross_correlogram(spike_indices_1, dt_1, spike_indices_2, dt_2; time_axis)

Plot the cross correllogram given the spike indices found for neuron 1 and neuron 2.
The two spike indices are allowed to have different time frames (i.e., 1 neuron could
be downsampled from a 40kHz sampling rate to 1ms intervals). The time bases are
passed in via the respective dt parameters which is 1/sampling rate (0.001 for a 1kHz
sampling rate).

"""
function plot_cross_correlogram(spike_indices_1::AbstractVector{<:Real}, dt_1::Real, spike_indices_2::AbstractVector{<:Real}, dt_2::Real;
        fig=nothing, time_axis::AbstractRange=-100e-3:1e-3:100e-3,
        normalize::Symbol=:rate, kwargs...)
    counts = get_cross_correlogram(spike_indices_1, dt_1, spike_indices_2, dt_2; time_axis=time_axis, kwargs...)
    if fig == nothing
        fig = figure()
    else
        sca(fig.gca())
    end
    if normalize == :rate
        #bar(time_axis * 1000, 1000 * counts, width=1, color="k")
        plot(time_axis * 1000, (1.0 / step(time_axis)) * counts)
        ylabel("Firing rate (Hz)")
    elseif normalize == :probability
        plot(time_axis * 1000, counts)
        ylabel("Probability")
    elseif normalize == :chance
        plot(time_axis * 1000, counts .* spike_probability_to_chance(spike_indices_2, dt_2, step(time_axis)))
        ylabel("Probability w.r.t. chance")
    else
        error("Unknown normalization parameter. Should be :rate or :probability")
    end
    xlim(time_axis[1] * 1000, time_axis[end] * 1000)
    xlabel("Time (ms)")
    return fig
end

"""
    plot_cross_correlogram(neuron_1, neuron_2)
    plot_cross_correlogram(neuron_1, spike_indices_2, dt_2)
    plot_cross_correlogram(spik_indices_1, dt_1, neuron_2)

Plots the cross correlogram for a pair of neurons.
"""
plot_cross_correlogram(neuron_1::Neuron, neuron_2::Neuron; args...) = plot_cross_correlogram(spike_indices(neuron_1), dt(neuron_1), spike_indices(neuron_2), dt(neuron_2); args...)
plot_cross_correlogram(neuron_1::Neuron, inds::AbstractVector{<:Real}, delta::Real; args...) = plot_cross_correlogram(spike_indices(neuron_1), dt(neuron_1), inds, delta; args...)
plot_cross_correlogram(inds::AbstractVector{<:Integer}, delta::Real, neuron_2::Neuron, ; args...) = plot_cross_correlogram(inds, delta, spike_indices(neuron_2), dt(neuron_2), ; args...)

"""
    plot_auto_correlogram(neuron; time_axis)

Plots the auto-correlogram for a given neuron versus time. This is identical
to calling plot_cross_correlogram with the same neurons on as neuron 1 and
neuron 2.
"""
plot_auto_correlogram(neuron::Neuron; args...) = plot_cross_correlogram(neuron, neuron; set_zero=true, args...)


"""
    plot_auto_correlogram(spike_indices, dt; time_axis)

Plots the auto-correlogram for a given neuron versus time given the
spike_indices and the dt (1/sampling_rate).
"""
plot_auto_correlogram(spike_inds::AbstractVector{<:Real}, delta::Real; args...) = plot_cross_correlogram(spike_inds, delta, spike_inds, delta; set_zero=true, args...)

"""
	get_cross_correlogram_vs_firing_rate(spike_indices, dt)

Computes an auto-correlogram given the spike indices and the
dt (1/sampling) across rates. This is essentially a three dimensional
auto-correlogram that shows firing regularity when the neuron is
firing at different firing rates.

The output is a 2D matrix where the first dimension is the firing rate
axis, specified by `firing_rate_axis` which can either be an integer specifying
the number of bins (quantiles) to divide the data into or a range (e.g., 0:10:100)
specifying the absolute firing rates. The second dimension is the time axis.

This function returns
the binning used for the first dimension (vector of length num_bins).
By default, we calculate the firing rate using a box-car smoothing
kernel, which provides a better estimate of the average firing rate.
This can be disabled by passing `smooth=nothing`.
"""
function get_cross_correlogram_vs_firing_rate(spike_indices_1::AbstractVector{<:Real}, dt_1::Real, spike_indices_2::AbstractVector{<:Real}, dt_2::Real;
        time_axis=-100e-3:1e-3:100e-3, firing_rate_axis::Union{AbstractRange, Integer}=10, smooth::Union{Nothing, Real}=250e-3, set_zero::Bool=false)
    # Round the new dt to the nearest microsecond
    dt_new = Integer(round(mean(diff(time_axis)) * 1E6)) ./ 1E6

    # Convert the spike indices to units of ms
    spike_indices_1 = convert.(Float64, ceil.(spike_indices_1 ./ (dt_new ./ dt_1)))
    spike_indices_2 = convert.(Float64, ceil.(spike_indices_2 ./ (dt_new ./ dt_2)))
    deleteat!(spike_indices_1, spike_indices_1 .< 1)
    deleteat!(spike_indices_2, spike_indices_2 .< 1)

    # Convert to spike trains
    len = Integer(ceil(max(maximum(spike_indices_1), maximum(spike_indices_2))))
    spike_train = zeros(Float64, len)
    for i = 1:length(spike_indices_2)
        if isnan(spike_indices_2[i]) || isinf(spike_indices_2[i])
            # Nan from previous non-nan spike to next nan-nan spike
            start_index = findlast((.! isnan.(spike_indices_2)) .& (1:length(spike_indices_2) .< i))
            stop_index = findfirst((.! isnan.(spike_indices_2)) .& (1:length(spike_indices_2) .> i))
            start_index = start_index === nothing ? 1 : Integer(spike_indices_2[start_index]) + 1
            stop_index = stop_index === nothing ? length(spike_train) : Integer(spike_indices_2[stop_index]) - 1
            spike_train[start_index:stop_index] .= NaN
        else
            spike_train[Integer(spike_indices_2[i])] += 1
        end
    end

    # Compute our firing rate using the inverse ISI method
    firing_rate = zeros(Float32, length(spike_train)) .* NaN
    for i = 1:length(spike_indices_2)-1
        current_firing_rate = 1.0 / ((spike_indices_2[i+1] - spike_indices_2[i]) * dt_new)
        if isnan(current_firing_rate) || isinf(current_firing_rate)
            continue
        end
        firing_rate[Integer(spike_indices_2[i]):Integer(spike_indices_2[i+1])] .= current_firing_rate
        if i == 1
            firing_rate[1:Integer(spike_indices_2[i])] .= current_firing_rate
        end
        if i == length(spike_indices_2) - 1
            firing_rate[Integer(spike_indices_2[i+1]):end] .= current_firing_rate
        end
    end

    # Smooth with the appropriate width filter (boxcar)
    if smooth != nothing
        # The below code uses a boxcar filter brute force
        # we replace with conv, which should be faster
        # smooth_indices = Integer(ceil((smooth / dt_new) / 2)) # Half of smooth width, centered on current bin
        # smoothed_firing_rate = similar(firing_rate)
        # for i = 1:length(smoothed_firing_rate)
        #     start = max(1, i - smooth_indices)
        #     stop = min(length(smoothed_firing_rate), i + smooth_indices)
        #     smoothed_firing_rate[i] = mean(firing_rate[start:stop])
        # end
        # firing_rate = smoothed_firing_rate
        smooth_indices = Integer(ceil(smooth / dt_new))
        half_smooth_indices = Integer(round(smooth_indices / 2))
        firing_rate = DSP.conv(ones(smooth_indices) ./ smooth_indices, firing_rate)[half_smooth_indices:half_smooth_indices + length(firing_rate)-1]
    end

    # Get quartiles of firing rates
    if isa(firing_rate_axis, Integer)
        x_axis = range(0, stop=1, length=firing_rate_axis+2)[2:end-1]
        rates = firing_rate[convert.(Integer, spike_indices_1[.! isnan.(spike_indices_1)])]
        firing_rate_axis = quantile(rates[.! isnan.(rates)], x_axis)
    end
    # This is essentially a histogram, were we count the number of events
    counts = zeros(length(firing_rate_axis), length(time_axis))
    times = zeros(length(firing_rate_axis))

    for (i, spike_index) = enumerate(spike_indices_1)
        # Construct a timeseries for neuron 2 based on the current spike index as the
        # zero point
        if isnan(spike_index) || isinf(spike_index)
            continue
        else
            spike_index = Integer(spike_index)
        end
        start = spike_index + Integer(ceil(time_axis[1] * (1.0/dt_new)))
        stop = start + length(time_axis) - 1
        if (start < 1) || (stop >= len)
            continue
        end
        current_firing_rate = firing_rate[spike_index]
        bin_number = argmin(abs.(firing_rate_axis .- current_firing_rate))
        # If the current firing rate exceeds the first last value in the firing rate axis
        # we skip it
        if bin_number == length(firing_rate_axis) && current_firing_rate > firing_rate_axis[bin_number] + (firing_rate_axis[end] - firing_rate_axis[end-1]) / 2.0
            continue
        end
        if bin_number == 1 && current_firing_rate < firing_rate_axis[1] - (firing_rate_axis[2] - firing_rate_axis[1]) / 2.0
            continue
        end
        bin_number = (bin_number == nothing ? length(firing_rate_axis) : bin_number)
        if any(isnan.(spike_train[start:stop]))
            continue
        end
        counts[bin_number, :] += spike_train[start:stop]
        times[bin_number] = times[bin_number] + 1
    end

    zero_index = findfirst(time_axis .== 0e-3)
    if zero_index != nothing && set_zero == true # This is correlation with ourself, set to nan
        counts[:, zero_index] .= 0
    end
    counts = counts ./ times
    return firing_rate_axis, counts
end
get_auto_correlogram_vs_firing_rate(spike_indices_1::AbstractVector{<:Real}, dt_1::Real; kwargs...) = get_cross_correlogram_vs_firing_rate(spike_indices_1, dt_1, spike_indices_1, dt_1; set_zero=true, kwargs...)
get_auto_correlogram_vs_firing_rate(neuron::Neuron; kwargs...) = get_auto_correlogram_vs_firing_rate(spike_indices(neuron), dt(neuron); kwargs...)

"""
	plot_auto_correlogram_vs_firing_rate(spike_indices_1, dt_1, ...)

Creates a surface plot of the auto correlogram across binned firing rates.
Refer to the documentation for `get_auto_correlogram_vs_firing_rate`
"""
function plot_cross_correlogram_vs_firing_rate(spike_indices_1::AbstractVector{<:Real}, dt_1::Real, spike_indices_2::AbstractVector{<:Real}, dt_2::Real; time_axis::AbstractRange=-100e-3:1e-3:100e-3, fig=nothing, remove_mean::Bool=false, kwargs...)
    bins, counts = get_cross_correlogram_vs_firing_rate(spike_indices_1, dt_1, spike_indices_2, dt_2; time_axis=time_axis, kwargs...)
    if fig == nothing
        fig = figure()
    else
        sca(fig.gca())
    end
    delta_t = mean(diff(time_axis))
    if remove_mean == true
        for i = 1:length(bins)
            counts[i, :] = counts[i, :] .- mean(counts[i, :])
        end
    end
	pcolormesh(time_axis * 1000, bins, (1.0 / delta_t) .* counts, shading="auto", linewidth=0, rasterized=true)
	xlabel("Time (ms)")
	ylabel("Firing rate (Hz)")
    return fig
end
plot_cross_correlogram_vs_firing_rate(neuron_1::Neuron, neuron_2::Neuron; args...) = plot_cross_correlogram_vs_firing_rate(spike_indices(neuron_1), dt(neuron_1), spike_indices(neuron_2), dt(neuron_2); args...)
plot_auto_correlogram_vs_firing_rate(spike_indices_1::AbstractVector{<:Real}, dt_1::Real; args...) = plot_cross_correlogram_vs_firing_rate(spike_indices_1, dt_1, spike_indices_1, dt_1, set_zero=true; args...)
plot_auto_correlogram_vs_firing_rate(neuron::Neuron; args...) = plot_auto_correlogram_vs_firing_rate(spike_indices(neuron), dt(neuron); args...)
