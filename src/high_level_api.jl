export predict
"""
    predict(spike_recording, lfp_recording, neurons)

Predict the labels for each neuron given the spike and lfp recordings passed.

This function returns a vector of labels, confidence ratios, and highest probabilities.
"""
function predict(recording_spike_band::AbstractRecordingTimeseries, recording_lfp_band::AbstractRecordingTimeseries, neurons::AbstractVector{Neuron})
    @assert size(recording_spike_band, 1) == size(recording_lfp_band, 1)

    labels = Vector{AbstractString}(undef, length(neurons))
    confidence_ratios = zeros(length(neurons))
    probabilities = zeros(length(neurons))
    acg_time_axis = 1e-3:1e-3:250e-3 

    trained_models, classes = get_trained_models()

    ProgressMeter.@showprogress for i = 1:length(neurons)
        # Primary waveform
        _, primary_template, primary_channel = drift_shift_match(recording_spike_band, neurons[i], verbose=false)
        primary_template = preprocess_template(primary_template, 
            original_sampling_rate=sampling_rate(recording_spike_band), 
            output_sampling_rate=40000.0, peak_sign=:negative, normalize=true, clip_width=(1e-3, 3e-3))

        # Spike-triggered LFP
        lfp_template = get_template(recording_lfp_band, neurons[i], channels=[primary_channel], clip_width=(5e-3, 25e-3))
        lfp_template = preprocess_template(lfp_template, 
            original_sampling_rate=sampling_rate(recording_lfp_band), 
            output_sampling_rate=2500, normalize=true, clip_width=(5e-3, 10e-3), peak_sign=:negative, peak_range=(1e-3, 10e-3))

        # 2D ACG
        acg_2d = get_auto_correlogram(neurons[i], time_axis=acg_time_axis)

        # 3D ACG
        _, acg_3d = get_auto_correlogram_vs_firing_rate(neurons[i], time_axis=acg_time_axis)

        # Scale
        acg_2d = Float32.(acg_2d ./ (100 * step(acg_time_axis)))
        acg_3d = Float32.(acg_3d ./ (100 * step(acg_time_axis)))
        mlp_input = vcat(primary_template, lfp_template, acg_2d) 

        labels[i], confidence_ratios[i], current_probabilities = predict(trained_models, classes, acg_3d, mlp_input)
        probabilities[i] = current_probabilities[labels[i]]
    end

    return labels, confidence_ratios, probabilities
end

"""
    predict(wideband_recording, neurons)

Predict the labels for each neuron given the wideband recording. The recording
is appropriately filtered and preprocessed into spike and LFP bands before classification.
"""
function predict(wideband_recording::AbstractRecordingTimeseries, neurons::AbstractVector{Neuron})
    recording_spike_band = preprocess_spikeband(wideband_recording)
    recording_lfp_band = preprocess_lfpband(wideband_recording)
    return predict(recording_spike_band, recording_lfp_band, neurons)
end

predict(wideband_recording::AbstractRecordingTimeseries, neuron::Neuron) = predict(wideband_recording, [neuron])