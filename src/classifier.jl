export predict
export get_trained_models

"""
    ConvolutionalNetwork

A CCN used for image based inputs
"""
struct ConvolutionalMLPNetwork
    conv_model 
    mlp_model
end
function (m::ConvolutionalMLPNetwork)(image_data::AbstractArray{<:Float32}, flat_data::AbstractArray{<:Float32}) 
    return m.mlp_model(vcat(m.conv_model(image_data), flat_data))
end
function (m::ConvolutionalMLPNetwork)(data::Tuple{<:AbstractArray{<:Float32}, <:AbstractArray{<:Float32}})
    return m(data[1], data[2])
end

"""
    convolutional_mlp_network(mlp_input_length, num_outputs)

Construct a convolution neural network with an MLP input. T


The last layer is flattened, resulting in a vector output
"""
function convolutional_mlp_network(mlp_input_length::Integer, num_outputs::Integer; num_hidden_units::Integer=100)
    image_model = Chain(
        Conv((2, 10), 1 => 32, pad=SamePad(), relu),
        MeanPool((2, 10)),
        BatchNorm(32),
        Conv((2, 20), 32 => 64, pad=SamePad(), relu),
        MeanPool((2, 20)),
        Flux.flatten,
        );
    
        mlp_model = Chain(
            Dense(128 + mlp_input_length => num_hidden_units), # 256
            BatchNorm(num_hidden_units, relu),
            Dropout(0.1),
            Dense(num_hidden_units => num_outputs)
        )

    return ConvolutionalMLPNetwork(image_model, mlp_model)
end

function _ask_yes_no(question::AbstractString)
    while true
        print(question, " (y/n): ")
        answer = readline()
        if answer == "y" || answer == "yes"
            return true
        elseif answer == "n" || answer == "no"
            return false
        else
            println("Invalid input. Please enter 'y', 'yes', 'n', or 'no'.")
        end
    end
end

"""
    get_trained_models()

Return the trained models from the OSF repository. 

"""
function get_trained_models(mlp_input_length::Integer=449)
    classes = ["pc", "cs", "putative_mli", "putative_golgi", "putative_ubc", "putative_mf"];
    # This is the path to the Artifacts.toml we will manipulate
    artifact_toml = joinpath(@__DIR__, "..", "Artifacts.toml")
    # Query the `Artifacts.toml` file for the hash bound to the name "iris"
    # (returns `nothing` if no such binding exists)
    artifact_hash = Pkg.Artifacts.artifact_hash("acg_acg3d_waveform_lfp_classifier", artifact_toml)

    if artifact_hash === nothing || ! Pkg.Artifacts.artifact_exists(artifact_hash)
        response = _ask_yes_no("This will download the models which are quite large, are you sure you with to continue? ")
        if response == false
            return
        end
    end
    classifier_path = Pkg.Artifacts.artifact"acg_acg3d_waveform_lfp_classifier"
    trained_models = JLD2.load(joinpath(classifier_path, "trained_models_acg_acg3d_waveform_lfp.jld2"), "models")

    models = Vector{ConvolutionalMLPNetwork}(undef, 0)
    for i = 1:length(trained_models)
        m = convolutional_mlp_network(mlp_input_length, length(classes))
        Flux.loadmodel!(m, trained_models[1])
        push!(models, m)
    end
    return models, classes
end

"""
    predict(models, classes, 3d_acg, mlp_input)

Classify the presented neuron using the models passed via `models`. The
output of this function is the mostly label (as a string), the confidence
ratio, and a dictionary containing the probabilities of this neuron 
being each of the predicted classes.
"""
function predict(models::AbstractVector{ConvolutionalMLPNetwork}, classes::AbstractVector{<:AbstractString}, acg_input::AbstractMatrix{<:Real}, mlp_input::AbstractVector{<:Real})
    predictions = zeros(length(models), length(classes))
    acg_input = convert.(Float32, reshape(acg_input, size(acg_input)..., 1, 1))
    mlp_input = convert.(Float32, mlp_input)
    for (i, model) = enumerate(models)
        predictions[i, :] = Flux.softmax(model(acg_input, mlp_input))
    end
    predictions = dropdims(mean(predictions, dims=1), dims=1)
    I = argmax(predictions)
    sorted_predictions = sort(predictions)
    confidence_ratio = sorted_predictions[end] ./ sorted_predictions[end-1]
    return classes[I], confidence_ratio, Dict(zip(classes, predictions))
end