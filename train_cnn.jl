include("structures.jl")
include("graph.jl")
include("forward_pass.jl")
include("backward_pass.jl")
include("operators.jl")
include("broadcast_operators.jl")
include("convolution.jl")

using Flux: onehotbatch, DataLoader, glorot_uniform
using MLDatasets: MNIST

function dense(w, x, activation)
    return activation(w * x)
end

function logitcrossentropy(y, ŷ)
    softmax_ŷ = softmax(ŷ)
    return -sum(y .* log.(softmax_ŷ .+ Constant(eps(Float64))))
end


function cnn(w, x, activation)
    return activation(conv(x, w))
end

wc = Variable{Array{Float64,4}}(Float64.(glorot_uniform(3, 3, 1, 6))::Array{Float64,4}, name="wc")
wd = Variable{Array{Float64,2}}(Float64.(glorot_uniform(84, 1014))::Array{Float64,2}, name="wd")
wo = Variable{Array{Float64,2}}(Float64.(glorot_uniform(10, 84))::Array{Float64,2}, name="wo")

x = Variable{Array{Float64,4}}(zeros(28, 28, 1, 1)::Array{Float64,4}, name="x")
y = Variable{Array{Float64,2}}(zeros(10, 1)::Array{Float64,2}, name="y")


function net()

    cx = cnn(wc, x, relu)
    cx.name = "cx"
    mx = maxpool(cx)
    mx.name = "mx"
    fx = flatten(mx)
    dx = dense(wd, fx, relu)
    dx.name = "dx"

    ŷ = dense(wo, dx, identity)
    ŷ.name = "ŷ"

    E = logitcrossentropy(y, ŷ)
    E.name = "loss"

    return topological_sort(E), ŷ, E
end

graph, ŷ, E = net()

function loader(data; batchsize::Int=1)
    x4dim = reshape(data.features, 28, 28, 1, :)
    yhot = onehotbatch(data.targets, 0:9)
    DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

settings = (;
    eta=1e-2,
    epochs=3,
    batchsize=100,
)


function train_model_and_count_acc(network, train_data, test_data)

    for epoch ∈ 1:settings.epochs
        println("Epoch $epoch :")

        @time train_model(network, train_data)

        acc_test = count_accuracy(test_data, network)

        println("Accurancy: $acc_test")
    end
end

function train_model(network, train_data)
    gradients_in_batch = Dict{String,Any}()

    for batch ∈ loader(train_data, batchsize=settings.batchsize)
        images = batch[1]
        labels = batch[2]
        for i in 1:settings.batchsize
            img = reshape(images[:, :, :, i], 28, 28, 1, 1)
            label = reshape(Float64.(labels[:, i]), 10, 1)

            x.output = img
            y.output = label

            learn(network)

            for w in (wd, wc, wo)
                if haskey(gradients_in_batch, w.name)
                    push!(gradients_in_batch[w.name], w.gradient)
                else
                    gradients_in_batch[w.name] = []
                    push!(gradients_in_batch[w.name], w.gradient)
                end
            end
        end

        for w in (wd, wc, wo)

            gradients = gradients_in_batch[w.name]
            sum_of_gradients = zeros(size(gradients[1]))

            for gradient in gradients
                sum_of_gradients .+= gradient
            end

            mean_gradients = sum_of_gradients ./ size(gradients)
            w.output .-= settings.eta .* mean_gradients

        end
        empty!(gradients_in_batch)
    end
end

function learn(network)
    forward!(network)
    backward!(network)
end

function count_accuracy(data, network)
    all_elems = 0
    positive_prediction = 0
    for sample in data
        img = reshape(sample.features, 28, 28, 1, :)
        label = sample.targets
        x.output = img
        forward!(network)
        prediction = get_predicted_number(ŷ.output)

        if label == prediction
            positive_prediction += 1
        end
        all_elems += 1
    end
    return positive_prediction / all_elems
end

function get_predicted_number(y_pred)
    index = argmax(y_pred)[1] - 1
    return index
end

train_data = MNIST(split=:train)
test_data = MNIST(split=:test)

function train_model_MNIST()
    train_model_and_count_acc(graph, train_data, test_data)
end

# train_model_MNIST()