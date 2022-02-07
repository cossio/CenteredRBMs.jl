#=
# MNIST

We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

import CairoMakie
using Random: bitrand
using Statistics: mean
import MLDatasets
import Flux
import RestrictedBoltzmannMachines as RBMs
import CenteredRBMs
using RestrictedBoltzmannMachines: BinaryRBM
nothing #hide

#=
Useful function to plot MNIST digits.
=#

"""
    imggrid(A)

Given a four dimensional tensor `A` of size `(width, height, ncols, nrows)`
containing `width x height` images in a grid of `nrows x ncols`, this returns
a matrix of size `(width * ncols, height * nrows)`, that can be plotted in a heatmap
to display all images.
"""
function imggrid(A::AbstractArray{<:Any,4})
    reshape(permutedims(A, (1,3,2,4)), size(A,1)*size(A,3), size(A,2)*size(A,4))
end

#=
Now load the dataset.
=#

Float = Float32
selected_digits = (0, 1) # only work with these digits for speed
train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float}(train_x[:, :, train_y .∈ Ref(selected_digits)] .> 0.5)
nothing #hide

#=
Initialize and train a centered RBM
Notice how we pass the `Float` type, to set the parameter type of the layers and weights
in the RBM.
=#

rbm = CenteredRBMs.center(BinaryRBM(Float, (28,28), 400))
RBMs.initialize!(rbm, train_x) # centers from data
batchsize = 256
optim = Flux.ADAM()
vm = bitrand(28, 28, batchsize) # fantasy chains
history = MVHistory()
push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))
@time for epoch in 1:100
    RBMs.pcd!(rbm, train_x; epochs=5, vm, history, batchsize, optim)
    # track pseudolikelihood every 5 epochs
    push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))
end
rbm = CenteredRBMs.uncenter(centered_rbm)
nothing #hide

#=
Plot of log-pseudolikelihood during learning.
Note that this shows the pseudolikelihood of the train data.
=#

CairoMakie.lines(get(history, :lpl)...)

#=
Now we do the Gibbs sampling to generate the RBM digits.
=#

@elapsed fantasy_x .= RBMs.sample_v_from_v(rbm, bitrand(28,27,nrows*ncols); steps=10000)

#=
Plot the resulting samples.
=#

fig = CairoMakie.Figure(resolution=(40ncols, 40nrows))
ax = CairoMakie.Axis(fig[1,1], yreversed=true)
CairoMakie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
CairoMakie.hidedecorations!(ax)
CairoMakie.hidespines!(ax)
fig
