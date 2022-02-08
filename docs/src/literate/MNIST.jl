#=
# MNIST

We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

import Makie
import CairoMakie
using Random: bitrand
using Statistics: mean
import MLDatasets
import Flux
import RestrictedBoltzmannMachines as RBMs
using ValueHistories: MVHistory
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
Load the dataset.
=#

Float = Float32
selected_digits = (0, 1) # only work with these digits for speed
train_x, train_y = MLDatasets.MNIST.traindata()
train_x = Array{Float}(train_x[:, :, train_y .∈ Ref(selected_digits)] .> 0.5)
size(train_x, 3) # number of train samples

#=
Initialize and train a centered RBM
=#

rbm_c = CenteredRBMs.center(BinaryRBM(Float, (28,28), 400))
RBMs.initialize!(rbm_c, train_x) # centers from data
batchsize = 256
optim = Flux.ADAM()
vm = bitrand(28, 28, batchsize) # fantasy chains
history_c = MVHistory()
push!(history_c, :lpl, mean(RBMs.log_pseudolikelihood(rbm_c, train_x)))
@time for epoch in 1:100 # track pseudolikelihood every 5 epochs
    RBMs.pcd!(rbm_c, train_x; epochs=5, vm, history=history_c, batchsize, optim)
    push!(history_c, :lpl, mean(RBMs.log_pseudolikelihood(rbm_c, train_x)))
end
rbm_c = CenteredRBMs.uncenter(rbm_c) # equivalent RBM without offsets
nothing #hide

#=
For comparison, we also train a normal RBM.
=#

rbm = BinaryRBM(Float, (28,28), 400)
RBMs.initialize!(rbm, train_x)
vm = bitrand(28, 28, batchsize)
history = MVHistory()
push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))
@time for epoch in 1:100 # track pseudolikelihood every 5 epochs
    RBMs.pcd!(rbm, train_x; epochs=5, vm, history, batchsize, optim)
    push!(history, :lpl, mean(RBMs.log_pseudolikelihood(rbm, train_x)))
end
nothing #hide

# Plot log-pseudolikelihood of train data during learning.

fig = Makie.Figure(resolution=(600, 300))
ax = Makie.Axis(fig[1,1], xlabel="epochs", ylabel="pseudolikelihood")
Makie.lines!(ax, get(history, :lpl)..., label="normal")
Makie.lines!(ax, get(history_c, :lpl)..., label="centered")
Makie.axislegend(ax, position=:rb)

# Seconds per epoch.

fig = Makie.Figure(resolution=(600, 300))
ax = Makie.Axis(fig[1,1], xlabel="epochs", ylabel="seconds")
Makie.lines!(ax, get(history, :Δt)..., label="normal")
Makie.lines!(ax, get(history_c, :Δt)..., label="centered")
Makie.axislegend(ax, position=:rb)
fig

# Log-pseudolikelihood vs. computation time instead of epoch count.

fig = Makie.Figure(resolution=(600, 300))
ax = Makie.Axis(fig[1,1], xlabel="seconds", ylabel="pseudolikelihood")
Makie.lines!(ax, cumsum(get(history, :Δt)[2]), get(history, :lpl)[2], label="normal")
Makie.lines!(ax, cumsum(get(history_c, :Δt)[2]), get(history_c, :lpl)[2], label="centered")
Makie.axislegend(ax, position=:rb)
fig

# Now we do the Gibbs sampling to generate RBM digits.

nrows, ncols = 10, 15
@time fantasy_x_c = RBMs.sample_v_from_v(rbm_c, bitrand(28,28,nrows*ncols); steps=10000)
@time fantasy_x = RBMs.sample_v_from_v(rbm, bitrand(28,28,nrows*ncols); steps=10000)
nothing #hide

# Plot the resulting samples.

# Normal RBM.

ax = Makie.Axis(fig[1,1], yreversed=true)
Makie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)
fig

# Centered RBM.

fig = Makie.Figure(resolution=(40ncols, 40nrows))
ax = Makie.Axis(fig[1,1], yreversed=true)
Makie.image!(ax, imggrid(reshape(fantasy_x_c, 28, 28, ncols, nrows)), colorrange=(1,0))
Makie.hidedecorations!(ax)
Makie.hidespines!(ax)
fig
