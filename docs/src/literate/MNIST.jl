#=
# MNIST

We begin by importing the required packages.
We load MNIST via the MLDatasets.jl package.
=#

using CairoMakie, Statistics
import MLDatasets, Flux
import RestrictedBoltzmannMachines as RBMs
import CenteredRBMs
using RestrictedBoltzmannMachines: BinaryRBM
using CenteredRBMs: BinaryCenteredRBM
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
Let's visualize some random digits.
=#

nrows, ncols = 10, 15
fig = CairoMakie.Figure(resolution=(40ncols, 40nrows))
ax = CairoMakie.Axis(fig[1,1], yreversed=true)
digits = MLDatasets.MNIST.traintensor()
digits = digits[:,:,rand(1:size(digits,3), nrows * ncols)]
digits = reshape(digits, 28, 28, ncols, nrows)
CairoMakie.image!(ax, imggrid(digits), colorrange=(1,0))
CairoMakie.hidedecorations!(ax)
CairoMakie.hidespines!(ax)
fig

#=
Now load the full dataset.
=#

train_x, train_y = MLDatasets.MNIST.traindata()
tests_x, tests_y = MLDatasets.MNIST.testdata()
nothing #hide

#=
`train_x`, `tests_x` contain the digit images, while
`train_y`, `tests_y` contain the labels.
We will train an RBM with binary (0,1) visible and hidden units.
Therefore we binarize the data first.
In addition, we restrict our attention to `0,1` digits only,
so that training and so on are faster.
=#

Float = Float32
selected_digits = (0, 1)
train_x = Array{Float}(train_x[:, :, train_y .∈ Ref(selected_digits)] .> 0.5)
tests_x = Array{Float}(tests_x[:, :, tests_y .∈ Ref(selected_digits)] .> 0.5)
train_y = train_y[train_y .∈ Ref(selected_digits)]
tests_y = tests_y[tests_y .∈ Ref(selected_digits)]
train_nsamples = length(train_y)
tests_nsamples = length(tests_y)
(train_nsamples, tests_nsamples)

#=
The original binarized `train_x` and `tests_x` are `BitArray`s.
Though convenient in terms of memory space, these are very slow in linear algebra.
Since we frequently multiply data configurations times the weights of our RBM,
we want to speed this up.
So we convert to floats, which have much faster matrix multiplies thanks to BLAS.
We will use `Float32` here.
To hit BLAS, this must be consistent with the types we use in the parameters of the RBM
below.
=#

#=
Plot some examples of the binarized data.
=#

nrows, ncols = 10, 15
fig = CairoMakie.Figure(resolution=(40ncols, 40nrows))
ax = CairoMakie.Axis(fig[1,1], yreversed=true)
digits = reshape(train_x[:, :, rand(1:size(train_x,3), nrows * ncols)], 28, 28, ncols, nrows)
CairoMakie.image!(ax, imggrid(digits), colorrange=(1,0))
CairoMakie.hidedecorations!(ax)
CairoMakie.hidespines!(ax)
fig

#=
Initialize and train a centered RBM
Notice how we pass the `Float` type, to set the parameter type of the layers and weights
in the RBM.
=#
rbm = BinaryRBM(zeros(Float,28,28), zeros(Float,400), zeros(Float,28,28,400))
RBMs.initialize!(rbm, train_x)
centered_rbm = CenteredRBMs.center(rbm)
CenteredRBMs.center_from_data!(centered_rbm, train_x)
@time history = RBMs.pcd!(centered_rbm, train_x; epochs=500, batchsize=256)
rbm = CenteredRBMs.uncenter(centered_rbm)
nothing #hide

#=
Plot of log-pseudolikelihood during learning.
Note that this shows the pseudolikelihood of the train data.
=#

CairoMakie.lines(get(history, :lpl)...)

#=
Now let's generate some random RBM samples.
First, we select random data digits to be initial conditions for the Gibbs sampling, and
let's plot them.
=#

nrows, ncols = 10, 15
fantasy_x = train_x[:, :, rand(1:train_nsamples, nrows * ncols)]
fig = CairoMakie.Figure(resolution=(40ncols, 40nrows))
ax = CairoMakie.Axis(fig[1,1], yreversed=true)
CairoMakie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
CairoMakie.hidedecorations!(ax)
CairoMakie.hidespines!(ax)
fig

#=
Now we do the Gibbs sampling to generate the RBM digits.
=#

@elapsed fantasy_x .= RBMs.sample_v_from_v(rbm, fantasy_x; steps=10000)

#=
Plot the resulting samples.
=#

fig = CairoMakie.Figure(resolution=(40ncols, 40nrows))
ax = CairoMakie.Axis(fig[1,1], yreversed=true)
CairoMakie.image!(ax, imggrid(reshape(fantasy_x, 28, 28, ncols, nrows)), colorrange=(1,0))
CairoMakie.hidedecorations!(ax)
CairoMakie.hidespines!(ax)
fig
