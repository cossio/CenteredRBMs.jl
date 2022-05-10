module CenteredRBMs
    import Random
    import Statistics
    import LinearAlgebra
    import ValueHistories
    import Flux
    import FillArrays
    using ValueHistories: MVHistory
    import RestrictedBoltzmannMachines as RBMs
    using RestrictedBoltzmannMachines: RBM, AbstractLayer, visible, hidden, weights
    using RestrictedBoltzmannMachines: Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU
    using RestrictedBoltzmannMachines: gradnorms, update!, minibatches

    include("centered_rbm.jl")
    include("centering.jl")
    include("binary.jl")
    include("layers.jl")
    include("sampling.jl")
    include("train/pcd.jl")
    include("train/centered_gradient.jl")
    include("train/initialization.jl")
end
