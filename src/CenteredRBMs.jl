module CenteredRBMs
    import Random
    import Statistics
    import LinearAlgebra
    import ValueHistories
    import Flux
    import FillArrays
    import RestrictedBoltzmannMachines as RBMs
    using RestrictedBoltzmannMachines: RBM, AbstractLayer
    using RestrictedBoltzmannMachines: Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU

    include("centered_rbm.jl")
    include("centering.jl")
    include("binary_centered_rbm.jl")
    include("layers.jl")
    include("train/cd.jl")
    include("train/pcd.jl")
    include("train/centered_gradient.jl")
end
