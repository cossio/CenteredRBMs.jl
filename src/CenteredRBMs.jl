module CenteredRBMs
    import Random
    import Statistics
    using Optimisers: AbstractRule, setup, update!, Adam
    using FillArrays: Zeros, Falses
    using EllipsisNotation: (..)
    #using CudaRBMs: gpu, cpu
    import CudaRBMs
    import RestrictedBoltzmannMachines as RBMs
    using RestrictedBoltzmannMachines: RBM, AbstractLayer,
        Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU,
        sample_v_from_v_once, sample_h_from_h_once, sample_from_inputs,
        sample_v_from_v, sample_v_from_h, sample_h_from_v,
        energy, interaction_energy, cgf, moments_from_samples,
        inputs_h_from_v, inputs_v_from_h, mean_from_inputs,
        ∂free_energy, ∂cgfs, ∂energy_from_moments, ∂RBM, ∂interaction_energy,
        BinaryRBM, infinite_minibatches, wmean, log_pseudolikelihood

    include("centered_rbm.jl")
    include("centering.jl")
    include("binary.jl")
    include("layers.jl")
    include("from_grad.jl")
    include("sampling.jl")
    include("pcd.jl")
    include("initialization.jl")
    #include("gpu.jl")
end
