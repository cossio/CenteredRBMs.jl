function RestrictedBoltzmannMachines.pcd!(
    rbm::CenteredRBM,
    data::AbstractArray;

    batchsize::Int = 1,
    iters::Int = 1,

    optim::AbstractRule = Adam(), # a rule from Optimisers
    steps::Int = 1, # Monte-Carlo steps to update persistent chains

    # data point weights
    wts::Union{AbstractVector, Nothing} = nothing,

    # init fantasy chains
    vm = sample_from_inputs(rbm.visible, Falses(size(rbm.visible)..., batchsize)),

    moments = moments_from_samples(rbm.visible, data; wts),

    # damping to update hidden statistics
    hidden_offset_damping::Real = 1//100,

    # regularization
    l2_fields::Real = 0, # visible fields L2 regularization
    l1_weights::Real = 0, # weights L1 regularization
    l2_weights::Real = 0, # weights L2 regularization
    l2l1_weights::Real = 0, # weights L2/L1 regularization

    # gauge
    zerosum::Bool = true, # zerosum gauge for Potts layers
    rescale::Bool = true, # normalize weights to unit norm (for continuous hidden units only)

    callback = Returns(nothing)
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    isnothing(wts) || @assert size(data)[end] == length(wts)

    # inital centering from data
    center_from_data!(rbm, data)

    # gauge constraints
    zerosum && zerosum!(rbm)
    rescale && rescale_weights!(rbm)

    # define parameters for Optimiser
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w)
    state = setup(optim, ps)

    wts_mean = isnothing(wts) ? 1 : mean(wts)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize))
        # update fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        batch_weight = isnothing(wts) ? 1 : mean(wd) / wts_mean
        ∂ *= batch_weight

        # weight decay
        ∂regularize!(∂, rbm; l2_fields, l1_weights, l2_weights, l2l1_weights)

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        # centering
        offset_h_new = grad2ave(rbm.hidden, -∂d.hidden) # <h>_d from minibatch
        offset_h = (1 - hidden_offset_damping) * rbm.offset_h + hidden_offset_damping * offset_h_new
        center_hidden!(rbm, offset_h)

        # gauge constraints
        zerosum && zerosum!(rbm)
        rescale && rescale_weights!(rbm)


        callback(; rbm, optim, iter, vm, vd, wd)
    end
    return state, ps
end
