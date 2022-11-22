function pcd!(
    rbm::CenteredRBM,
    data::AbstractArray;

    batchsize::Int = 1,
    iters::Int = 1,

    optim::AbstractRule = Adam(), # a rule from Optimisers
    steps::Int = 1, # Monte-Carlo steps to update persistent chains

    # data point weights
    wts::Union{AbstractVector, Nothing} = nothing,

    # init fantasy chains
    vm = sample_from_inputs(rbm.visible, falses(size(rbm.visible)..., batchsize)),

    moments = moments_from_samples(rbm.visible, data; wts),

    # damping to update hidden statistics
    hidden_offset_damping::Real = 0.01,

    callback = Returns(nothing)
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    isnothing(wts) || @assert size(data)[end] == length(wts)

    # inital centering from data
    center_from_data!(rbm, data)

    # define parameters for Optimiser
    ps = (; visible = rbm.visible.par, hidden = rbm.hidden.par, w = rbm.w)
    state = setup(optim, ps)

    for (iter, (vd, wd)) in zip(1:iters, infinite_minibatches(data, wts; batchsize))
        # update fantasy chains
        vm .= sample_v_from_v(rbm, vm; steps)

        # compute gradient
        ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
        ∂m = ∂free_energy(rbm, vm)
        ∂ = ∂d - ∂m

        # feed gradient to Optimiser rule
        gs = (; visible = ∂.visible, hidden = ∂.hidden, w = ∂.w)
        state, ps = update!(state, ps, gs)

        # centering
        offset_h_new = grad2ave(rbm.hidden, -∂d.hidden) # <h>_d from minibatch
        offset_h = (1 - hidden_offset_damping) * rbm.offset_h + hidden_offset_damping * offset_h_new
        center_hidden!(rbm, offset_h)

        callback(; rbm, optim, iter, vm, vd, wd)

        # # full center after ending each epoch
        # center_from_data!(rbm, data)
    end
    return rbm, state
end
