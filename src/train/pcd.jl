function pcd!(
    rbm::CenteredRBM,
    data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optim = Adam(),
    wts = nothing,
    steps::Int = 1,
    vm = sample_from_inputs(rbm.visible, falses(size(rbm.visible)..., batchsize)),
    moments = moments_from_samples(rbm.visible, data; wts),
    hidden_offset_damping::Real = 0.01,
    callback = Returns(nothing)
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    center_from_data!(rbm, data)
    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize)
        for (batch_idx, (vd, wd)) in enumerate(batches)
            vm .= sample_v_from_v(rbm, vm; steps)
            ∂ = ∂contrastive_divergence_and_center!(rbm, vd, vm; wd, moments, hidden_offset_damping)
            RBMs.update!(rbm, RBMs.update!(∂, rbm, optim))
            callback(; rbm, optim, epoch, batch_idx, vm, vd, wd)
        end
        center_from_data!(rbm, data) # full center each epoch
    end
    return rbm
end

function ∂contrastive_divergence_and_center!(
    rbm::CenteredRBM, vd::AbstractArray, vm::AbstractArray;
    wd = nothing, wm = nothing,
    moments = moments_from_samples(rbm.visible, vd; wts = wd),
    hidden_offset_damping::Real = 0.01
)
    ∂d = ∂free_energy(rbm, vd; wts = wd, moments)
    ∂m = ∂free_energy(rbm, vm; wts = wm)
    ∂ = ∂d - ∂m
    offset_h_new = grad2ave(rbm.hidden, ∂d.hidden) # <h>_d from minibatch
    offset_h = (1 - hidden_offset_damping) * rbm.offset_h + hidden_offset_damping * offset_h_new
    center_hidden!(rbm, offset_h)
    return ∂
end

RBMs.update!(rbm::CenteredRBM, ∂::∂RBM) = RBMs.update!(RBM(rbm), ∂)
RBMs.update!(∂::∂RBM, rbm::CenteredRBM, optim) = RBMs.update!(∂, RBM(rbm), optim)
