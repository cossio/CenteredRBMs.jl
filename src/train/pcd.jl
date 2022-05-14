function RBMs.pcd!(rbm::CenteredRBM, data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optim = Flux.ADAM(),
    history::MVHistory = MVHistory(),
    wts = nothing,
    steps::Int = 1,
    vm = RBMs.sample_from_inputs(rbm.visible, falses(size(rbm.visible)..., batchsize)),
    stats = RBMs.suffstats(rbm.visible, data; wts),
    hidden_offset_damping::Real = 0.01
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)
    center_from_data!(rbm, data)
    for epoch in 1:epochs
        batches = RBMs.minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # update fantasy chains
            vm .= RBMs.sample_v_from_v(rbm, vm; steps = steps)
            # get gradient, and center! the rbm exploiting moment data in gradients
            ∂ = ∂contrastive_divergence_and_center!(rbm, vd, vm; wd, stats, hidden_offset_damping)
            # save gradient norms
            push!(history, :∂, RBMs.gradnorms(∂))
            # update parameters
            RBMs.update!(rbm, RBMs.update!(∂, rbm, optim))
            # save parameter update steps
            push!(history, :Δ, RBMs.gradnorms(∂))
        end
        # full center each epoch
        center_from_data!(rbm, data)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        push!(history, :vm, copy(vm))
        @debug "epoch $epoch/$epochs ($(round(Δt, digits=2))s)"
    end
    return history
end

function ∂contrastive_divergence_and_center!(
    rbm::CenteredRBM, vd::AbstractArray, vm::AbstractArray;
    wd = nothing, wm = nothing,
    stats = RBMs.suffstats(rbm.visible, vd; wts = wd),
    hidden_offset_damping::Real = 0.01
)
    ∂d = RBMs.∂free_energy(rbm, vd; wts = wd, stats)
    ∂m = RBMs.∂free_energy(rbm, vm; wts = wm)
    ∂ = RBMs.subtract_gradients(∂d, ∂m)
    # extract moment estimates from the gradients
    offset_h_new = grad2mean(rbm.hidden, ∂d.hidden)  # <h>_d from minibatch
    # since <h>_d uses minibatches, we keep a running average
    offset_h = hidden_offset_damping * rbm.offset_h + (1 - hidden_offset_damping) * offset_h_new
    center_hidden!(rbm, offset_h)
    return ∂
end

function RBMs.update!(rbm::CenteredRBM, ∂::NamedTuple)
    RBMs.update!(RBM(rbm), ∂)
    return rbm
end

RBMs.update!(∂::NamedTuple, rbm::CenteredRBM, optim) = RBMs.update!(∂, RBM(rbm), optim)
