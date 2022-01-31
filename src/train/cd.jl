function RBMs.cd!(
    centered_rbm::CenteredRBM,
    data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optimizer = Flux.ADAM(),
    history::ValueHistories.MVHistory = ValueHistories.MVHistory(),
    wts = nothing,
    steps::Int = 1,
    center_h::Bool = true,
    center_v::Bool = true,
    damping::Real = 0.01,
)
    @assert size(data) == (size(centered_rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    @assert 0 ≤ damping ≤ 1
    center_v && center_visible_from_data!(centered_rbm, data)
    center_h && center_hidden_from_data!(centered_rbm, data)

    stats = RBMs.sufficient_statistics(centered_rbm.visible, data; wts)

    for epoch in 1:epochs
        batches = RBMs.minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # fantasy particles
            vm = RBMs.sample_v_from_v(centered_rbm, vd; steps = steps)
            # update offsets
            if center_h
                hd = RBMs.mean_v_from_h(centered_rbm, vd)
                offset_h_new = RBMs.batchmean(centered_rbm.hidden, hd; wts=wd)
                offset_h_damped = damping * centered_rbm.offset_h + (1 - damping) * offset_h_new
                center_hidden!(centered_rbm, offset_h_damped)
            end
            # compute gradients
            ∂ = RBMs.∂contrastive_divergence(centered_rbm, vd, vm; wd = wd, wm = wd, stats)
            # update parameters with gradients
            RBMs.update!(optimizer, centered_rbm, ∂)
            # store gradient norms
            push!(history, :∂, RBMs.gradnorms(∂))
        end

        lpl = RBMs.wmean(RBMs.log_pseudolikelihood(uncenter(centered_rbm), data); wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)

        Δt_ = round(Δt, digits=2)
        lpl_ = round(lpl, digits=2)
        @debug "epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_"
    end
    return history
end

function RBMs.contrastive_divergence(
    centered_rbm::CenteredRBM, vd::AbstractArray, vm::AbstractArray; wd = nothing, wm = nothing
)
    Fd = RBMs.mean_free_energy(centered_rbm, vd; wts=wd)::Number
    Fm = RBMs.mean_free_energy(centered_rbm, vm; wts=wm)::Number
    return Fd - Fm
end

function RBMs.mean_free_energy(centered_rbm::CenteredRBM, v::AbstractArray; wts = nothing)
    @assert size(centered_rbm.visible) == size(v)[1:ndims(centered_rbm.visible)]
    F = RBMs.free_energy(centered_rbm, v)
    if ndims(centered_rbm.visible) == ndims(v)
        return F::Number
    else
        @assert size(F) == RBMs.batchsize(centered_rbm.visible, v)
        return RBMs.wmean(F; wts)
    end
end

function RBMs.∂contrastive_divergence(
    centered_rbm::CenteredRBM, vd::AbstractArray, vm::AbstractArray; wd = nothing, wm = nothing,
    stats = RBMs.sufficient_statistics(centered_rbm.visible, vd; wts = wd)
)
    ∂d = RBMs.∂free_energy(centered_rbm, vd; wts = wd, stats)
    ∂m = RBMs.∂free_energy(centered_rbm, vm; wts = wm)
    return RBMs.subtract_gradients(∂d, ∂m)
end

function RBMs.update!(optimizer, centered_rbm::CenteredRBM, ∂::NamedTuple)
    RBMs.update!(optimizer, centered_rbm.w, ∂.w)
    RBMs.update!(optimizer, centered_rbm.visible, ∂.visible)
    RBMs.update!(optimizer, centered_rbm.hidden, ∂.hidden)
end
