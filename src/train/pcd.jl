function RBMs.pcd!(
    centered_rbm::CenteredRBM,
    data::AbstractArray;
    batchsize::Int = 1,
    epochs::Int = 1,
    optimizer = Flux.ADAM(),
    history::ValueHistories.MVHistory = ValueHistories.MVHistory(),
    wts = nothing,
    steps::Int = 1,
    vm::AbstractArray = RBMs.transfer_sample(centered_rbm.visible, falses(size(centered_rbm.visible)..., batchsize)),
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
            # update fantasy chains
            vm .= RBMs.sample_v_from_v(centered_rbm, vm; steps = steps)
            # update offsets
            if center_h
                hd = RBMs.mean_h_from_v(centered_rbm, vd)
                offset_h_new = RBMs.batchmean(centered_rbm.hidden, hd; wts=wd)
                offset_h_damped = damping * centered_rbm.offset_h + (1 - damping) * offset_h_new
                center_hidden!(centered_rbm, offset_h_damped)
            end
            # compute contrastive divergence gradient
            ∂ = RBMs.∂contrastive_divergence(centered_rbm, vd, vm; wd, stats)
            # update parameters using gradient
            RBMs.update!(optimizer, centered_rbm, ∂)
            # store gradient norms
            push!(history, :∂, RBMs.gradnorms(∂))
        end

        lpl = RBMs.wmean(RBMs.log_pseudolikelihood(uncenter(centered_rbm), data); wts)
        push!(history, :lpl, lpl)
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        push!(history, :vm, copy(vm))

        Δt_ = round(Δt, digits=2)
        lpl_ = round(lpl, digits=2)
        @debug "epoch $epoch/$epochs ($(Δt_)s), log(PL)=$lpl_"
    end

    return history
end
