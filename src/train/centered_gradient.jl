"""
    pcdc!(rbm, data)

Trains the RBM on data using Persistent Contrastive divergence, with centered gradients.
See:

J. Melchior, A. Fischer, and L. Wiskott. JMLR 17.1 (2016): 3387-3447.
<http://jmlr.org/papers/v17/14-237.html>
"""
function pcdc!(rbm::RBM, data::AbstractArray;
    batchsize = 1,
    epochs = 1,
    optim = Flux.ADAM(),
    history::MVHistory = MVHistory(),
    wts = nothing,
    steps::Int = 1,
    center_h::Bool = true, # center hidden?
    damping::Real = true # running average for λh
)
    @assert size(data) == (size(rbm.visible)..., size(data)[end])
    @assert isnothing(wts) || _nobs(data) == _nobs(wts)

    # data statistics
    stats = RBMs.suffstats(rbm.visible, data; wts)
    avg_h = RBMs.batchmean(rbm.hidden, RBMs.mean_h_from_v(rbm, data); wts)

    # initialize fantasy chains by sampling visible layer
    vm = RBMs.transfer_sample(rbm.visible, falses(size(rbm.visible)..., batchsize))

    for epoch in 1:epochs
        batches = minibatches(data, wts; batchsize = batchsize)
        Δt = @elapsed for (vd, wd) in batches
            # update fantasy chains
            vm = RBMs.sample_v_from_v(rbm, vm; steps = steps)
            # compute centered gradients
            ∂ = ∂contrastive_divergence_centered(
                rbm, vd, vm; wd, stats, center_h, damping, avg_h
            )
            # update parameters using gradient
            RBMs.update!(optim, rbm, ∂)
            # store gradient norms
            push!(history, :∂, RBMs.gradnorms(∂))
        end
        push!(history, :epoch, epoch)
        push!(history, :Δt, Δt)
        @debug "epoch $epoch/$epochs ($(round(Δt, digits=2))s)"
    end

    return history
end

function ∂contrastive_divergence_centered(
    rbm::RBM, vd::AbstractArray, vm::AbstractArray;
    wd = nothing, wm = nothing,
    stats = RBMs.suffstats(rbm.visible, vd; wts = wd),
    center_h::Bool = true, damping::Real = 0.01, avg_h::AbstractArray
)
    ∂d = RBMs.∂free_energy(rbm, vd; wts = wd, stats)
    ∂m = RBMs.∂free_energy(rbm, vm; wts = wm)
    ∂ = RBMs.subtract_gradients(∂d, ∂m)

    # extract moment estimates from the gradients
    λv = grad2mean(visible(rbm), ∂d.visible)   # <v>_d, uses full data from sufficient_statistics
    λh = grad2mean(hidden(hidden), ∂d.hidden)  # <h>_d, uses minibatch

    # since <h>_d uses minibatches, we keep a running average
    @assert size(avg_h) == size(λh)
    λh .= avg_h .= (1 - damping) * avg_h .+ damping .* λh

    ∂c = center_gradients(rbm, ∂, λv, center_h * λh)
    return ∂c
end

function center_gradients(rbm::RBM, ∂::NamedTuple, λv::AbstractArray, λh::AbstractArray)
    @assert size(rbm.visible) == size(λv)
    @assert size(rbm.hidden) == size(λh)
    @assert size(∂.w) == size(rbm.w)

    ∂wmat = reshape(∂.w, length(rbm.visible), length(rbm.hidden))
    ∂cwmat = ∂wmat - vec(λv) * vec(∂.hidden.θ)' - vec(∂.visible.θ) * vec(λh)'
    ∂cw = reshape(∂cwmat, size(rbm.w))

    shift_v = reshape(∂cwmat  * vec(λh), size(rbm.visible))
    shift_h = reshape(∂cwmat' * vec(λv), size(rbm.hidden))
    ∂cv = center_gradients(rbm.visible, ∂.visible, shift_v)
    ∂ch = center_gradients(rbm.hidden,  ∂.hidden,  shift_h)

    return (visible = ∂cv, hidden = ∂ch, w = ∂cw)
end

function center_gradients(
    layer::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU},
    ∂::NamedTuple, λ::AbstractArray
)
    @assert size(layer) == size(∂.θ) == size(λ)
    return (∂..., θ = ∂.θ - λ)
end

function center_gradients(layer::dReLU, ∂::NamedTuple, λ::AbstractArray)
    @assert size(layer) == size(∂.θp) == size(∂.θn) == size(λ)
    return (θp = ∂.θp - λ, θn = ∂.θn - λ, γp = ∂.γp, γn = ∂.γn)
end

# get moments from layer gradients, e.g. <v> = -derivative w.r.t. θ
grad2mean(::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU}, ∂::NamedTuple) = -∂.θ
grad2mean(::dReLU, ∂::NamedTuple) = -(∂.θp + ∂.θn)
