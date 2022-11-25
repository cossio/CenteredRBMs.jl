struct CenteredRBM{V,H,W,Ov,Oh}
    visible::V
    hidden::H
    w::W
    offset_v::Ov
    offset_h::Oh
    function CenteredRBM{V,H,W,Ov,Oh}(
        visible::V, hidden::H, w::W, λv::Ov, λh::Oh
    ) where {V<:AbstractLayer, H<:AbstractLayer, W<:AbstractArray, Ov<:AbstractArray, Oh<:AbstractArray}
        @assert size(w) == (size(visible)..., size(hidden)...)
        @assert size(visible) == size(λv)
        @assert size(hidden) == size(λh)
        return new{V,H,W,Ov,Oh}(visible, hidden, w, λv, λh)
    end
end

function CenteredRBM(
    visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray,
    offset_v::AbstractArray, offset_h::AbstractArray
)
    V, H, W = typeof(visible), typeof(hidden), typeof(w)
    Ov, Oh = typeof(offset_v), typeof(offset_h)
    return CenteredRBM{V,H,W,Ov,Oh}(visible, hidden, w, offset_v, offset_h)
end

"""
    CenteredRBM(rbm, λv, λh)

Creates a centered RBM, with offsets `λv` (visible) and `λh` (hidden).
See <http://jmlr.org/papers/v17/14-237.html> for details.
The resulting model is *not* equivalent to the original `rbm`, unless `λv = 0` and `λh = 0`.
"""
function CenteredRBM(rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray)
    return CenteredRBM(rbm.visible, rbm.hidden, rbm.w, offset_v, offset_h)
end

"""
    CenteredRBM(visible, hidden, w)

Creates a centered RBM, with offsets initialized to zero.
"""
function CenteredRBM(visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray)
    offset_v = similar(w, size(visible)) .= 0
    offset_h = similar(w, size(hidden)).= 0
    return CenteredRBM(RBM(visible, hidden, w), offset_v, offset_h)
end

CenteredRBM(rbm::RBM) = CenteredRBM(rbm.visible, rbm.hidden, rbm.w)

"""
    RBM(centered_rbm::CenteredRBM)

Returns an (uncentered) `RBM` which neglects the offsets of `centered_rbm`.
The resulting model is *not* equivalent to the original `centered_rbm`.
To construct an equivalent model, use the function
`uncenter(centered_rbm)` instead (see [`uncenter`](@ref)).
Shares parameters with `centered_rbm`.
"""
RBMs.RBM(rbm::CenteredRBM) = RBM(rbm.visible, rbm.hidden, rbm.w)

function RBMs.energy(rbm::CenteredRBM, v::AbstractArray, h::AbstractArray)
    Ev = energy(rbm.visible, v)
    Eh = energy(rbm.hidden, h)
    Ew = interaction_energy(rbm, v, h)
    return Ev .+ Eh .+ Ew
end

function RBMs.interaction_energy(centered_rbm::CenteredRBM, v::AbstractArray, h::AbstractArray)
    centered_v = v .- centered_rbm.offset_v
    centered_h = h .- centered_rbm.offset_h
    return interaction_energy(RBM(centered_rbm), centered_v, centered_h)
end

function RBMs.inputs_h_from_v(centered_rbm::CenteredRBM, v::AbstractArray)
    centered_v = v .- centered_rbm.offset_v
    return inputs_h_from_v(RBM(centered_rbm), centered_v)
end

function RBMs.inputs_v_from_h(centered_rbm::CenteredRBM, h::AbstractArray)
    centered_h = h .- centered_rbm.offset_h
    return inputs_v_from_h(RBM(centered_rbm), centered_h)
end

function RBMs.free_energy(rbm::CenteredRBM, v::AbstractArray)
    E = energy(rbm.visible, v)
    inputs = inputs_h_from_v(rbm, v)
    F = -cgf(rbm.hidden, inputs)
    ΔE = energy(Binary(; θ = rbm.offset_h), inputs)
    return E + F - ΔE
end

function RBMs.mean_h_from_v(rbm::CenteredRBM, v::AbstractArray)
    inputs = inputs_h_from_v(rbm, v)
    return mean_from_inputs(rbm.hidden, inputs)
end

function RBMs.mean_v_from_h(rbm::CenteredRBM, h::AbstractArray)
    inputs = inputs_v_from_h(rbm, h)
    return mean_from_inputs(rbm.visible, inputs)
end

function RBMs.∂free_energy(
    rbm::CenteredRBM, v::AbstractArray; wts = nothing,
    moments = moments_from_samples(rbm.visible, v; wts)
)
    inputs = inputs_h_from_v(rbm, v)
    ∂v = ∂energy_from_moments(rbm.visible, moments)

    ∂Γ = ∂cgfs(rbm.hidden, inputs)
    h = grad2ave(rbm.hidden, ∂Γ)

    dims = (ndims(rbm.hidden.par) + 1):ndims(∂Γ)
    ∂h = reshape(wmean(-∂Γ; wts, dims), size(rbm.hidden.par))

    ∂w = ∂interaction_energy(rbm, v, h; wts)
    return ∂RBM(∂v, ∂h, ∂w)
end

function RBMs.∂interaction_energy(
    rbm::CenteredRBM, v::AbstractArray, h::AbstractArray; wts = nothing
)
    centered_v = v .- rbm.offset_v
    centered_h = h .- rbm.offset_h
    ∂w = ∂interaction_energy(RBM(rbm), centered_v, centered_h; wts)
    return ∂w
end

RBMs.log_pseudolikelihood(rbm::CenteredRBM, v::AbstractArray) = log_pseudolikelihood(uncenter(rbm), v)
