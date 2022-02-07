struct CenteredRBM{V,H,W,Ov,Oh} <: AbstractRBM{V,H,W}
    visible::V
    hidden::H
    w::W
    offset_v::Ov
    offset_h::Oh
    """
        CenteredRBM(rbm, offset_v, offset_h)

    Creates a centered RBM, with offsets `offset_v` (visible) and `offset_h` (hidden).
    See <http://jmlr.org/papers/v17/14-237.html> for details.
    The resulting model is *not* equivalent to the original `rbm`
    (unless offset_v = 0 and offset_h = 0).
    """
    function CenteredRBM(
        rbm::AbstractRBM{V,H,W}, offset_v::AbstractArray, offset_h::AbstractArray
    ) where {V,H,W}
        @assert size(visible(rbm)) == size(offset_v)
        @assert size(hidden(rbm)) == size(offset_h)
        return new{V,H,W,typeof(offset_v),typeof(offset_h)}(
            visible(rbm), hidden(rbm), weights(rbm), offset_v, offset_h
        )
    end
end

function CenteredRBM(
    visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray,
    offset_v::AbstractArray, offset_h::AbstractArray
)
    return CenteredRBM(RBM(visible, hidden, w), offset_v, offset_h)
end

"""
    CenteredRBM(visible, hidden, w)

Creates a centered RBM, with offsets initialized to zero.
"""
function CenteredRBM(visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray)
    offset_v = zeros(eltype(w), size(visible))
    offset_h = zeros(eltype(w), size(hidden))
    return CenteredRBM(RBM(visible, hidden, w), offset_v, offset_h)
end

CenteredRBM(rbm::RBM) = CenteredRBM(visible(rbm), hidden(rbm), weights(rbm))

RBMs.visible(rbm::CenteredRBM) = rbm.visible
RBMs.hidden(rbm::CenteredRBM) = rbm.hidden
RBMs.weights(rbm::CenteredRBM) = rbm.w
visible_offset(rbm::CenteredRBM) = rbm.offset_v
hidden_offset(rbm::CenteredRBM) = rbm.offset_h

"""
    RBM(centered_rbm::CenteredRBM)

Returns an (uncentered) `RBM` which neglects the offsets of `centered_rbm`.
The resulting model is *not* equivalent to the original `centered_rbm`.
To construct an equivalent model, use the function
`uncenter(centered_rbm)` instead (see [`uncenter`](@ref)).
"""
RBMs.RBM(rbm::CenteredRBM) = RBM(visible(rbm), hidden(rbm), weights(rbm))

function RBMs.interaction_energy(centered_rbm::CenteredRBM, v::AbstractArray, h::AbstractArray)
    centered_v = v .- visible_offset(centered_rbm)
    centered_h = h .- hidden_offset(centered_rbm)
    return RBMs.interaction_energy(RBM(centered_rbm), centered_v, centered_h)
end

function RBMs.inputs_v_to_h(centered_rbm::CenteredRBM, v::AbstractArray)
    centered_v = v .- visible_offset(centered_rbm)
    return RBMs.inputs_v_to_h(RBM(centered_rbm), centered_v)
end

function RBMs.inputs_h_to_v(centered_rbm::CenteredRBM, h::AbstractArray)
    centered_h = h .- hidden_offset(centered_rbm)
    return RBMs.inputs_h_to_v(RBM(centered_rbm), centered_h)
end

function RBMs.free_energy(rbm::CenteredRBM, v::AbstractArray; β::Real = true)
    E = RBMs.energy(visible(rbm), v)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    F = RBMs.free_energy(hidden(rbm), inputs; β)
    ΔE = RBMs.energy(RBMs.Binary(hidden_offset(rbm)), inputs)
    return E + F - ΔE
end

function RBMs.mirror(rbm::CenteredRBM)
    return CenteredRBM(mirror(RBM(rbm)), hidden_offset(rbm), visible_offset(rbm))
end

function RBMs.∂interaction_energy(
    rbm::CenteredRBM, v::AbstractArray, h::AbstractArray; wts = nothing
)
    centered_v = v .- visible_offset(rbm)
    centered_h = h .- hidden_offset(rbm)
    ∂w = RBMs.∂interaction_energy(RBM(rbm), centered_v, centered_h; wts)
    return ∂w
end
