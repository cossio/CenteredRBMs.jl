struct CenteredRBM{M,Ov,Oh}
    rbm::M
    offset_v::Ov
    offset_h::Oh
    """
        CenteredRBM(rbm, offset_v, offset_h)

    Creates a centered RBM, with offsets `offset_v` (visible) and `offset_h` (hidden).
    See <http://jmlr.org/papers/v17/14-237.html> for details.
    The resulting model is *not* equivalent to the original `rbm`
    (unless offset_v = 0 and offset_h = 0).
    """
    function CenteredRBM(rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray)
        @assert size(visible(rbm)) == size(offset_v)
        @assert size(hidden(rbm)) == size(offset_h)
        return new{typeof(rbm),typeof(offset_v),typeof(offset_h)}(rbm, offset_v, offset_h)
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

RBMs.visible(rbm::CenteredRBM) = visible(RBM(rbm))
RBMs.hidden(rbm::CenteredRBM) = hidden(RBM(rbm))
RBMs.weights(rbm::CenteredRBM) = weights(RBM(rbm))
visible_offset(rbm::CenteredRBM) = rbm.offset_v
hidden_offset(rbm::CenteredRBM) = rbm.offset_h

"""
    RBM(centered_rbm::CenteredRBM)

Returns an (uncentered) `RBM` which neglects the offsets of `centered_rbm`.
The resulting model is *not* equivalent to the original `centered_rbm`.
To construct an equivalent model, use the function
`uncenter(centered_rbm)` instead (see [`uncenter`](@ref)).
Shares parameters with `centered_rbm`.
"""
RBMs.RBM(rbm::CenteredRBM) = rbm.rbm

function RBMs.energy(rbm::CenteredRBM, v::AbstractArray, h::AbstractArray)
    Ev = RBMs.energy(visible(rbm), v)
    Eh = RBMs.energy(hidden(rbm), h)
    Ew = RBMs.interaction_energy(rbm, v, h)
    return Ev .+ Eh .+ Ew
end

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

function RBMs.mean_h_from_v(rbm::CenteredRBM, v::AbstractArray; β::Real=1)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    return RBMs.transfer_mean(hidden(rbm), inputs; β)
end

function RBMs.mean_v_from_h(rbm::CenteredRBM, h::AbstractArray; β::Real = true)
    inputs = RBMs.inputs_h_to_v(rbm, h)
    return RBMs.transfer_mean(visible(rbm), inputs; β)
end

function RBMs.∂free_energy(
    rbm::CenteredRBM, v::AbstractArray;
    wts = nothing, stats = RBMs.suffstats(visible(rbm), v; wts)
)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    h = RBMs.transfer_mean(hidden(rbm), inputs)
    ∂v = RBMs.∂energy(visible(rbm), stats)
    ∂h = RBMs.∂free_energy(hidden(rbm), inputs; wts)
    ∂w = RBMs.∂interaction_energy(rbm, v, h; wts)
    return (visible = ∂v, hidden = ∂h, w = ∂w)
end

function RBMs.∂interaction_energy(
    rbm::CenteredRBM, v::AbstractArray, h::AbstractArray; wts = nothing
)
    centered_v = v .- visible_offset(rbm)
    centered_h = h .- hidden_offset(rbm)
    ∂w = RBMs.∂interaction_energy(RBM(rbm), centered_v, centered_h; wts)
    return ∂w
end

function RBMs.log_pseudolikelihood(rbm::CenteredRBM, v::AbstractArray; β::Real=1)
    return RBMs.log_pseudolikelihood(uncenter(rbm), v; β)
end
