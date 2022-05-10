struct CenteredRBM{V,H,W,Ov,Oh}
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
    function CenteredRBM(rbm::RBM, λv::AbstractArray, λh::AbstractArray)
        @assert size(rbm.visible) == size(λv)
        @assert size(rbm.hidden) == size(λh)
        V,H,W,Ov,Oh = typeof(rbm.visible), typeof(rbm.hidden), typeof(rbm.w), typeof(λv), typeof(λh)
        return new{V,H,W,Ov,Oh}(rbm.visible, rbm.hidden, rbm.w, λv, λh)
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
    Ev = RBMs.energy(rbm.visible, v)
    Eh = RBMs.energy(rbm.hidden, h)
    Ew = RBMs.interaction_energy(rbm, v, h)
    return Ev .+ Eh .+ Ew
end

function RBMs.interaction_energy(centered_rbm::CenteredRBM, v::AbstractArray, h::AbstractArray)
    centered_v = v .- centered_rbm.offset_v
    centered_h = h .- centered_rbm.offset_h
    return RBMs.interaction_energy(RBM(centered_rbm), centered_v, centered_h)
end

function RBMs.inputs_v_to_h(centered_rbm::CenteredRBM, v::AbstractArray)
    centered_v = v .- centered_rbm.offset_v
    return RBMs.inputs_v_to_h(RBM(centered_rbm), centered_v)
end

function RBMs.inputs_h_to_v(centered_rbm::CenteredRBM, h::AbstractArray)
    centered_h = h .- centered_rbm.offset_h
    return RBMs.inputs_h_to_v(RBM(centered_rbm), centered_h)
end

function RBMs.free_energy(rbm::CenteredRBM, v::AbstractArray)
    E = RBMs.energy(rbm.visible, v)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    F = RBMs.free_energy(rbm.hidden, inputs)
    ΔE = RBMs.energy(RBMs.Binary(rbm.offset_h), inputs)
    return E + F - ΔE
end

function RBMs.mean_h_from_v(rbm::CenteredRBM, v::AbstractArray)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    return RBMs.transfer_mean(rbm.hidden, inputs)
end

function RBMs.mean_v_from_h(rbm::CenteredRBM, h::AbstractArray)
    inputs = RBMs.inputs_h_to_v(rbm, h)
    return RBMs.transfer_mean(rbm.visible, inputs)
end

function RBMs.∂free_energy(
    rbm::CenteredRBM, v::AbstractArray;
    wts = nothing, stats = RBMs.suffstats(rbm.visible, v; wts)
)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    h = RBMs.transfer_mean(rbm.hidden, inputs)
    ∂v = RBMs.∂energy(rbm.visible, stats)
    ∂h = RBMs.∂free_energy(rbm.hidden, inputs; wts)
    ∂w = RBMs.∂interaction_energy(rbm, v, h; wts)
    return (visible = ∂v, hidden = ∂h, w = ∂w)
end

function RBMs.∂interaction_energy(
    rbm::CenteredRBM, v::AbstractArray, h::AbstractArray; wts = nothing
)
    centered_v = v .- rbm.offset_v
    centered_h = h .- rbm.offset_h
    ∂w = RBMs.∂interaction_energy(RBM(rbm), centered_v, centered_h; wts)
    return ∂w
end

RBMs.log_pseudolikelihood(rbm::CenteredRBM, v::AbstractArray) = RBMs.log_pseudolikelihood(uncenter(rbm), v)
