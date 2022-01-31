struct CenteredRBM{
    V<:AbstractLayer, H<:AbstractLayer, W<:AbstractArray,
    Ov<:AbstractArray, Oh<:AbstractArray
}
    visible::V
    hidden::H
    w::W
    offset_v::Ov
    offset_h::Oh
    """
        CenteredRBM(visible, hidden, w, offset_v, offset_h)

    Creates a centered RBM, with offsets `offset_v` (visible) and `offset_h` (hidden).
    See <http://jmlr.org/papers/v17/14-237.html> for details.
    """
    function CenteredRBM(
        visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray,
        offset_v::AbstractArray, offset_h::AbstractArray
    )
        @assert size(w) == (size(visible)..., size(hidden)...)
        @assert size(visible) == size(offset_v)
        @assert size(hidden) == size(offset_h)
        return new{typeof(visible), typeof(hidden), typeof(w), typeof(offset_v), typeof(offset_h)}(
            visible, hidden, w, offset_v, offset_h
        )
    end
end

"""
    CenteredRBM(visible, hidden, w)

Creates a centered RBM, with offsets initialized to zero.
"""
function CenteredRBM(visible::AbstractLayer, hidden::AbstractLayer, w::AbstractArray)
    offset_v = zeros(eltype(w), size(visible))
    offset_h = zeros(eltype(w), size(hidden))
    return CenteredRBM(visible, hidden, w, offset_v, offset_h)
end

"""
    CenteredRBM(rbm::RBM, offset_v, offset_h)

Creates a `CenteredRBM` with offsets `offset_v` (visible) and `offset_h` (hidden).
The resulting model is *not* equivalent to the original `rbm`
(unless offset_v = 0 and offset_h = 0).
"""
function CenteredRBM(rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray)
    @assert size(rbm.visible) == size(offset_v)
    @assert size(rbm.hidden) == size(offset_h)
    return CenteredRBM(rbm.visible, rbm.hidden, rbm.w, offset_v, offset_h)
end

CenteredRBM(rbm::RBM) = CenteredRBM(rbm.visible, rbm.hidden, rbm.w)

"""
    RBM(centered_rbm::CenteredRBM)

Returns an (uncentered) `RBM` which neglects the offsets of `centered_rbm`.
The resulting model is *not* equivalent to the original `centered_rbm`.
To construct an equivalent model, use the function
`uncenter(centered_rbm)` instead (see [`uncenter`](@ref)).
"""
RBMs.RBM(centered_rbm::CenteredRBM) = RBM(centered_rbm.visible, centered_rbm.hidden, centered_rbm.w)

function RBMs.energy(centered_rbm::CenteredRBM, v::AbstractArray, h::AbstractArray)
    Ev = RBMs.energy(centered_rbm.visible, v)
    Eh = RBMs.energy(centered_rbm.hidden, h)
    Ew = RBMs.interaction_energy(centered_rbm, v, h)
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

function RBMs.free_energy(centered_rbm::CenteredRBM, v::AbstractArray; β::Real = true)
    E = RBMs.energy(centered_rbm.visible, v)
    inputs = RBMs.inputs_v_to_h(centered_rbm, v)
    F = RBMs.free_energy(centered_rbm.hidden, inputs; β)
    ΔE = RBMs.energy(RBMs.Binary(centered_rbm.offset_h), inputs)
    return E + F - ΔE
end

function RBMs.sample_h_from_v(centered_rbm::CenteredRBM, v::AbstractArray; β::Real = true)
    inputs = RBMs.inputs_v_to_h(centered_rbm, v)
    return RBMs.transfer_sample(centered_rbm.hidden, inputs; β)
end

function RBMs.sample_v_from_h(centered_rbm::CenteredRBM, h::AbstractArray; β::Real = true)
    inputs = RBMs.inputs_h_to_v(centered_rbm, h)
    return RBMs.transfer_sample(centered_rbm.visible, inputs; β)
end

function RBMs.sample_v_from_v(centered_rbm::CenteredRBM, v::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(centered_rbm.visible) == size(v)[1:ndims(centered_rbm.visible)]
    v1 = copy(v)
    for _ in 1:steps
        v1 .= RBMs.sample_v_from_v_once(centered_rbm, v1; β)
    end
    return v1
end

function RBMs.sample_h_from_h(centered_rbm::CenteredRBM, h::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(centered_rbm.hidden) == size(h)[1:ndims(centered_rbm.hidden)]
    h1 = copy(h)
    for _ in 1:steps
        h1 .= RBMs.sample_h_from_h_once(centered_rbm, h1; β)
    end
    return h1
end

function RBMs.sample_v_from_v_once(centered_rbm::CenteredRBM, v::AbstractArray; β::Real = true)
    h = RBMs.sample_h_from_v(centered_rbm, v; β)
    v = RBMs.sample_v_from_h(centered_rbm, h; β)
    return v
end

function RBMs.sample_h_from_h_once(centered_rbm::CenteredRBM, h::AbstractArray; β::Real = true)
    v = RBMs.sample_v_from_h(centered_rbm, h; β)
    h = RBMs.sample_h_from_v(centered_rbm, v; β)
    return h
end

function RBMs.mean_h_from_v(rbm::CenteredRBM, v::AbstractArray; β::Real = true)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    return RBMs.transfer_mean(rbm.hidden, inputs; β)
end

function RBMs.mean_v_from_h(rbm::CenteredRBM, h::AbstractArray; β::Real = true)
    inputs = RBMs.inputs_h_to_v(rbm, h)
    return RBMs.transfer_mean(rbm.visible, inputs; β)
end

function RBMs.mode_v_from_h(rbm::CenteredRBM, h::AbstractArray)
    inputs = RBMs.inputs_h_to_v(rbm, h)
    return RBMs.transfer_mode(rbm.visible, inputs)
end

function RBMs.mode_h_from_v(rbm::CenteredRBM, v::AbstractArray)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    return RBMs.transfer_mode(rbm.hidden, inputs)
end

function RBMs.reconstruction_error(rbm::CenteredRBM, v::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    v1 = RBMs.sample_v_from_v(rbm, v; β, steps)
    ϵ = Statistics.mean(abs.(v .- v1); dims = 1:ndims(rbm.visible))
    if ndims(v) == ndims(rbm.visible)
        return only(ϵ)
    else
        return reshape(ϵ, size(v)[end])
    end
end

function RBMs.mirror(rbm::CenteredRBM)
    function p(i)
        if i ≤ ndims(rbm.visible)
            return i + ndims(rbm.hidden)
        else
            return i - ndims(rbm.visible)
        end
    end
    perm = ntuple(p, ndims(rbm.w))
    w = permutedims(rbm.w, perm)
    return CenteredRBM(rbm.hidden, rbm.visible, w, rbm.offset_h, rbm.offset_v)
end

function RBMs.∂free_energy(
    centered_rbm::CenteredRBM, v::AbstractArray; wts = nothing,
    stats = RBMs.sufficient_statistics(centered_rbm.visible, v; wts)
)
    inputs = RBMs.inputs_v_to_h(centered_rbm, v)
    h = RBMs.transfer_mean(centered_rbm.hidden, inputs)
    ∂v = RBMs.∂energy(centered_rbm.visible; stats...)
    ∂h = RBMs.∂free_energy(centered_rbm.hidden, inputs; wts)
    ∂w = RBMs.∂interaction_energy(centered_rbm, v, h; wts)
    return (visible = ∂v, hidden = ∂h, w = ∂w)
end

function RBMs.∂interaction_energy(
    centered_rbm::CenteredRBM, v::AbstractArray, h::AbstractArray; wts = nothing
)
    centered_v = v .- centered_rbm.offset_v
    centered_h = h .- centered_rbm.offset_h
    ∂w = RBMs.∂interaction_energy(RBM(centered_rbm), centered_v, centered_h)
    return ∂w
end
