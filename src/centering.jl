
@doc raw"""
    uncenter(centered_rbm::CenteredRBM)

Constructs an `RBM` equivalent to the given `CenteredRBM`.
The energies assigned by the two models differ by a constant amount,

```math
E(v,h) - E_c(v,h) = \sum_{i\mu}w_{i\mu}\lambda_i\lambda_\mu
```

where ``E_c(v,h)`` is the energy assigned by `centered_rbm` and ``E(v,h)`` is the energy
assigned by the `RBM` constructed by this method.

This is the inverse operation of [`center`](@ref).

To construct an `RBM` that simply neglects the offsets, call `RBM(centered_rbm)` instead.
"""
uncenter(centered_rbm::CenteredRBM) = RBM(center(centered_rbm))
uncenter(rbm::RBM) = rbm

@doc raw"""
    center(rbm::RBM, offset_v = 0, offset_h = 0)

Constructs a `CenteredRBM` equivalent to the given `rbm`.
The energies assigned by the two models differ by a constant amount,

```math
E(v,h) - E_c(v,h) = \sum_{i\mu}w_{i\mu}\lambda_i\lambda_\mu
```

where ``E(v,h)`` is the energy assigned by the original `rbm`, and
``E_c(v,h)`` is the energy assigned by the returned `CenteredRBM`.

This is the inverse operation of [`uncenter`](@ref).

To construct a `CenteredRBM` that simply includes these offsets,
call `CenteredRBM(rbm, offset_v, offset_h)` instead.
"""
function center(rbm::RBM, offset_v::AbstractArray, offset_h::AbstractArray)
    @assert size(rbm.visible) == size(offset_v)
    @assert size(rbm.hidden) == size(offset_h)
    centered_rbm = center(rbm)
    return center(centered_rbm, offset_v, offset_h)
end

function center(centered_rbm::CenteredRBM, offset_v::AbstractArray, offset_h::AbstractArray)
    @assert size(centered_rbm.visible) == size(offset_v)
    @assert size(centered_rbm.hidden) == size(offset_h)
    return center!(deepcopy(centered_rbm), offset_v, offset_h)
end

center(centered_rbm::CenteredRBM) = center!(deepcopy(centered_rbm))
center(rbm::RBM) = CenteredRBM(rbm)

"""
    center!(centered_rbm, offset_v = 0, offset_h = 0)

Transforms the offsets of `centered_rbm`. The transformed model is equivalent to
the original one (energies differ by a constant).
"""
function center!(centered_rbm::CenteredRBM, offset_v::AbstractArray, offset_h::AbstractArray)
    @assert size(centered_rbm.visible) == size(offset_v)
    @assert size(centered_rbm.hidden) == size(offset_h)
    center_visible!(centered_rbm, offset_v)
    center_hidden!(centered_rbm, offset_h)
    return centered_rbm
end

function center!(centered_rbm::CenteredRBM)
    offset_v = FillArrays.Zeros(size(centered_rbm.visible))
    offset_h = FillArrays.Zeros(size(centered_rbm.hidden))
    center!(centered_rbm, offset_v, offset_h)
    return centered_rbm
end

function center_visible!(centered_rbm::CenteredRBM, offset_v::AbstractArray)
    @assert size(centered_rbm.visible) == size(offset_v)
    inputs = RBMs.inputs_v_to_h(centered_rbm, offset_v)
    shift_fields!(centered_rbm.hidden, inputs)
    centered_rbm.offset_v .= offset_v
    return centered_rbm
end

function center_hidden!(centered_rbm::CenteredRBM, offset_h::AbstractArray)
    @assert size(centered_rbm.hidden) == size(offset_h)
    inputs = RBMs.inputs_h_to_v(centered_rbm, offset_h)
    shift_fields!(centered_rbm.visible, inputs)
    centered_rbm.offset_h .= offset_h
    return centered_rbm
end

function center_visible_from_data!(centered_rbm::CenteredRBM, data::AbstractArray; wts=nothing)
    offset_v = RBMs.batchmean(centered_rbm.visible, data; wts)
    center_visible!(centered_rbm, offset_v)
    return centered_rbm
end

function center_hidden_from_data!(centered_rbm::CenteredRBM, data::AbstractArray; wts=nothing)
    h = RBMs.mean_h_from_v(centered_rbm, data)
    offset_h = RBMs.batchmean(centered_rbm.hidden, h; wts)
    center_hidden!(centered_rbm, offset_h)
    return centered_rbm
end

function center_from_data!(centered_rbm::CenteredRBM, data::AbstractArray)
    center_visible_from_data!(centered_rbm, data)
    center_hidden_from_data!(centered_rbm, data)
    return centered_rbm
end
