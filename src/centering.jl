
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
    centered_rbm = center(rbm)
    return center(centered_rbm, offset_v, offset_h)
end

function center(rbm::CenteredRBM, offset_v::AbstractArray, offset_h::AbstractArray)
    center!(deepcopy(rbm), offset_v, offset_h)
end

center(rbm::CenteredRBM) = center!(deepcopy(rbm))
center(rbm::RBM) = CenteredRBM(rbm)

"""
    center!(centered_rbm, offset_v = 0, offset_h = 0)

Transforms the offsets of `centered_rbm`. The transformed model is equivalent to
the original one (energies differ by a constant).
"""
function center!(rbm::CenteredRBM, offset_v::AbstractArray, offset_h::AbstractArray)
    center_visible!(rbm, offset_v)
    center_hidden!(rbm, offset_h)
    return rbm
end

function center!(rbm::CenteredRBM)
    offset_v = FillArrays.Falses(size(rbm.visible))
    offset_h = FillArrays.Falses(size(rbm.hidden))
    center!(rbm, offset_v, offset_h)
    return rbm
end

function center_visible!(rbm::CenteredRBM, offset_v::AbstractArray)
    @assert size(offset_v) == size(rbm.visible)
    inputs = RBMs.inputs_v_to_h(rbm, offset_v)
    shift_fields!(rbm.hidden, inputs)
    rbm.offset_v .= offset_v
    return rbm
end

function center_hidden!(rbm::CenteredRBM, offset_h::AbstractArray)
    @assert size(offset_h) == size(rbm.hidden)
    inputs = RBMs.inputs_h_to_v(rbm, offset_h)
    shift_fields!(rbm.visible, inputs)
    rbm.offset_h .= offset_h
    return rbm
end

function center_visible_from_data!(rbm::CenteredRBM, data::AbstractArray; wts=nothing)
    offset_v = RBMs.batchmean(rbm.visible, data; wts)
    center_visible!(rbm, offset_v)
    return rbm
end

function center_hidden_from_data!(rbm::CenteredRBM, data::AbstractArray; wts=nothing)
    h = RBMs.mean_h_from_v(rbm, data)
    offset_h = RBMs.batchmean(rbm.hidden, h; wts)
    center_hidden!(rbm, offset_h)
    return rbm
end

function center_from_data!(rbm::CenteredRBM, data::AbstractArray)
    center_visible_from_data!(rbm, data)
    center_hidden_from_data!(rbm, data)
    return rbm
end
