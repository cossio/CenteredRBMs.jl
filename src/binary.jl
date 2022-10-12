"""
    CenteredBinaryRBM(a, b, w, λv = 0, λh = 0)

Construct a centered binary RBM. The energy function is given by:

```math
E(v,h) = -a' * v - b' * h - (v - λv)' * w * (h - λh)
```
"""
function CenteredBinaryRBM(
    a::AbstractArray, b::AbstractArray, w::AbstractArray,
    offset_v::AbstractArray, offset_h::AbstractArray
)
    return CenteredRBM(RBMs.BinaryRBM(a, b, w), offset_v, offset_h)
end

function CenteredBinaryRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    return CenteredRBM(RBMs.BinaryRBM(a, b, w))
end
