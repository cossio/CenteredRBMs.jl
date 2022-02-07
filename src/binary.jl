"""
    CenteredBinaryRBM(a, b, w, offset_v = 0, offset_h = 0)

Construct a centered RBM with binary visible and hidden units
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
