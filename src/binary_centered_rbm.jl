"""
    BinaryCenteredRBM(a, b, w, offset_v = 0, offset_h = 0)

Construct a centered RBM with binary visible and hidden units
"""
function BinaryCenteredRBM(
    a::AbstractArray, b::AbstractArray, w::AbstractArray,
    offset_v::AbstractArray, offset_h::AbstractArray
)
    @assert size(w) == (size(a)..., size(b)...)
    @assert size(a) == size(offset_v)
    @assert size(b) == size(offset_h)
    return CenteredRBM(RBMs.Binary(a), RBMs.Binary(b), w, offset_v, offset_h)
end

function BinaryCenteredRBM(a::AbstractArray, b::AbstractArray, w::AbstractArray)
    @assert size(w) == (size(a)..., size(b)...)
    @assert size(a) == size(offset_v)
    @assert size(b) == size(offset_h)
    return CenteredRBM(RBMs.Binary(a), RBMs.Binary(b), w)
end
