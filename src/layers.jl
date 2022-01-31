"""
    shift_fields!(layer, offset)

Adds `offset` to the `layer` fields.
"""
function shift_fields! end

function shift_fields!(
    layer::Union{Binary,Spin,Potts,Gaussian,ReLU,pReLU,xReLU}, offset::AbstractArray
)
    @assert size(layer) == size(offset)
    layer.θ .+= offset
    return layer
end

function shift_fields!(layer::dReLU, offset::AbstractArray)
    @assert size(layer) == size(offset)
    layer.θp .+= offset
    layer.θn .+= offset
    return layer
end
