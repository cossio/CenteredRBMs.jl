using Test: @test, @testset, @inferred
using RestrictedBoltzmannMachines: Binary, Spin, Potts, Gaussian, ReLU, dReLU, pReLU, xReLU,
    sample_from_inputs, energy

function energy_shift(offset::AbstractArray, x::AbstractArray)
    @assert size(offset) == size(x)[1:ndims(offset)]
    if ndims(offset) == ndims(x)
        return -sum(offset .* x)
    elseif ndims(offset) < ndims(x)
        ΔE = -sum(offset .* x; dims=1:ndims(offset))
        return reshape(ΔE, size(x)[(ndims(offset) + 1):end])
    end
end
