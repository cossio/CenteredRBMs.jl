using Test: @test, @testset
import RestrictedBoltzmannMachines as RBMs
import CenteredRBMs

function energy_shift(offset::AbstractArray, x::AbstractArray)
    @assert size(offset) == size(x)[1:ndims(offset)]
    if ndims(offset) == ndims(x)
        return -sum(offset .* x)
    elseif ndims(offset) < ndims(x)
        ΔE = -sum(offset .* x; dims=1:ndims(offset))
        return reshape(ΔE, size(x)[(ndims(offset) + 1):end])
    end
end

@testset "shift_fields!" begin
    N = (3, 4)
    layers = (
        RBMs.Binary(; θ = randn(N...)),
        RBMs.Spin(; θ = randn(N...)),
        RBMs.Potts(; θ = randn(N...)),
        RBMs.Gaussian(; θ = randn(N...), γ = rand(N...)),
        RBMs.ReLU(; θ = randn(N...), γ = rand(N...)),
        RBMs.dReLU(; θp = randn(N...), θn = randn(N...), γp = rand(N...), γn = rand(N...)),
        RBMs.pReLU(; θ = randn(N...), γ = rand(N...), Δ = randn(N...), η = rand(N...) .- 0.5),
        RBMs.xReLU(; θ = randn(N...), γ = rand(N...), Δ = randn(N...), ξ = randn(N...)),
    )
    for layer in layers
        offset = randn(size(layer)...)
        x = RBMs.sample_from_inputs(layer, randn(size(layer)..., 2, 3))
        layer_shifted = CenteredRBMs.shift_fields!(deepcopy(layer), offset)
        @test RBMs.energy(layer_shifted, x) ≈ RBMs.energy(layer, x) + energy_shift(offset, x)
    end
end
