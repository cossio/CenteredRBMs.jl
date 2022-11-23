using Base: tail
using Test: @test, @testset, @inferred
using Statistics: mean
using EllipsisNotation: (..)
using RestrictedBoltzmannMachines: RBM, Binary, Spin, Potts, Gaussian, ReLU, dReLU, xReLU, pReLU,
    mean_from_inputs, var_from_inputs, ∂cgf, ∂free_energy, sample_v_from_v
using CenteredRBMs: grad2ave, grad2var

_layers = (
    Binary,
    Spin,
    Potts,
    Gaussian,
    ReLU,
    dReLU,
    pReLU,
    xReLU
)

@testset "Binary" begin
    layer = Binary(; θ = randn(7, 4, 5))
    ∂ = ∂cgf(layer)
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
    @test grad2var(layer, ∂) ≈ var_from_inputs(layer)
end

@testset "Spin" begin
    layer = Spin(; θ = randn(7, 4, 5))
    ∂ = ∂cgf(layer)
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
    @test grad2var(layer, ∂) ≈ var_from_inputs(layer)
end

@testset "Potts" begin
    N = (3, 4, 5)
    layer = Potts(; θ = randn(N...))
    ∂ = ∂cgf(layer)
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
    @test grad2var(layer, ∂) ≈ var_from_inputs(layer)
end

@testset "Gaussian" begin
    N = (3, 4, 5)
    layer = Gaussian(; θ = randn(N...), γ = (rand(N...) .+ 1) .* (rand(Bool, N...) .- 0.5))
    ∂ = ∂cgf(layer)
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
    @test grad2var(layer, ∂) ≈ var_from_inputs(layer)
end

@testset "ReLU" begin
    N = (3, 4, 5)
    layer = ReLU(; θ = randn(N...), γ = (rand(N...) .+ 1) .* (rand(Bool, N...) .- 0.5))
    ∂ = ∂cgf(layer)
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
    @test grad2var(layer, ∂) ≈ var_from_inputs(layer)
end

@testset "dReLU" begin
    N = (3, 5)
    layer = dReLU(
        θp = randn(N...), θn = randn(N...),
        γp = (rand(N...) .+ 1) .* (rand(Bool, N...) .- 0.5),
        γn = (rand(N...) .+ 1) .* (rand(Bool, N...) .- 0.5)
    )
    ∂ = @inferred ∂cgf(layer)
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
    @test grad2var(layer, ∂) ≈ var_from_inputs(layer)
end

@testset "pReLU" begin
    N = (3, 5, 7)
    layer = pReLU(; θ = randn(N...), γ = randn(N...), Δ = randn(N...), η = 2rand(N...) .- 1)
    ∂ = ∂cgf(layer)
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
    @test grad2var(layer, ∂) ≈ var_from_inputs(layer)
end

@testset "xReLU" begin
    N = (3, 5, 7)
    layer = xReLU(; θ = randn(N...), γ = randn(N...), Δ = randn(N...), ξ = randn(N...))
    ∂ = ∂cgf(layer)
    @test grad2ave(layer, ∂) ≈ mean_from_inputs(layer)
    @test grad2var(layer, ∂) ≈ var_from_inputs(layer)
end

@testset "grad2ave $Layer" for Layer in _layers
    layer = Layer((5,))
    rbm = RBM(layer, Binary(; θ = randn(3)), randn(5,3))
    v = sample_v_from_v(rbm, randn(5,100); steps=100)
    ∂ = ∂free_energy(rbm, v)
    @test (@inferred grad2ave(rbm.visible, -∂.visible)) ≈ dropdims(mean(v; dims=2); dims=2)
end
