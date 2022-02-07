import Random
import LinearAlgebra
import InteractiveUtils
import Zygote
import RestrictedBoltzmannMachines as RBMs
import CenteredRBMs
using Statistics: mean
using RestrictedBoltzmannMachines: visible, hidden, weights, energy, interaction_energy
using RestrictedBoltzmannMachines: RBM, Binary

function centered_energy(
    rbm::RBM, v::AbstractArray, h::AbstractArray, λv::AbstractArray, λh::AbstractArray
)
    Ev = energy(visible(rbm), v)
    Eh = energy(hidden(rbm), h)
    return Ev .+ Eh .+ interaction_energy(rbm, v .- λv, h .- λh) .- interaction_energy(rbm, λv, λh)
end

function binary_center(rbm::RBM{<:Binary,<:Binary}, λv::AbstractArray, λh::AbstractArray)
    RBMs.BinaryRBM(visible(rbm).θ + weights(rbm) * λh, hidden(rbm).θ + weights(rbm)' * λv, weights(rbm))
end

@testset "centering binary RBMs" begin
    rbm = RBMs.BinaryRBM(randn(4), randn(2), randn(4,2))
    λv = randn(size(visible(rbm)))
    λh = randn(size(hidden(rbm)))
    rbmc = binary_center(rbm, λv, λh)
    v = Random.bitrand(size(visible(rbm))...,3,2)
    h = Random.bitrand(size(hidden(rbm))...,3,2)
    @test RBMs.energy(rbm, v, h) ≈ centered_energy(rbmc, v, h, λv, λh)
    @test visible(binary_center(rbmc, -λv, -λh)).θ ≈ visible(rbm).θ
    @test hidden(binary_center(rbmc, -λv, -λh)).θ ≈ hidden(rbm).θ
    @test weights(binary_center(rbmc, -λv, -λh)) ≈ weights(rbm)
    @test visible(rbmc).θ ≈ visible(CenteredRBMs.center(rbm, λv, λh)).θ
    @test hidden(rbmc).θ ≈ hidden(CenteredRBMs.center(rbm, λv, λh)).θ
    @test weights(rbmc) ≈ weights(CenteredRBMs.center(rbm, λv, λh))

    ∂rbmc, = Zygote.gradient(rbmc -> sum(centered_energy(rbmc, v, h, λv, λh)), rbmc)
    ∂rbm, = Zygote.gradient(rbm -> sum(RBMs.energy(rbm, v, h)), rbm)
    ∂crbm = CenteredRBMs.center_gradients(rbm, ∂rbm, λv, λh)

    for i in eachindex(visible(rbm).θ)
        J = only(Zygote.gradient(rbmc -> visible(binary_center(rbmc, -λv, -λh)).θ[i], rbmc))
        @test ∂crbm.visible.θ[i] ≈ (
            (isnothing(J.visible) ? 0 : LinearAlgebra.dot(J.visible.θ, ∂rbmc.visible.θ)) +
            (isnothing(J.hidden) ? 0 : LinearAlgebra.dot(J.hidden.θ, ∂rbmc.hidden.θ)) +
            (isnothing(J.w) ? 0 : LinearAlgebra.dot(J.w, ∂rbmc.w))
        )
    end

    for μ in eachindex(hidden(rbm).θ)
        J = only(Zygote.gradient(rbmc -> hidden(binary_center(rbmc, -λv, -λh)).θ[μ], rbmc))
        @test ∂crbm.hidden.θ[μ] ≈ (
            (isnothing(J.visible) ? 0 : LinearAlgebra.dot(J.visible.θ, ∂rbmc.visible.θ)) +
            (isnothing(J.hidden) ? 0 : LinearAlgebra.dot(J.hidden.θ, ∂rbmc.hidden.θ)) +
            (isnothing(J.w) ? 0 : LinearAlgebra.dot(J.w, ∂rbmc.w))
        )
    end

    for k in eachindex(weights(rbm))
        J = only(Zygote.gradient(rbmc -> weights(binary_center(rbmc, -λv, -λh))[k], rbmc))
        @test ∂crbm.w[k] ≈ (
            (isnothing(J.visible) ? 0 : LinearAlgebra.dot(J.visible.θ, ∂rbmc.visible.θ)) +
            (isnothing(J.hidden) ? 0 : LinearAlgebra.dot(J.hidden.θ, ∂rbmc.hidden.θ)) +
            (isnothing(J.w) ? 0 : LinearAlgebra.dot(J.w, ∂rbmc.w))
        )
    end
end

struct2nt(s) = NamedTuple{propertynames(s)}(([getproperty(s, p) for p in propertynames(s)]...,))

@testset "centering $Layer gradients" for Layer in InteractiveUtils.subtypes(RBMs.AbstractLayer)
    layer = Layer(5)
    ∂ = struct2nt(layer)
    λ = randn(size(layer))
    applicable(CenteredRBMs.center_gradients, layer, ∂, λ) || continue
    for p in propertynames(layer)
        Random.rand!(getproperty(layer, p))
    end
    ∂c = CenteredRBMs.center_gradients(layer, ∂, λ)
    layerc = deepcopy(layer)
    for p in propertynames(layer)
        getproperty(layerc, p) .= getproperty(∂c, p)
    end
    x = randn(size(layer)..., 10)
    @test RBMs.energy(layer, x) ≈ RBMs.energy(layerc, x) - x' * λ
end

@testset "grad2mean $Layer" for Layer in InteractiveUtils.subtypes(RBMs.AbstractLayer)
    layer = Layer(5)
    rbm = RBMs.RBM(layer, RBMs.Binary(randn(3)), randn(5,3))
    v = RBMs.sample_v_from_v(rbm, randn(5,100); steps=100)
    ∂ = RBMs.∂free_energy(rbm, v)
    applicable(CenteredRBMs.grad2mean, layer, ∂.visible) || continue
    @test CenteredRBMs.grad2mean(visible(rbm), ∂.visible) ≈ dropdims(mean(v; dims=2); dims=2)
end
