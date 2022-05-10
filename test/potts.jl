using Test: @testset, @test, @inferred
import Random
import Statistics
import Zygote
import RestrictedBoltzmannMachines as RBMs
import CenteredRBMs
using CenteredRBMs: CenteredBinaryRBM, center, uncenter
using RestrictedBoltzmannMachines: visible, hidden, weights, RBM, Potts

@testset "center / uncenter" begin
    rbm = RBMs.BinaryRBM(randn(3), randn(2), randn(3,2))
    offset_v = randn(3)
    offset_h = randn(2)
    centered_rbm = center(rbm, offset_v, offset_h)
    @test centered_rbm.offset_v ≈ offset_v
    @test centered_rbm.offset_h ≈ offset_h
    @test uncenter(centered_rbm).visible.θ ≈ rbm.visible.θ
    @test uncenter(centered_rbm).hidden.θ ≈ rbm.hidden.θ
    @test uncenter(centered_rbm).w ≈ rbm.w

    v = Random.bitrand(3,2)
    @test RBMs.mean_h_from_v(rbm, v) ≈ RBMs.mean_h_from_v(uncenter(rbm), v)
    h = Random.bitrand(2,2)
    @test RBMs.mean_v_from_h(rbm, h) ≈ RBMs.mean_v_from_h(uncenter(rbm), h)
end

@testset "rbm energy invariance" begin
    centered_rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    rbm = uncenter(centered_rbm)
    ΔE = RBMs.interaction_energy(rbm, centered_rbm.offset_v, centered_rbm.offset_h)::Number
    v = Random.bitrand(size(rbm.visible)..., 100)
    h = Random.bitrand(size(rbm.hidden)..., 100)
    @test RBMs.energy(centered_rbm, v, h) ≈ RBMs.energy(rbm, v, h) .+ ΔE
    @test RBMs.free_energy(centered_rbm, v) ≈ RBMs.free_energy(rbm, v) .+ ΔE
end

@testset "free energy" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = Random.bitrand(size(rbm.visible)...)
    F = -log(sum(exp(-RBMs.energy(rbm, v, h)) for h in [[0,0], [0,1], [1,0], [1,1]]))
    @test RBMs.free_energy(rbm, v) ≈ F
end

@testset "∂free energy" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = Random.bitrand(size(rbm.visible)...)
    gs = Zygote.gradient(rbm) do rbm
        Statistics.mean(RBMs.free_energy(rbm, v))
    end
    ∂ = RBMs.∂free_energy(rbm, v)
    @test ∂.visible.θ ≈ only(gs).visible.θ
    @test ∂.hidden.θ ≈ only(gs).hidden.θ
    @test ∂.w ≈ only(gs).w
end
