import Test
import Random
import Statistics
import Zygote
import RestrictedBoltzmannMachines as RBMs
import CenteredRBMs
using CenteredRBMs: CenteredBinaryRBM, center, uncenter
using RestrictedBoltzmannMachines: visible, hidden, weights

@testset "center / uncenter" begin
    rbm = RBMs.BinaryRBM(randn(3), randn(2), randn(3,2))
    offset_v = randn(3)
    offset_h = randn(2)
    centered_rbm = center(rbm, offset_v, offset_h)
    @test centered_rbm.offset_v ≈ offset_v
    @test centered_rbm.offset_h ≈ offset_h
    @test visible(uncenter(centered_rbm)).θ ≈ visible(rbm).θ
    @test hidden(uncenter(centered_rbm)).θ ≈ hidden(rbm).θ
    @test weights(uncenter(centered_rbm)) ≈ weights(rbm)

    v = Random.bitrand(3,2)
    @test RBMs.mean_h_from_v(rbm, v) ≈ RBMs.mean_h_from_v(uncenter(rbm), v)
    h = Random.bitrand(2,2)
    @test RBMs.mean_v_from_h(rbm, h) ≈ RBMs.mean_v_from_h(uncenter(rbm), h)
end

@testset "rbm energy invariance" begin
    centered_rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    rbm = uncenter(centered_rbm)
    ΔE = RBMs.interaction_energy(rbm, centered_rbm.offset_v, centered_rbm.offset_h)::Number
    v = Random.bitrand(size(visible(rbm))..., 100)
    h = Random.bitrand(size(hidden(rbm))..., 100)
    @test RBMs.energy(centered_rbm, v, h) ≈ RBMs.energy(rbm, v, h) .+ ΔE
    @test RBMs.free_energy(centered_rbm, v) ≈ RBMs.free_energy(rbm, v) .+ ΔE
end

@testset "free energy" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = Random.bitrand(size(visible(rbm))...)
    F = -log(sum(exp(-RBMs.energy(rbm, v, h)) for h in [[0,0], [0,1], [1,0], [1,1]]))
    @test RBMs.free_energy(rbm, v) ≈ F
end

@testset "∂free energy" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = Random.bitrand(size(visible(rbm))...)
    gs = Zygote.gradient(rbm) do rbm
        Statistics.mean(RBMs.free_energy(rbm, v))
    end
    ∂ = RBMs.∂free_energy(rbm, v)
    @test ∂.visible.θ ≈ only(gs).rbm.visible.θ
    @test ∂.hidden.θ ≈ only(gs).rbm.hidden.θ
    @test ∂.w ≈ only(gs).rbm.w
end
