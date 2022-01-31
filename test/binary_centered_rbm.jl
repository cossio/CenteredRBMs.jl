import Test
import Random
import Statistics
import Zygote
import RestrictedBoltzmannMachines as RBMs
import CenteredRBMs
using CenteredRBMs: BinaryCenteredRBM

@testset "center / uncenter" begin
    rbm = RBMs.BinaryRBM(randn(3), randn(2), randn(3,2))
    offset_v = randn(3)
    offset_h = randn(2)
    centered_rbm = CenteredRBMs.center(rbm, offset_v, offset_h)
    @test centered_rbm.offset_v ≈ offset_v
    @test centered_rbm.offset_h ≈ offset_h
    @test CenteredRBMs.uncenter(centered_rbm).visible.θ ≈ rbm.visible.θ
    @test CenteredRBMs.uncenter(centered_rbm).hidden.θ ≈ rbm.hidden.θ
    @test CenteredRBMs.uncenter(centered_rbm).w ≈ rbm.w
end

@testset "rbm energy invariance" begin
    centered_rbm = BinaryCenteredRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    rbm = CenteredRBMs.uncenter(centered_rbm)
    ΔE = RBMs.interaction_energy(rbm, centered_rbm.offset_v, centered_rbm.offset_h)::Number
    v = Random.bitrand(size(rbm.visible)..., 100)
    h = Random.bitrand(size(rbm.hidden)..., 100)
    @test RBMs.energy(centered_rbm, v, h) ≈ RBMs.energy(rbm, v, h) .+ ΔE
    @test RBMs.free_energy(centered_rbm, v) ≈ RBMs.free_energy(rbm, v) .+ ΔE
end

@testset "free energy" begin
    centered_rbm = BinaryCenteredRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = Random.bitrand(size(centered_rbm.visible)...)
    F = -log(sum(exp(-RBMs.energy(centered_rbm, v, h)) for h in [[0,0], [0,1], [1,0], [1,1]]))
    @test RBMs.free_energy(centered_rbm, v) ≈ F
end

@testset "∂free energy" begin
    centered_rbm = BinaryCenteredRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = Random.bitrand(size(centered_rbm.visible)...)
    gs, = Zygote.gradient(centered_rbm) do centered_rbm
        Statistics.mean(RBMs.free_energy(centered_rbm, v))
    end
    ∂ = RBMs.∂free_energy(centered_rbm, v)
    @test ∂.visible.θ ≈ gs.visible.θ
    @test ∂.hidden.θ ≈ gs.hidden.θ
    @test ∂.w ≈ gs.w
end
