using Test: @testset, @test, @inferred
using Random: bitrand
using Statistics: mean
using Zygote: gradient
using CenteredRBMs: CenteredBinaryRBM, center, uncenter
using RestrictedBoltzmannMachines: BinaryRBM, energy, free_energy, interaction_energy,
    inputs_v_from_h, inputs_h_from_v, mean_h_from_v, mean_v_from_h, ∂free_energy, mirror

@testset "CenteredBinaryRBM" begin
    rbm = @inferred CenteredBinaryRBM(randn(5), randn(3), randn(5, 3), randn(5), randn(3))
    v = bitrand(size(rbm.visible))
    h = bitrand(size(rbm.hidden))
    E = -rbm.visible.θ' * v - rbm.hidden.θ' * h - (v - rbm.offset_v)' * rbm.w * (h - rbm.offset_h)
    @test @inferred(energy(rbm, v, h)) ≈ E
    @test @inferred(inputs_v_from_h(rbm, h)) ≈ rbm.w  * (h - rbm.offset_h)
    @test @inferred(inputs_h_from_v(rbm, v)) ≈ rbm.w' * (v - rbm.offset_v)
end

@testset "center / uncenter" begin
    rbm = BinaryRBM(randn(3), randn(2), randn(3,2))
    offset_v = randn(3)
    offset_h = randn(2)
    centered_rbm = @inferred center(rbm, offset_v, offset_h)
    @test centered_rbm.offset_v ≈ offset_v
    @test centered_rbm.offset_h ≈ offset_h
    @test @inferred(uncenter(centered_rbm)).visible.θ ≈ rbm.visible.θ
    @test @inferred(uncenter(centered_rbm)).hidden.θ ≈ rbm.hidden.θ
    @test @inferred(uncenter(centered_rbm)).w ≈ rbm.w

    v = bitrand(3,2)
    h = bitrand(2,2)
    @test mean_h_from_v(rbm, v) ≈ mean_h_from_v(uncenter(rbm), v)
    @test mean_v_from_h(rbm, h) ≈ mean_v_from_h(uncenter(rbm), h)
end

@testset "rbm energy invariance" begin
    centered_rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    rbm = uncenter(centered_rbm)
    ΔE = interaction_energy(rbm, centered_rbm.offset_v, centered_rbm.offset_h)::Number
    v = bitrand(size(rbm.visible)..., 100)
    h = bitrand(size(rbm.hidden)..., 100)
    @test @inferred(energy(centered_rbm, v, h)) ≈ energy(rbm, v, h) .+ ΔE
    @test @inferred(free_energy(centered_rbm, v)) ≈ free_energy(rbm, v) .+ ΔE
end

@testset "free energy" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = bitrand(size(rbm.visible)...)
    F = -log(sum(exp(-energy(rbm, v, h)) for h in [[0,0], [0,1], [1,0], [1,1]]))
    @test @inferred(free_energy(rbm, v)) ≈ F
end

@testset "∂free energy" begin
    rbm = CenteredBinaryRBM(randn(3), randn(2), randn(3,2), randn(3), randn(2))
    v = bitrand(size(rbm.visible)...)
    gs = gradient(rbm) do rbm
        mean(free_energy(rbm, v))
    end
    ∂ = @inferred ∂free_energy(rbm, v)
    @test ∂.visible ≈ only(gs).visible.par
    @test ∂.hidden ≈ only(gs).hidden.par
    @test ∂.w ≈ only(gs).w
end

@testset "mirror" begin
    rbm = CenteredBinaryRBM(randn(5,2), randn(7,4,3), randn(5,2,7,4,3), randn(5,2), randn(7,4,3))
    rbm_mirror = @inferred mirror(rbm)
    @test rbm_mirror.visible == rbm.hidden
    @test rbm_mirror.hidden == rbm.visible
    @test rbm_mirror.offset_v == rbm.offset_h
    @test rbm_mirror.offset_h == rbm.offset_v
    v = rand(Bool, size(rbm.visible)..., 13)
    h = rand(Bool, size(rbm.hidden)..., 13)
    @test energy(rbm_mirror, h, v) ≈ energy(rbm, v, h)
end
