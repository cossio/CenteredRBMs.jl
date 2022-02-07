using Test: @testset, @test
using Random: bitrand
using Statistics: mean
using LinearAlgebra: norm
import RestrictedBoltzmannMachines as RBMs
import CenteredRBMs
using CenteredRBMs: center, uncenter

@testset "pcd" begin
    rbm = CenteredRBMs.center(RBMs.BinaryRBM((28,28), 100))
    train_x = bitrand(28,28,1024)

    RBMs.initialize!(rbm, train_x) # fit independent site statistics and center
    @test CenteredRBMs.visible_offset(rbm) ≈ dropdims(mean(train_x; dims=3); dims=3)
    train_h = RBMs.mean_h_from_v(rbm, train_x)
    @test CenteredRBMs.hidden_offset(rbm) ≈ vec(mean(train_h; dims=2))
    @test norm(mean(RBMs.inputs_v_to_h(rbm, train_x); dims=2)) < 1e-6

    RBMs.pcd!(rbm, train_x; batchsize=128)
    @test CenteredRBMs.visible_offset(rbm) ≈ dropdims(mean(train_x; dims=3); dims=3)
    train_h = RBMs.mean_h_from_v(rbm, train_x)
    @test CenteredRBMs.hidden_offset(rbm) ≈ vec(mean(train_h; dims=2))
end
