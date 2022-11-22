using Test: @testset, @test
using Random: bitrand
using Statistics: mean
using LinearAlgebra: norm
import RestrictedBoltzmannMachines as RBMs
import CenteredRBMs
using CenteredRBMs: center, uncenter, pcd!
using RestrictedBoltzmannMachines: initialize!, mean_h_from_v, inputs_h_from_v, BinaryRBM

@testset "pcd" begin
    rbm = center(BinaryRBM((28,28), 100))
    train_x = bitrand(28,28,1024)

    initialize!(rbm, train_x) # fit independent site statistics and center
    @test rbm.offset_v ≈ dropdims(mean(train_x; dims=3); dims=3)
    train_h = mean_h_from_v(rbm, train_x)
    @test rbm.offset_h ≈ vec(mean(train_h; dims=2))
    @test norm(mean(inputs_h_from_v(rbm, train_x); dims=2)) < 1e-6

    pcd!(rbm, train_x; batchsize=1024, iters=100)
    @test rbm.offset_v ≈ dropdims(mean(train_x; dims=3); dims=3)
    train_h = mean_h_from_v(rbm, train_x)
    @test rbm.offset_h ≈ vec(mean(train_h; dims=2)) rtol=0.1
    # not exact because offset is updated after having updated the parameters!
end
