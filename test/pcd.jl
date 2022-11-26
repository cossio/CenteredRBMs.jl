using Test: @testset, @test
using Random: bitrand
using Statistics: mean
using LinearAlgebra: norm
import CenteredRBMs
using Optimisers: Adam
using CenteredRBMs: center, uncenter
using RestrictedBoltzmannMachines: initialize!, mean_h_from_v, inputs_h_from_v, BinaryRBM, pcd!, sample_v_from_v

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

@testset "pcd training" begin
    data = falses(2, 1000)
    data[1, 1:2:end] .= true
    data[2, 1:2:end] .= true

    rbm = center(BinaryRBM(2, 5))
    initialize!(rbm, data)
    pcd!(rbm, data; iters = 10000, batchsize = 64, steps = 10, optim = Adam(5e-4))

    v_sample = sample_v_from_v(rbm, bitrand(2, 10000); steps=50)

    @test 0.4 < mean(v_sample[1,:]) < 0.6
    @test 0.4 < mean(v_sample[2,:]) < 0.6
    @test 0.4 < mean(v_sample[1,:] .* v_sample[2,:]) < 0.6
end
