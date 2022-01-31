#= As far as I know, Github Actions uses Intel CPUs.
So it is faster to use MKL than OpenBLAS.
It is recommended to load MKL before ANY other package.=#
import MKL
import SafeTestsets

@time SafeTestsets.@safetestset "layers" begin include("layers.jl") end
@time SafeTestsets.@safetestset "binary_centered_rbm" begin include("binary_centered_rbm.jl") end
@time SafeTestsets.@safetestset "centered_gradient" begin include("centered_gradient.jl") end
