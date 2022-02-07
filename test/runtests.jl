import MKL
using SafeTestsets: @safetestset

@time @safetestset "layers" begin include("layers.jl") end
@time @safetestset "binary" begin include("binary.jl") end
@time @safetestset "centered_gradient" begin include("centered_gradient.jl") end
