import Aqua
import CenteredRBMs

using Test: @testset

@testset "aqua" begin
    Aqua.test_all(
        CenteredRBMs;
        ambiguities=(exclude=[reshape],),
    )
end
