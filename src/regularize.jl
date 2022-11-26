function RestrictedBoltzmannMachines.∂regularize!(∂::∂RBM, rbm::CenteredRBM; kwargs...)
    ∂regularize!(∂, RBM(rbm); kwargs...)
end
