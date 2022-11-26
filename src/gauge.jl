function RestrictedBoltzmannMachines.zerosum!(rbm::CenteredRBM)
    zerosum!(RBM(rbm))
    return rbm
end

function RestrictedBoltzmannMachines.rescale_weights!(rbm::CenteredRBM)
    rescale_weights!(RBM(rbm))
    return rbm
end
