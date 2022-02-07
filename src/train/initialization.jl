function initialize!(rbm::CenteredRBM, data::AbstractArray; ϵ::Real = 1e-6)
    initialize!(RBM(rbm), data; ϵ)
    center_from_data!(rbm, data)
    return rbm
end
