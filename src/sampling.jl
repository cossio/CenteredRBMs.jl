function RBMs.sample_h_from_v(rbm::CenteredRBM, v::AbstractArray; β::Real = true)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    return RBMs.transfer_sample(hidden(rbm), inputs; β)
end

function RBMs.sample_v_from_h(rbm::CenteredRBM, h::AbstractArray; β::Real = true)
    inputs = RBMs.inputs_h_to_v(rbm, h)
    return RBMs.transfer_sample(visible(rbm), inputs; β)
end

function RBMs.sample_v_from_v(rbm::CenteredRBM, v::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(visible(rbm)) == size(v)[1:ndims(visible(rbm))]
    v1 = copy(v)
    for _ in 1:steps
        v1 .= RBMs.sample_v_from_v_once(rbm, v1; β)
    end
    return v1
end

function RBMs.sample_h_from_h(rbm::CenteredRBM, h::AbstractArray; β::Real = true, steps::Int = 1)
    @assert size(rbm.hidden) == size(h)[1:ndims(hidden(rbm))]
    h1 = copy(h)
    for _ in 1:steps
        h1 .= RBMs.sample_h_from_h_once(rbm, h1; β)
    end
    return h1
end

function RBMs.sample_v_from_v_once(rbm::CenteredRBM, v::AbstractArray; β::Real = true)
    h = RBMs.sample_h_from_v(rbm, v; β)
    v = RBMs.sample_v_from_h(rbm, h; β)
    return v
end

function RBMs.sample_h_from_h_once(rbm::CenteredRBM, h::AbstractArray; β::Real = true)
    v = RBMs.sample_v_from_h(rbm, h; β)
    h = RBMs.sample_h_from_v(rbm, v; β)
    return h
end
