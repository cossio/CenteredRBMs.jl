function RBMs.sample_h_from_v(rbm::CenteredRBM, v::AbstractArray)
    inputs = RBMs.inputs_v_to_h(rbm, v)
    return RBMs.transfer_sample(rbm.hidden, inputs)
end

function RBMs.sample_v_from_h(rbm::CenteredRBM, h::AbstractArray)
    inputs = RBMs.inputs_h_to_v(rbm, h)
    return RBMs.transfer_sample(rbm.visible, inputs)
end

function RBMs.sample_v_from_v(rbm::CenteredRBM, v::AbstractArray; steps::Int = 1)
    @assert size(rbm.visible) == size(v)[1:ndims(rbm.visible)]
    for _ in 1:steps
        v = oftype(v, RBMs.sample_v_from_v_once(rbm, v))
    end
    return v
end

function RBMs.sample_h_from_h(rbm::CenteredRBM, h::AbstractArray; steps::Int = 1)
    @assert size(rbm.hidden) == size(h)[1:ndims(rbm.hidden)]
    for _ in 1:steps
        h = oftype(h, RBMs.sample_h_from_h_once(rbm, h))
    end
    return h
end

function RBMs.sample_v_from_v_once(rbm::CenteredRBM, v::AbstractArray)
    h = RBMs.sample_h_from_v(rbm, v)
    v = RBMs.sample_v_from_h(rbm, h)
    return v
end

function RBMs.sample_h_from_h_once(rbm::CenteredRBM, h::AbstractArray)
    v = RBMs.sample_v_from_h(rbm, h)
    h = RBMs.sample_h_from_v(rbm, v)
    return h
end
