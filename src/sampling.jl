function RBMs.sample_h_from_v(rbm::CenteredRBM, v::AbstractArray)
    inputs = inputs_h_from_v(rbm, v)
    return sample_from_inputs(rbm.hidden, inputs)
end

function RBMs.sample_v_from_h(rbm::CenteredRBM, h::AbstractArray)
    inputs = inputs_v_from_h(rbm, h)
    return sample_from_inputs(rbm.visible, inputs)
end

function RBMs.sample_v_from_v(rbm::CenteredRBM, v::AbstractArray; steps::Int = 1)
    for _ in 1:steps
        v = oftype(v, sample_v_from_v_once(rbm, v))
    end
    return v
end

function RBMs.sample_h_from_h(rbm::CenteredRBM, h::AbstractArray; steps::Int = 1)
    for _ in 1:steps
        h = oftype(h, sample_h_from_h_once(rbm, h))
    end
    return h
end

function RBMs.sample_v_from_v_once(rbm::CenteredRBM, v::AbstractArray)
    h = sample_h_from_v(rbm, v)
    v = sample_v_from_h(rbm, h)
    return v
end

function RBMs.sample_h_from_h_once(rbm::CenteredRBM, h::AbstractArray)
    v = sample_v_from_h(rbm, h)
    h = sample_h_from_v(rbm, v)
    return h
end
