CudaRBMs.gpu(rbm::CenteredRBM) = CenteredRBM(
    gpu(rbm.visible), gpu(rbm.hidden), gpu(rbm.w), gpu(rbm.offset_v), gpu(rbm.offset_h)
)

CudaRBMs.cpu(rbm::CenteredRBM) = CenteredRBM(
    cpu(rbm.visible), cpu(rbm.hidden), cpu(rbm.w), cpu(rbm.offset_v), cpu(rbm.offset_h)
)
