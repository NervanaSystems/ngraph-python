import itertools as itt
import numpy as np


def update_conv(conv_slices, I, E, U):
    mSlice, pSlice, qSlice, _, _, _ = conv_slices
    K, M, P, Q, N = E.shape
    C, _, _, _, K = U.shape
    U.fill(0.0)

    for (m, mS), (p, pS), (q, qS) in itt.product(enumerate(mSlice),
                                                 enumerate(pSlice),
                                                 enumerate(qSlice)):
        sliceT, sliceD, tlen = mS
        sliceR, sliceH, rlen = pS
        sliceS, sliceW, slen = qS
        slicedI = I[:, sliceD, sliceH, sliceW, :].reshape((-1, N))
        slicedE = E[:, m, p, q, :]
        update = np.dot(slicedI, slicedE.T).reshape((C, tlen, rlen, slen, K))
        U[:, sliceT, sliceR, sliceS, :] += update


def fprop_pool(pool_slices, arrI, arrO):
    kSlice, mSlice, pSlice, qSlice, op, arrA = pool_slices
    K, M, P, Q, N = arrO.shape

    for (k, kS), (m, mS), (p, pS), (q, qS) in itt.product(enumerate(kSlice),
                                                          enumerate(mSlice),
                                                          enumerate(pSlice),
                                                          enumerate(qSlice)):
        sliceC, _ = kS
        sliceD, _ = mS
        sliceH, _ = pS
        sliceW, _ = qS

        sliceI = arrI[sliceC, sliceD, sliceH, sliceW, :].reshape(-1, N)
        if op == "max":
            arrA[k, m, p, q, :] = np.argmax(sliceI, axis=0)
            arrO[k, m, p, q, :] = np.max(sliceI, axis=0)
        elif op == "avg":
            arrO[k, m, p, q, :] = np.mean(sliceI, axis=0)
        elif op == "l2":
            arrO[k, m, p, q, :] = np.sqrt(np.sum(np.square(sliceI), axis=0))


def bprop_pool(pool_slices, arrE, arrD):
    kSlice, mSlice, pSlice, qSlice, op, arrA = pool_slices
    arrD[:] = 0
    K, M, P, Q, N = arrE.shape

    for (k, kS), (m, mS), (p, pS), (q, qS) in itt.product(enumerate(kSlice),
                                                          enumerate(mSlice),
                                                          enumerate(pSlice),
                                                          enumerate(qSlice)):
        sliceC, clen = kS
        sliceD, dlen = mS
        sliceH, hlen = pS
        sliceW, wlen = qS

        patch_in = (sliceC, sliceD, sliceH, sliceW, slice(None))
        patch_out = (k, m, p, q, slice(None))
        sliceB = arrD[patch_in].reshape((-1, N))
        if op == "max":
            max_n = arrA[patch_out]
            sliceB[max_n, list(range(N))] += arrE[patch_out]
        elif op == "avg":
            sliceB += arrE[patch_out] * (1.0 / sliceB.shape[0])
        else:
            raise NotImplementedError
        arrD[patch_in] = sliceB.reshape((clen, dlen, hlen, wlen, N))


def fprop_lut(lut, idx, axis, output):
    output[:] = lut.take(idx.astype(int), axis)


def update_lut(error, idx, pad_idx, axis, dW):
    dW[:] = 0
    idx = idx.astype(int)
    unqidx, inv = np.unique(idx, return_inverse=True)
    groups = [np.where(inv == i) for i in range(len(unqidx))]
    for (wrd_id, group) in zip(unqidx, groups):
        if wrd_id != pad_idx:
            if axis == 0:
                dW[wrd_id, :] = np.sum(error.take(group[0], axis=axis), axis=axis)
            else:
                dW[:, wrd_id] = np.sum(error.take(group[0], axis=axis), axis=axis)
