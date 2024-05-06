include("structures.jl")

conv(x::GraphNode, w::GraphNode) = BroadcastedOperator(conv, x, w)

function forward(::BroadcastedOperator{typeof(conv)}, x, w)
    
    stride = 1
    (H, W, C, _) = size(x)
    (WH, WW, _, K) = size(w)

    cnn_h = Int(floor((H - WH) / stride)) + 1
    cnn_w = Int(floor((W - WW) / stride)) + 1
    
    cnn_out = zeros(cnn_h, cnn_w, K, 1)
    for c in 1:C
        x2c = im2col(x[:,:,c,1], WH, WW)
        for k in 1:K
            cnn_out[:,:,k,1] .+= reshape(reshape(w[:,:,1,k], 1, WH*WW) * x2c, cnn_h, cnn_w, 1, 1)
        end
    end
    return cnn_out
end

function im2col(A, n, m)
    M, N = size(A)
    B = Array{eltype(A)}(undef, m*n, (M-m+1)*(N-n+1))
    indx = reshape(1:M*N, M, N)[1:M-m+1, 1:N-n+1]
    for (i, value) in enumerate(indx)
        for j = 0:n-1
            @views B[(i-1)*m*n+j*m+1 : (i-1)*m*n+(j+1)*m] = A[value+j*M:value+m-1+j*M]
        end 
    end
    return B
end

function backward(::BroadcastedOperator{typeof(conv)}, x, w, g)

    (H, W, C, _) = size(x)
    (WH, WW, _, K) = size(w)

    g_x = zeros(H, W, C, 1)
    g_kernels = zeros(WH, WW, 1, K)

    g_h, g_w, _, _ = size(g)

    for k in 1:K
        for k_h in 1:WH
            for k_w in 1:WW
                g_kernels[k_h, k_w, 1, k] = sum(g[:, :, k, 1] .* x[k_h:k_h+g_h-1, k_w:k_w+g_w-1, :, 1])
            end
        end
    end

    for c in 1:C
        for i in 1:g_h
            for j in 1:g_w
                g_field = g[i, j, :, :]
                g_x[i:i-1+WH, j:j-1+WW, c, :] .+= sum(reshape(g_field, 1, 1, K, 1) .* reshape(w, WH, WW, K, :), dims=3)
            end
        end
    end

    return tuple(g_x, g_kernels)
end


maxpool(x::GraphNode) = BroadcastedOperator(maxpool, x)
function forward(::BroadcastedOperator{typeof(maxpool)}, x)
    mH = 2
    mW = 2
    (H, W, C, _) = size(x)

    outH = Int(H / mH)
    outW = Int(W / mW)

    out = zeros(outH, outW, C, 1)
    field = zeros(mH, mW)
    outIterH = 1
    outIterW = 1
    for i in 1:mH:H
        for j in 1:mW:W
            field = x[i:(i+mH-1), j:(j+mW-1), :, :]
            maxValues = maximum(field, dims=(1, 2))
            out[outIterH, outIterW, :, :] = maxValues
            outIterW += 1
        end
        outIterW = 1
        outIterH += 1
    end

    return out
end

function backward(::BroadcastedOperator{typeof(maxpool)}, x, g)
    mH = 2
    mW = 2
    (H, W, C, _) = size(x)

    out = zeros(H, W, C, 1)
    field = zeros(mH, mW)

    mpH = Int(H / mH)
    mpW = Int(W / mW)

    for i in 1:mpH
        for j in 1:mpW
            field = x[(i-1)*mH+1:i*mH, (j-1)*mW+1:j*mW, :, :]
            max_elem_idx = argmax(field, dims=(1, 2))
            for c in 1:C
                idx_i, idx_j = max_elem_idx[c][1], max_elem_idx[c][2]
                out[idx_i+mH*(i-1), idx_j+mW*(j-1), c, 1] = g[i, j, c]
            end
        end
    end
    return out
end
