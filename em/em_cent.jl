# Traditional EM
"""
Computes EM for 2MLR
"""
function compute_β_em!(storage, data, p, cache; iso=true)
    βs = storage.βs_em
    x, y = data.x, data.y
    if !iso
        cache.Σ .= cov(x)
        # cholesky!(cache.Σ)
        Σ = cholesky(cache.Σ)
    else
        Σ = nothing
    end

    _em_iters(βs, x, y, p, cache; iso=iso, Σ=Σ)

end


# Helper function for above (non-allocating)
function _em_iters(βs, x, y, p, cache; iso=true, Σ=I)
    σ2 = p.σ2₀
    @inbounds for n in 1:p.max_em_iters
        σ2_new = zero(eltype(y))
        # uses β_new as a cache for itermediate values
        β_new = @view(βs[:,n+1])
        β = @view(βs[:,n])

        @inbounds for ii in 1:p.N
            xi = @view(x[ii,:])
            yi = y[ii]
            ymβ = (yi - dot(xi, β))
            ypβ = (yi + dot(xi, β))
            wi = logistic(-(ymβ^2 - ypβ^2)/2σ2)

            @. β_new += (2wi - 1) * yi * xi
            σ2_new += (wi*ymβ^2 + (1-wi)*ypβ^2)
        end

        β_new .= β_new ./ p.N
        !iso && ldiv!(Σ, β_new)

        σ2 = p.N / σ2_new
    end
end


"""
Allocates storage for EM
"""
function get_storage_cache_em(p)
    storage = (
        βs_em = [p.β₀ zeros((d, p.max_em_iters))],
        σ2s_em = [p.σ2₀; zeros(p.max_em_iters)],
    )
    cache = (
        Σ = zeros(d,d),
    )
    return storage, cache
end


# -----------------------------------------------------
# -------------------- Gradient EM --------------------
# -----------------------------------------------------
"""
Computes gradient ∇Q(θ'|θ)
"""
function dQ!(results, vars, data, p)
    Δβ = results[1]
    β = vars[1]
    @inbounds for ii in 1:p.d
        Δβ[ii] = 0.0
    end
    Δσ2 = 0.0
    σ2 = vars[2][1]

    @inbounds for ii in 1:p.N
        xi = @view(data.x[ii,:])
        yi = data.y[ii]

        ymβ = (yi - dot(xi, β))
        ypβ = (yi + dot(xi, β))
        w_i = logistic(-(ymβ^2 - ypβ^2)/2σ2)
        coeff_β = (w_i * ymβ + (w_i - 1) * ypβ)/ (2*σ2*p.N)
        @. Δβ +=  coeff_β * xi

        Δσ2 += (w_i * ymβ^2  + (1 - w_i) * ypβ^2)
    end
    Δσ2 = (Δσ2/p.N)* 1/(2σ2^2) - 1/2σ2
    results[2][1] = Δσ2
end


"""
Computes Gradient EM
"""
function compute_β_gem!(storage, data, p)
    βs = storage.βs_em
    σ2s = storage.σ2s_em

    results = (
        zeros(p.d),
        zeros(1)
    )
    for tt in 1:p.max_em_iters
        # grad step
        vars = (@view(βs[:,tt]), @view(σ2s[tt]))
        dQ!(results, vars, data, p)

        # update
        @inbounds for ii in 1:p.d
            βs[ii,tt+1] = βs[ii,tt] + p.α*results[1][ii]
        end
        σ2s[tt+1] = max(σ2s[tt] + p.α*results[2][1], 1e-4)
    end
end


"""
Allocates storage for GEM
"""
function get_storage_gem(p)
    storage = (
        βs_em = [p.β₀ zeros((d, p.max_em_iters))],
        σ2s_em = [p.σ2₀; zeros(p.max_em_iters)],
    )
    return storage
end
