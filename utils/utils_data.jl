"""
Generates data (x,y) from the 2MLR model

Returns x, y, true regressor β⋆, sqrt of variance √∑
"""
function generate_data(d, N, SNR; rng=0, iso=true, κ=10)
    Random.seed!(rng)
    if iso
        Σ_sqrt = LinearAlgebra.I
    else
        Σ_seed = randn((d,d))
        Q_ = qr(Σ_seed).Q
        D_ = 1 .+ (κ-1)*rand(d)
        Σ_sqrt = Q_ * diagm(sqrt.(D_))
    end
    x =  (Σ_sqrt * randn((d, N)))'
    bstar = randn(d)
    SNR = SNR == Inf ? 1.0 : SNR
    bstar .= SNR .* bstar ./ norm(bstar)
    y = x * bstar

    if SNR == Inf
        y .= rand((-1, 1), N) .* y
    else
        y .= rand((-1, 1), N) .* y .+ randn((N))
    end

    return x, bstar, y, Σ_sqrt
end


"""
Cacluate negative log likelihood under the symmetric 2MLR model
"""
function NLL(x, y, β̂, σ2=1.0)
    σ2 = σ2 > 0 ? σ2 : 1.0

    LL = 0.0
    for n in 1:length(y)
        xi = @view(x[n,:])
        yi = y[n]
        ymβ = -(yi - dot(xi, β̂))^2 / 2σ2
        ypβ = -(yi + dot(xi, β̂))^2 / 2σ2
        LL += logaddexp(ymβ+log(1/(2*sqrt(2*π*σ2))), ypβ+log(1/(2*sqrt(2*π*σ2))))
    end
    return -LL/length(y)
end


"""
Generates n points logarithmically spaced between x to y (inclusive)
"""
logspace(x, y, n) = (10^z for z in range(log10(x), log10(y), length=n))


"""
Determines the number of iterations until convergence

Def: the sequence x has converged at iterate t₀ if for all t ≧ t₀, x(t) is no
     more than rtol bigger than the final value x(T)
"""
function iter_conv(errors; rtol=1.05)
    min_err = minimum(errors)
    min_inds = findall(x->(x <= min_err*rtol), errors)
    min_ind = min_inds[1]
    for ii = min_inds[1]:length(errors)
        if !(ii in min_inds)
            min_ind = ii+1
        end
    end
    return min_ind
end


"""
Returns all trials xᵢ where xᵢ(T) is no more than 50% bigger than minᵢxᵢ(T)
"""
function get_best_trials(metrics, SNRs; old=false)
    trials_best = Dict()
    if old
        for SNR in SNRs
            min_err = minimum(x->(isnan(x) ? Inf : x), [v[:err] for (k,v) in metrics if k[1] == SNR])
            for trial in keys(metrics)
                if trial[1] == SNR && metrics[trial][:err] < min_err*1.5
                    trials_best[trial] = metrics[trial]
                end
            end
        end
        return sort(OrderedDict(trials_best); by=x->x[2])
    end

    for SNR in SNRs
        min_err = minimum([v["err"] for (k,v) in metrics if k[1] == SNR])
        for trial in keys(metrics)
            if trial[1] == SNR && metrics[trial]["err"] < min_err*1.5
                trials_best[trial] = metrics[trial]
            end
        end
    end
    return sort(OrderedDict(trials_best); by=x->x[2])
end


"""
Returns fastest trials (& hyperparameter) judged in terms of iter_conv
"""
function get_fastest_trials(trials)
    iters = Dict(
        snr => (Inf, 0.0, 0.0) for snr in SNRs
    )
    trials_fastest = Dict()
    for (key, trial) in trials
        iters_trial = iter_conv(trial["beta_error"]; rtol=rtol)
        if iters_trial < iters[key[1]][1]
            iters[key[1]] = (iters_trial, key[2], trial["err"])
            trials_fastest[key[1]] = trial
        end
    end
    return iters, trials_fastest
end


"""
Returns best trials in centralized setting (table 1)
"""
function compute_best_trials_cent(SNRs, αs, m_em, m_gem, m_wmlr)
    tab_wmlr = Dict(
        SNR => (Inf, Inf, Inf) for SNR in SNRs
    )
    tab_gem = Dict(
        SNR => (Inf, Inf, Inf, Inf) for SNR in SNRs
    )

    tab_em = Dict(
        SNR => (Inf, Inf, Inf) for SNR in SNRs
    )

    tb_wmlr = Dict()
    tb_em = Dict()
    tb_gem = Dict()

    for trial in keys(m_wmlr)
        SNR, λ = trial
        nll = m_wmlr[trial]["nll"]
        if nll < tab_wmlr[SNR][end-1]
            tab_wmlr[SNR] = (λ, nll, m_wmlr[trial]["err"])
            tb_wmlr[SNR] = m_wmlr[trial]
        end
    end

    for trial in keys(m_gem)
        SNR, σ2₀, α = trial
        nll = m_gem[trial]["nll"]
        if nll < tab_gem[SNR][end-1]
            tab_gem[SNR] = (σ2₀, α, nll, m_gem[trial]["err"])
            tb_gem[SNR] = m_gem[trial]
        end

        if α == αs[1]
            nll = m_em[(SNR, σ2₀)]["nll"]
            if nll < tab_em[SNR][end-1]
                tab_em[SNR] = (σ2₀, nll, m_em[(SNR, σ2₀)]["err"])
                tb_em[SNR] = m_em[(SNR, σ2₀)]
            end
        end
    end
    return (tab_em, tab_gem, tab_wmlr), (tb_em, tb_gem, tb_wmlr)
end


# Get lower and upper quartiles
q25_75(x) = quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])[2:2:4]
