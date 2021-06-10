cd(@__DIR__)
Pkg.activate(".")

using BenchmarkTools
using StatsBase
using LinearAlgebra
using Statistics, StatsFuns
using Random
using ReverseDiff: GradientTape, gradient!, compile
using Plots
using BSON
using OrderedCollections
using Statistics


include("utils/utils_data.jl")
include("utils/utils_plot.jl")
include("wmlr/wmlr_cent.jl")

## Setup trials
d, N = 128, 10_000
max_wmlr_iters = 200
# file_suffix = "N100k_T100"
file_suffix = "10k_new"

SNRs = [1.0, 10.0]
λs = collect(logspace(0.1, 2, 10))
trials = collect((SNR, λ) for SNR in SNRs for λ in λs)
trial_data = Dict()
for trial in trials
    trial_data[trial] = (
        βs_wmlr = zeros((d, max_wmlr_iters+1)),
        objs = zeros(max_wmlr_iters+1)
    )
end

## Trials
# count = Threads.Atomic{Int}(0);
count = 0
for trial in trials
    SNR, λ = trial
    x, bstar, y, _ = generate_data(d, N, SNR)
    α_max = 1/2λ
    α_min = 0.1*α_max

    p = (
        d = d,
        N = N,
        SNR = SNR,
        max_wmlr_iters=max_wmlr_iters,
        β₀ = randn(d)/sqrt(d),
        γ̃=eigen((y.*x)'*(y.*x) / N; sortby=x->-abs(x)).vectors[:,1],
        λ=λ,
        α_max = α_max,
        α_min = α_min,
    )
    init = (
        p.β₀,
        randn(d)/sqrt(d),
        randn(d)/sqrt(d),
    )
    data = (x=x, y=y)
    storage, cache = get_storage_cache_wmlr(p)

    compute_β_wmlr!(init, storage, data, p, cache)
    trial_data[trial].βs_wmlr .= storage.βs_wmlr
    trial_data[trial].objs .= storage.objs

    # Threads.atomic_add!(count, 1)
    global count += 1
    flush(stdout)
    println("Finished with trial $(count)")
end

## Save full data
BSON.bson("data/cent/wmlr_trials_$file_suffix.bson", trial_data)


## Compute metrics for paper
metrics = Dict()

for (tt, trial) in enumerate(trials)
    local SNR, λ = trial
    local x, bstar, y, _ = generate_data(d, N, SNR)

    β̂ = trial_data[trial].βs_wmlr[:,end]
    σ2_hat = mean(y.^2) - norm(β̂)^2

    β_error = rel_error(trial_data[trial].βs_wmlr, bstar)

    nll = NLL(x, y, β̂, σ2_hat)
    err_final = β_error[end]

    metrics[trial] = Dict(
        "beta_hat" => β̂,
        "beta_error" => vec(β_error),
        "objs" => trial_data[trial].objs,
        "nll" => nll,
        "err" => err_final
    )

end

## Save Stats Data
BSON.bson("data/cent/stats_wmlr_$file_suffix.bson", metrics)
metrics = BSON.load("data/cent/stats_wmlr_$file_suffix.bson")

## Compute variance with best parameters
best_trials = Dict(x => get_best_trials(metrics, x; old=false) for x in SNRs)


count = 0
N_var = 50
variance_data = Dict(SNR => Dict() for SNR in SNRs)
for SNR in SNRs
    variance_data[SNR] = Dict(i => Dict() for i in 1:N_var)
end

for SNR in SNRs
    if length(best_trials[SNR]) > 1
        @warn "Multiple best trials"
    end

    λ = first(keys(OrderedDict(sort(collect(best_trials[SNR]), by=x->x[2]["nll"]))))[2]

    for trial in 1:N_var
        x, bstar, y, _ = generate_data(d, N, SNR; rng=trial)
        α_max = 1/2λ
        α_min = 0.1*α_max

        p = (
            d = d,
            N = N,
            SNR = SNR,
            max_wmlr_iters=max_wmlr_iters,
            β₀ = randn(d)/sqrt(d),
            γ̃=eigen((y.*x)'*(y.*x) / N; sortby=x->-abs(x)).vectors[:,1],
            λ=λ,
            α_max = α_max,
            α_min = α_min,
        )
        init = (
            p.β₀,
            randn(d)/sqrt(d),
            randn(d)/sqrt(d),
        )
        data = (x=x, y=y)
        storage, cache = get_storage_cache_wmlr(p)

        compute_β_wmlr!(init, storage, data, p, cache)
        β_error = rel_error(storage.βs_wmlr, bstar)
        β̂ = storage.βs_wmlr[:,end]
        σ2_hat = mean(y.^2) - norm(β̂)^2
        nll = NLL(x, y, β̂, σ2_hat)
        err_final = β_error[end]

        variance_data[SNR][trial] = Dict(
            "betas" => storage.βs_wmlr,
            "objs" => storage.objs,
            "beta_hat" => β̂,
            "beta_error" => vec(β_error),
            "nll" => nll,
            "err" => err_final
        )


        global count += 1
        flush(stdout)
        println("Finished with trial $(count)")
    end
end


##
BSON.bson("data/cent/var_wmlr_$file_suffix.bson", variance_data)


##


variance_metrics = Dict(SNR => Dict() for SNR in SNRs)
calc_stats(x::Vector{<:Real}) = (mean(x), median(x), std(x), quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0]))

for SNR in SNRs
    nlls = [v["nll"] for (k,v) in variance_data[SNR]][2:end]
    errs = [v["err"] for (k,v) in variance_data[SNR]][2:end]

    # @info "SNR $SNR: \nNLL: $(calc_stats(nlls))\nerr: $(calc_stats(errs))"
    @info "SNR: $SNR \nNLL: $(q25_75(nlls))\nerr: $(q25_75(errs))"
end
