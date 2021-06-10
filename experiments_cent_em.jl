cd(@__DIR__)
Pkg.activate(".")

using BenchmarkTools
using StatsBase
using LinearAlgebra
using Statistics, StatsFuns
using Random
using ReverseDiff: GradientTape, gradient!, compile
using ForwardDiff
using Plots
using BSON
using OrderedCollections
using Statistics

include("utils/utils_data.jl")
include("utils/utils_plot.jl")
include("em/em_cent.jl")
## Setup
d, N = 128, 10_000
max_em_iters = 100
file_suffix = "10k_new"

SNRs = [1.0, 10.0]
# σ2₀s = rand(collect(logspace(0.01, 10, 100)), 5)
# σ2₀s = collect(logspace(0.01, 10, 5))
σ2₀s = [1.0]
αs = collect(logspace(0.0001, 10, 10))
trials = collect((SNR, σ2₀, α) for SNR in SNRs for σ2₀ in σ2₀s for α in αs)
trial_data = Dict(:em => Dict(), :gem => Dict())
for trial in trials
    trial_data[:gem][trial] = (
        βs_em = zeros((d, max_em_iters+1)),
        σ2s_em = zeros(max_em_iters+1)
    )
    if trial[3] == αs[1]
        trial_data[:em][trial] = (
            βs_em = zeros((d, max_em_iters+1)),
            σ2s_em = zeros(max_em_iters+1)
        )
    end
end


## Trials
count = Threads.Atomic{Int}(0);
Threads.@threads for trial in trials
    SNR, σ2₀, α = trial
    x, bstar, y, _ = generate_data(d, N, SNR)
    data = (x=x, y=y)
    p = (
        d = d,
        N = N,
        SNR = SNR,
        max_em_iters=max_em_iters,
        β₀ = randn(d)/sqrt(d),
        σ2₀ = σ2₀,
        α = α
    )
    # p.β₀ .= SNR * p.β₀ / norm(p.β₀)
    storage_gem = get_storage_gem(p)

    compute_β_gem!(storage_gem, data, p)
    trial_data[:gem][trial].βs_em .= storage_gem.βs_em
    trial_data[:gem][trial].σ2s_em .= storage_gem.σ2s_em


    # Only compute EM for 1 α, since α only is used for GEM
    if α == αs[1]
        storage_em, cache = get_storage_cache_em(p)
        compute_β_em!(storage_em, data, p, cache)
        trial_data[:em][trial].βs_em .= storage_em.βs_em
        trial_data[:em][trial].σ2s_em .= storage_em.σ2s_em
    end

    Threads.atomic_add!(count, 1)
    flush(stdout)
    @info "Finished with trial $(count.value)"
end


## Save raw data
BSON.bson("data/cent/em_trials_$(file_suffix).bson", trial_data)


## Compute metrics for paper
metrics_em = Dict()
metrics_gem = Dict()

for (tt, trial) in enumerate(trials)
    local SNR, σ2₀, α = trial
    local x, bstar, y, _ = generate_data(d, N, SNR)

    β̂ = trial_data[:gem][trial].βs_em[:,end]
    σ2_hat = trial_data[:gem][trial].σ2s_em[end]

    σ2_error = (trial_data[:gem][trial].σ2s_em .- 1).^2
    β_error = rel_error(trial_data[:gem][trial].βs_em, bstar)

    nll = NLL(x, y, β̂, σ2_hat)
    err_final = β_error[end]

    metrics_gem[trial] = Dict(
        "beta_hat" => β̂,
        "sigma_hat" => σ2_hat,
        "beta_error" => vec(β_error),
        "sigma_error" => σ2_error,
        "nll" => nll,
        "err" => err_final
    )

    if α == αs[1]
        β̂ = trial_data[:em][trial].βs_em[:,end]
        σ2_hat = trial_data[:em][trial].σ2s_em[end]

        σ2_error = (trial_data[:em][trial].σ2s_em .- 1).^2
        β_error = rel_error(trial_data[:em][trial].βs_em, bstar)

        nll = NLL(x, y, β̂, σ2_hat)
        err_final = β_error[end]

        metrics_em[(SNR, σ2₀)] = Dict(
            "beta_hat" => β̂,
            "sigma_hat" => σ2_hat,
            "beta_error" => β_error,
            "sigma_error" => σ2_error,
            "nll" => nll,
            "err" => err_final
        )
    end

end

## Save data
BSON.bson("data/cent/stats_em_$(file_suffix).bson", metrics_em)
BSON.bson("data/cent/stats_gem_$(file_suffix).bson", metrics_gem)


## Load EM Variance data for confidence intervals
metrics = BSON.load("data/cent/stats_em_$(file_suffix).bson")


## Compute variance with best parameters
N_var = 50
variance_data = Dict(SNR => Dict() for SNR in SNRs)
for SNR in SNRs
    variance_data[SNR] = Dict(i => Dict() for i in 1:N_var)
end

counter = Threads.Atomic{Int}(0);
Threads.@threads for SNR in SNRs
    for trial in 1:N_var
        # Use RNG = trial to re-randomize
        x, bstar, y, _ = generate_data(d, N, SNR; rng=trial)
        data = (x=x, y=y)
        p = (
            d = d,
            N = N,
            SNR = SNR,
            max_em_iters=max_em_iters,
            β₀ = randn(d)/sqrt(d),
            σ2₀ = 1.0,
            α = 1.0,
        )

        storage_em, cache = get_storage_cache_em(p)
        compute_β_em!(storage_em, data, p, cache)


        β_error = rel_error(storage_em.βs_em, bstar)
        β̂ = storage_em.βs_em[:,end]
        σ2_hat = storage_em.σ2s_em[end]
        nll = NLL(x, y, β̂, σ2_hat)
        err_final = β_error[end]

        variance_data[SNR][trial] = Dict(
            "betas" => storage_em.βs_em,
            "beta_hat" => β̂,
            "beta_error" => vec(β_error),
            "nll" => nll,
            "err" => err_final
        )


        Threads.atomic_add!(counter, 1)
        flush(stdout)
        @info "Finished with trial $(counter.value)"
    end
end


## Save variance data
BSON.bson("data/cent/var_em_$file_suffix.bson", variance_data)


## Compute metrics for table
variance_metrics = Dict(SNR => Dict() for SNR in SNRs)
calc_stats(x::Vector{<:Real}) = (mean(x), median(x), std(x), quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0]))

for SNR in SNRs
    nlls = [v["nll"] for (k,v) in variance_data[SNR]][2:end]
    errs = [v["err"] for (k,v) in variance_data[SNR]][2:end]

    # @info "SNR $SNR: \nNLL: $(calc_stats(nlls))\nerr: $(calc_stats(errs))"
    @info "SNR: $SNR \nNLL: $(q25_75(nlls))\nerr: $(q25_75(errs))"
end
