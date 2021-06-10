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


include("utils/utils_data.jl")
include("utils/utils_plot.jl")
include("wmlr/wmlr_fl.jl")

## Setup trials
d, M, N, SNR = 128, 10_000, 10, 10
max_wmlr_iters = 500
file_suffix = "snr20"

SNRs = [20.0]
λs = collect(logspace(0.1, 2, 20))
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
    _, bstar, _, _ = generate_data(d, 1, SNR)
    α_max = 1/2λ
    α_min = 0.1*α_max

    p = (
        d = d,
        N = N,
        M = M,
        SNR = SNR,
        max_wmlr_iters=max_wmlr_iters,
        β₀ = randn(d)/sqrt(d),
        γ̃=zeros(d),
        λ=λ,
        α_max = α_max,
        α_min = α_min,
    )
    init = (
        p.β₀,
        randn(d)/sqrt(d),
        randn(d)/sqrt(d),
    )

    storage, cache = get_storage_cache_wmlr(p)
    nodes = init_WMLR_nodes(p, bstar)

    compute_β_wmlr_fl!(init, storage, nodes, p, cache; debug=true)
    trial_data[trial].βs_wmlr .= storage.βs_wmlr
    trial_data[trial].objs .= storage.objs

    global count += 1
    flush(stdout)
    println("Finished with trial $(count) of $(length(trials))")
end

## Save raw data
BSON.bson("data/fl/wmlr_trials_$file_suffix.bson", trial_data)


## Compute metrics for paper
metrics = Dict()

for (tt, trial) in enumerate(trials)
    local SNR, λ = trial
    local _, bstar, _, _ = generate_data(d, 1, SNR)

    β̂ = trial_data[trial].βs_wmlr[:,end]
    y2_bar = 0.0
    p = (N=N, M=M, d=d)
    nodes = init_WMLR_nodes(p, bstar)
    for node in nodes
        y2_bar += sum(node.y.^2) / M
    end
    σ2_hat = y2_bar - norm(β̂)^2

    β_error = rel_error(trial_data[trial].βs_wmlr, bstar)

    nll = 0.0
    for node in nodes
        nll += NLL(node.x, node.y, β̂, σ2_hat) / M
    end

    # In case we stop early
    err_final = β_error[β_error .> 0][end]

    metrics[trial] = Dict(
        "beta_hat" => β̂,
        "beta_error" => vec(β_error),
        "objs" => trial_data[trial].objs,
        "nll" => nll,
        "err" => err_final
    )

end

## Create Param Plot



iter_conv(metrics_wmlr[(1.0, 0.4133113830244109)]["beta_error"]; rtol=tol)
iter_conv(metrics_wmlr[(1.0, 0.35302227018072674)]["beta_error"]; rtol=tol)
iter_conv(metrics_wmlr[(5.0, 0.4133113830244109)]["beta_error"]; rtol=tol)
iter_conv(metrics_wmlr[(5.0, 0.35302227018072674)]["beta_error"]; rtol=tol)
iter_conv(metrics_wmlr[(10.0, 0.4133113830244109)]["beta_error"]; rtol=tol)
iter_conv(metrics_wmlr[(10.0, 0.35302227018072674)]["beta_error"]; rtol=tol)
iter_conv(metrics_wmlr[(20.0, 0.4133113830244109)]["beta_error"]; rtol=tol)
iter_conv(metrics_wmlr[(20.0, 0.35302227018072674)]["beta_error"]; rtol=tol)
