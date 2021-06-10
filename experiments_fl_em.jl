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
include("em/em_fl.jl")

## Setup
d, M, N = 128, 10_000, 10
# SNRs = [1.0, 10.0]
SNRs = [5.0]
αs = collect(logspace(0.0001, 10, 20))
max_em_iters = 20_000
file_suffix = "SNR5"

trials = collect((SNR, α) for SNR in SNRs for α in αs)
trial_data = Dict(:em => Dict(), :gem => Dict())
for trial in trials
    trial_data[:gem][trial] = (
        βs_em = zeros((d, max_em_iters+1)),
        σ2s_em = zeros(max_em_iters+1)
    )
    trial_data[:em][trial] = (
        βs_em = zeros((d, max_em_iters+1)),
        σ2s_em = zeros(max_em_iters+1)
    )
end
println("There are $(length(trials)) trials.")

## Trials
count = Threads.Atomic{Int}(0);
Threads.@threads for trial in trials
    SNR, α = trial
    x, bstar, y, _ = generate_data(d, 1, SNR)
    p = (
        d = d,
        N = N,
        M = M,
        SNR = SNR,
        max_em_iters=max_em_iters,
        β₀ = randn(d)/sqrt(d),
        σ2₀ = 1.0,
        α = α
    )
    nodes = init_EM_nodes(p, bstar)

    storage_gem = get_storage_gem(p)
    compute_β_gem_fl!(storage_gem, nodes, p)
    trial_data[:gem][trial].βs_em .= storage_gem.βs_em
    trial_data[:gem][trial].σ2s_em .= storage_gem.σ2s_em

    storage_gem = get_storage_gem(p)
    compute_β_gem_fl!(storage_gem, nodes, p; ν=0.01)
    trial_data[:em][trial].βs_em .= storage_gem.βs_em
    trial_data[:em][trial].σ2s_em .= storage_gem.σ2s_em


    Threads.atomic_add!(count, 1)
    flush(stdout)
    println("Finished with trial $(count.value)")
end

## Save raw data
BSON.bson("data/fl/em_trials_$file_suffix.bson", trial_data)

## Compute trial stats
trial_data = BSON.load("data/fl/em_trials_$file_suffix.bson")
metrics_em = Dict()
metrics_gem = Dict()

for (tt, trial) in enumerate(trials)
    local SNR, α = trial
    local _, bstar, _, _ = generate_data(d, 1, SNR)
    p = (N=N, M=M, d=d)
    nodes = init_EM_nodes(p, bstar)

    β̂ = trial_data[:gem][trial].βs_em[:,end]
    σ2_hat = trial_data[:gem][trial].σ2s_em[end]

    σ2_error = (trial_data[:gem][trial].σ2s_em .- 1).^2
    β_error = rel_error(trial_data[:gem][trial].βs_em, bstar)

    nll = 0.0
    for node in nodes
        nll += NLL(node.x, node.y, β̂, σ2_hat) / M
    end
    err_final = β_error[end]

    metrics_gem[trial] = Dict(
        "beta_hat" => β̂,
        "beta_error" => vec(β_error),
        "nll" => nll,
        "err" => err_final,
        "σ2" => σ2_hat,
        "σ2_error" => σ2_error,
    )

    β̂ = trial_data[:em][trial].βs_em[:,end]
    σ2_hat = trial_data[:em][trial].σ2s_em[end]

    σ2_error = (trial_data[:em][trial].σ2s_em .- 1).^2
    β_error = rel_error(trial_data[:em][trial].βs_em, bstar)

    for node in nodes
        nll += NLL(node.x, node.y, β̂, σ2_hat) / M
    end
    err_final = β_error[end]

    metrics_em[trial] = Dict(
        "beta_hat" => β̂,
        "beta_error" => vec(β_error),
        "nll" => nll,
        "err" => err_final,
        "σ2" => σ2_hat,
        "σ2_error" => σ2_error,
    )

end


## Save stats
# BSON.bson("data/fl/stats_em_$file_suffix.bson", metrics_em)
# BSON.bson("data/fl/stats_gem_$file_suffix.bson", metrics_gem)
