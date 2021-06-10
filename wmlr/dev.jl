cd(@__DIR__)
Pkg.activate("..")

using BenchmarkTools
using StatsBase
using LinearAlgebra
using Statistics, StatsFuns
using Random
using ReverseDiff: GradientTape, gradient!, compile
using ForwardDiff
using Plots


include("../utils/utils_data.jl")
include("../utils/utils_plot.jl")

##
include("wmlr_cent.jl")
d, N, SNR = 128, 100_000, 10
λ = 0.5
α_max = 1/2λ
α_min = 0.1*α_max
iso = true

x, bstar, y, Σ_sqrt = generate_data(d, N, SNR)
p = (
    d = d,
    N = N,
    SNR = SNR,
    max_wmlr_iters=100,
    β₀ = randn(d)/sqrt(d),
    γ̃=eigen((y.*x)'*(y.*x) / N; sortby=x->-abs(x)).vectors[:,1],
    λ=λ,
    α_max = α_max,
    α_min = α_min,
)
# p.β₀ .= SNR * p.β₀ / norm(p.β₀)
init = (
    p.β₀,
    randn(d)/sqrt(d),
    randn(d)/sqrt(d),
)
data = (x=x, y=y)
storage, cache = get_storage_cache_wmlr(p)

##
compute_β_wmlr!(init, storage, data, p, cache)

##
err = rel_error(storage.βs_wmlr, bstar)
plt = plot(0:p.max_wmlr_iters, err,
    lw=2,
    # ylim=(1e-10,maximum(err_em)),
    yaxis=:log
)
plot!(plt, 0:p.max_wmlr_iters, err_gem)

##
err[end]
β̂ = storage.βs_wmlr[:,end]
σ2_hat = mean(y.^2) - norm(β̂)^2
NLL(x, y, β̂, σ2_hat)
NLL(x, y, β̂)


## Decentralized case
include("wmlr_fl.jl")
d, M, N, SNR = 128, 10_000, 10, 10
λ = 0.75
α_max = 1/2λ
α_min = 0.1*α_max

_, bstar, _, _ = generate_data(d, 1, SNR)
p = (
    d = d,
    N = N,
    M = M,
    SNR = SNR,
    max_wmlr_iters=300,
    β₀ = randn(d) / sqrt(d),
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

##
compute_β_wmlr_fl!(init, storage, nodes, p, cache; debug=true)

##
err = rel_error(storage.βs_wmlr, bstar)
plt = plot(0:p.max_wmlr_iters, err,
    lw=2,
    # ylim=(1e-10,maximum(err_em)),
    yaxis=:log
)

##
node = nodes[1]
(node.y .* node.x)'*(node.y .* node.x) / p.M
