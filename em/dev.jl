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
## EM Testing
include("em_cent.jl")

d, N, SNR = 128, 100_000, 0.5
iso = true
κ = 10
x, bstar, y, Σ_sqrt = generate_data(d, N, SNR; iso=iso, κ=κ)
Random.seed!(1)
p = (
    d = d,
    N = N,
    SNR = SNR,
    max_em_iters=100,
    β₀ = randn(d),
    σ2₀ = 1,
    α = 0.1
)
p.β₀ .= SNR * p.β₀ / norm(p.β₀)

data = (x=x, y=y)

storage = (
    βs_em = [p.β₀ zeros((d, p.max_em_iters))],
    σ2s_em = [p.σ2₀; zeros(p.max_em_iters)],
)

cache = (
    Σ = zeros(d,d),
    rhs = similar(y),
)


@time σ2 = compute_β_em!(storage, data, p, cache; iso=iso)

βs_em = storage.βs_em
scale = 1/norm(bstar)
err_em_p = vec(sqrt.(sum(abs.(βs_em .+ bstar).^2, dims=1))) * scale
err_em_m = vec(sqrt.(sum(abs.(βs_em .- bstar).^2, dims=1))) * scale
err_em = err_em_p[end] < err_em_m[end] ? err_em_p : err_em_m

plot(0:p.max_em_iters, err_em,
    lw=2,
    # ylim=(1e-10,maximum(err_em)),
    yaxis=:log
)


Δβ, β = randn(d), bstar
σ2 = [1.0]
@btime dQ!(Δβ, Δσ2, β, σ2, x, y, p)
@time dQ!(Δβ, β, σ2, β_old, σ2_old, x, y, p)
norm(Δβ)

results = (
    similar(β),
    zeros(1)
)
vars = (
    β,
    σ2
)

@time dQ!(results, vars, data, p)
f() = dQ_step!(vars, results, data, p)
@time  f()
results
vars
# Σ_sqrt = randn((d,d))
# Q_ = qr(Σ_sqrt).Q
# D_ = 1 .+ 9*rand(d)
# Σ = Q_ * diagm(sqrt.(D_))
# x = Σ * randn(d, N)
#
# Σx = cov(x')
# eigvals(Σx)

@time compute_β_gem!(storage, data, p)


## FL EM/GEM testing
include("em_fl.jl")
d, M, N, SNR = 128, 10_000, 10, 10
_, bstar, _, _ = generate_data(d, 1, SNR)
Random.seed!(1)
p = (
    d = d,
    N = N,
    M = M,
    SNR = SNR,
    max_em_iters=1000,
    β₀ = randn(d) / sqrt(d),
    σ2₀ = 1.0,
    α = 2
)

nodes = init_EM_nodes!(p, bstar)
storage = get_storage_gem(p)

##
compute_β_gem_fl!(storage, nodes, p)
compute_β_gem_fl!(storage, nodes, p; ν=(0.01*SNR), debug=false)

##
err = rel_error(storage.βs_em, bstar)
plot(0:p.max_em_iters, err,
    lw=2,
    # ylim=(1e-10,maximum(err_em)),
    yaxis=:log
)

# storage.βs_em

##
