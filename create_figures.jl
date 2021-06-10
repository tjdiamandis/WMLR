cd(@__DIR__)
Pkg.activate(".")

using Plots
using BSON

include("utils/utils_data.jl")

## Table 1 (Centralized)
# Print table 1 info
file_suffix = "10k_new"

function print_table1(table)
    println("\nTable 1")
    table_rows = ["EM", "GEM", "WMLR"]
    for k in keys(table[1])
        for i in 1:3
            method = table_rows[i]
            nll = table[i][k][end-1]
            err = table[i][k][end]
            println("SNR: $k, \tmethod: $method, \tNLL: $nll, \tError: $err")
        end
    end
end

# Load data for analysis
metrics_cent_em = BSON.load("data/cent/stats_em_$file_suffix.bson")
metrics_cent_gem = BSON.load("data/cent/stats_gem_$file_suffix.bson")
metrics_cent_wmlr = BSON.load("data/cent/stats_wmlr_$file_suffix.bson")


# Create table 1
SNRs = [1.0, 10.0]
αs = collect(logspace(0.0001, 10, 10))
table1, trials_best_cent = compute_best_trials_cent(SNRs, αs, metrics_cent_em, metrics_cent_gem, metrics_cent_wmlr)
print_table1(table1)


## Create param plots
# Load WMLR data
metrics_wmlr_snr1_10 = BSON.load("data/fl/stats_wmlr.bson")
metrics_wmlr_snr20 = BSON.load("data/fl/stats_wmlr_snr20.bson")
metrics_wmlr_snr2_5 = BSON.load("data/fl/stats_wmlr_SNR2_5.bson")
metrics_wmlr = merge(metrics_wmlr_snr1_10, metrics_wmlr_snr20, metrics_wmlr_snr2_5)
SNRs = unique([snr for (snr, _) in keys(metrics_wmlr) if snr ≠ 2.0])
trials_best_wmlr = get_best_trials(metrics_wmlr, SNRs)

# Load GEM data
metrics_gem_snr1 = filter(x -> first(x)[1] ≠ 10.0, BSON.load("data/fl/stats_gem_SNR1_10_iter2000.bson"))
metrics_gem_snr10 = BSON.load("data/fl/stats_gem_SNR10_iter5000.bson")
metrics_gem_snr20 = BSON.load("data/fl/stats_gem_SNR20.bson")
metrics_gem_snr5 = BSON.load("data/fl/stats_gem_SNR5.bson")
metrics_gem = merge(metrics_gem_snr1, metrics_gem_snr10, metrics_gem_snr20, metrics_gem_snr5)
trials_best_gem = get_best_trials(metrics_gem, SNRs)

# Load EM data (NOTE: only converges for 1)
metrics_em = filter(x -> first(x)[1] ≠ 10.0, BSON.load("data/fl/stats_em_SNR1_10_iter2000.bson"))
trials_best_em = get_best_trials(metrics_em, [1.0])

for SNR in SNRs
    make_param_plot(trials_best_wmlr, SNR, algo="F-WMLR", param="λ")
    make_param_plot(trials_best_gem, SNR, algo="F-GEM", param="α")
    SNR == 1.0 && make_param_plot(trials_best_em, SNR, algo="F-EM", param="α")
end

## Table 2: Get fastest trials
function print_table_2(table)
    println("\nTable 2")
    table_rows = ["EM", "GEM", "WMLR"]
    for k in keys(table[1])
        for i in 1:3
            method = table_rows[i]
            iters = table[i][k][1]
            iters == Inf && continue
            param = table[i][k][2]
            err = table[i][k][3]
            println("SNR: $k, \tmethod: $method, \titers: $iters, \terror: $err")
        end
    end
end
table2 = [get_fastest_trials(x)[1] for x in (trials_best_em, trials_best_gem, trials_best_wmlr)]
print_table_2(table2)


## Figure 1
trials_fastest_wmlr = get_fastest_trials(trials_best_wmlr)[2]
trials_fastest_gem = get_fastest_trials(trials_best_gem)[2]
trials_fastest_em = get_fastest_trials(trials_best_em)[2]

function get_β_error(trial, n)
    err = trial["beta_error"]
    if length(err) >= n
        return err[1:n]
    else
        return [err; err[end]*ones(n - length(err))]
    end
end


# Figure 1
max_iters = length(trials_fastest_wmlr[10.0]["beta_error"])
iter_range = 0:max_iters-1
rel_l2_norm = [
    get_β_error(trials_fastest_wmlr[10.0], max_iters),
    get_β_error(trials_fastest_wmlr[5.0], max_iters),
    get_β_error(trials_fastest_wmlr[1.0], max_iters),
    get_β_error(trials_fastest_gem[10.0], max_iters),
    get_β_error(trials_fastest_gem[5.0], max_iters),
    get_β_error(trials_fastest_gem[1.0], max_iters),
    get_β_error(trials_fastest_em[1.0], max_iters)
]
labels=[
    "WMLR, SNR = 10",
    "WMLR, SNR = 5",
    "WMLR, SNR = 1",
    "GEM, SNR = 10",
    "GEM, SNR = 5",
    "GEM, SNR = 1",
    "EM, SNR = 1",
]
styles = [:solid, :dashdot, :dash, :solid, :dashdot, :dash, :dash]
colors = [:red, :red, :red, :blue, :blue, :blue, :black]

plt = plot(
    dpi=300,
    title="Distance to β*",
    xlabel="Iteration",
    ylabel="Relative ℓ2 error",
    palette=:Paired_6,
)
for ii in 1:length(rel_l2_norm)
    plot!(plt,
        iter_range,
        rel_l2_norm[ii][1:max_iters],
        lw=2,
        label=labels[ii],
        legend=:topright,
        yaxis=:log,
        linestyle=styles[ii],
        legendfontsize=8,
        topmargin=5Plots.mm,
        linecolor=colors[ii]
    )
end
plt
# savefig(plt, "plots/FL_convergence_3.png")
