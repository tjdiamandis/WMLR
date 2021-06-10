"""
Returns relative error for each β̂
"""
function rel_error(βs, bstar)
    scale = 1/norm(bstar)
    err_p = vec(sqrt.(sum(abs2, (βs .+ bstar), dims=1))) * scale
    err_m = vec(sqrt.(sum(abs2, (βs .- bstar), dims=1))) * scale
    err = err_p[end] < err_m[end] ? err_p : err_m
    return err
end


"""
Plots convergence for a particular SNR over parameters.

    Saves fig at ./plots/(algo)_(SNR)_(param).png
"""
function make_param_plot(trials_best, SNR; algo="F-WMLR", param="λ")
    plt = plot(dpi=300, yaxis=:log);
    for trial in keys(trials_best)
        trial[1] != SNR && continue
        plot!(plt,
            0:length(trials_best[trial]["beta_error"])-1,
            trials_best[trial]["beta_error"],
            label="$param = $(round(trial[2], digits=2))",
            lw=2,
            yaxis="Relative ℓ2 error",
            xaxis="Iteration",
            title="$algo Convergence, SNR = $SNR",
            legendfontsize=8,

        )
    end
    savefig(plt, "plots/$(algo)_$(SNR)_$(param).png")
end
