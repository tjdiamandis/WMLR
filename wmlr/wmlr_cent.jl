include("wmlr_objective.jl")

"""
Computes WMLR in centralized setting
"""
function compute_β_wmlr!(init, storage, data, p, cache; ∇f=nothing)
    if isnothing(∇f)
        ∇f! = get_grad_function_rd(data, p)
    else
        ∇f! = ∇f
    end
    vars = deepcopy(init)

    storage.objs[1] = f(init[1], init[2], init[3], data, p, cache)
    results = (similar(init[1]), similar(init[2]), similar(init[3]))

    # flush(stdout)
    # println("\nStarting iterations...")
    # println("00 $(storage.objs[1])")
    for tt in 1:p.max_wmlr_iters
        _gda_step(∇f!, vars, results, p)

        storage.objs[tt+1] = f(vars[1], vars[2], vars[3], data, p, cache)
        storage.βs_wmlr[:,tt+1] .= vars[1]

        # if tt % 10 == 0
        #     flush(stdout)
        #     println("$tt $(storage.objs[tt+1])")
        # end
    end
end


"""
Allocates storage (iterates) and cache (for f computation) for MLR algorithm
"""
function get_storage_cache_wmlr(p)
    storage = (
        βs_wmlr = hcat(p.β₀, Matrix{Float64}(undef, (d, p.max_wmlr_iters))),
        objs = Vector{Float64}(undef, p.max_wmlr_iters+1)
    )
    cache = (
        y_gen = zeros(p.N),
        rand_k = zeros(p.N),
        rand_noise = zeros(p.N)
    )
    return storage, cache
end
