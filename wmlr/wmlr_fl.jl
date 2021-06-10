include("wmlr_objective.jl")
include("wmlr_cent.jl")

struct WMLR_Node{T}
    x::AbstractMatrix{T}
    y::AbstractVector{T}
    c::Int
end

struct WMLR_Node_Computed
    ∇f!::Function
end


"""
init nodes with their data and cluster assignments
"""
function init_WMLR_nodes(p, bstar)
    nodes = Vector{WMLR_Node}(undef, p.M)

    for m in 0:p.M-1
        x_node = randn(p.N, p.d)
        cluster = rand((-1,1))
        y_node = cluster .* x_node * bstar .+ randn(p.N)

        nodes[m+1] = WMLR_Node(
            x_node, y_node, cluster,
        )
        # if (m+1) % 1000 == 0
        #     flush(stdout)
        #     println("Created $(m+1) nodes")
        # end
    end
    return nodes
end


"""
Computes γ̃ from 𝖤[y²xxᵀ], collected from nodes
"""
function set_gamma_tilde!(nodes::Vector{WMLR_Node}, p)
    mat = zeros(p.d, p.d)
    for node in nodes
        mat .+= (node.y .* node.x)'*(node.y .* node.x) / p.M
    end
    p.γ̃ .= eigen(mat; sortby=x->-abs(x)).vectors[:,1]
end


"""
Computes precompiled gradient for each node. Stores ∇f in new struct bc struct
    is immutable for memory efficiency
"""
function get_gradients(nodes::Vector{WMLR_Node}, p)
    computed_nodes =  Vector{WMLR_Node_Computed}(undef, p.M)
    results = (
        Vector{Float64}(undef, p.d),
        Vector{Float64}(undef, p.d),
        Vector{Float64}(undef, p.d),
    )
    for ii in 1:p.M
        node = nodes[ii]
        f_tape = GradientTape((β, γ₁, γ₂) -> f(β, γ₁, γ₂, (x=node.x, y=node.y), p), results)
        compiled_f_tape = compile(f_tape)
        ∇f!(results, inputs) = gradient!(results, compiled_f_tape, inputs)
        computed_nodes[ii] = WMLR_Node_Computed(∇f!)
    end
    return computed_nodes
end


"""
Computes WMLR in the FL setting
"""
function compute_β_wmlr_fl!(init, storage, nodes::Vector{WMLR_Node}, p, cache; debug=false)
    βs = storage.βs_wmlr
    objs = storage.objs

    vars = deepcopy(init)
    vars_next = deepcopy(init)
    # objs[1] = f(init[1], init[2], init[3], data, p, cache)
    results = (similar(init[1]), similar(init[2]), similar(init[3]))

    set_gamma_tilde!(nodes, p)
    computed_nodes = get_gradients(nodes, p)

    # Compute first objective
    objs[1] = mean([
        f(vars[1], vars[2], vars[3], (x=node.x, y=node.y), p, cache) / p.M
        for node in nodes
    ])
    if debug
        flush(stdout)
        println("\nStarting iterations...")
        println("00 $(storage.objs[1])")
    end
    for tt in 1:p.max_wmlr_iters
        for var in vars_next
            var .= zeros(size(var))
        end

        objs[tt+1] = 0.0
        for m in 1:p.M
            node = computed_nodes[m]
            node.∇f!(results, vars)

            # Computes average of next iterates from nodes
            @inbounds for ii in 1:p.d
                vars_next[1][ii] += (vars[1][ii] - p.α_min*results[1][ii])/p.M
                vars_next[2][ii] += (vars[2][ii] + p.α_max*results[2][ii])/p.M
                vars_next[3][ii] += (vars[3][ii] + p.α_max*results[3][ii])/p.M
            end
            objs[tt+1] += f(vars[1], vars[2], vars[3], (x=nodes[m].x, y=nodes[m].y), p, cache) / p.M
        end

        vars[1] .= vars_next[1]
        vars[2] .= vars_next[2]
        vars[3] .= vars_next[3]

        βs[:,tt+1] .= vars[1]

        if abs(objs[tt+1]) > 1e4
            flush(stdout)
            println("Diverged -- breaking at iter $tt")
            break
        elseif abs(objs[tt+1]) < 1e-8
            flush(stdout)
            println("Converged -- breaking at iter $tt")
            break
        end

        if debug && tt % 10 == 0
            flush(stdout)
            println("$tt $(storage.objs[tt+1])")
        end
    end
end
