include("em_cent.jl")

# Node stores its own data and cluster
struct EM_Node{T}
    x::AbstractMatrix{T}
    y::AbstractVector{T}
    c::Int
end


"""
Initializes EM nodes with data and cluster assignment
"""
function init_EM_nodes(p, bstar)
    nodes = Vector{EM_Node}(undef, p.M)

    for m in 0:p.M-1
        x_node = randn(p.N, p.d)
        cluster = rand((-1,1))
        y_node = cluster .* x_node * bstar .+ randn(p.N)

        # # Precompile gradient
        # results = (
        #     Vector{Float64}(undef, p.d),
        #     Vector{Float64}(undef, 1),
        #     Vector{Float64}(undef, p.d),
        #     Vector{Float64}(undef, 1),
        # )
        # Qtape = GradientTape((β, σ2, β_old, σ2_old) -> Q(β, σ2, β_old, σ2_old, x_node, y_node, p), results)
        # compiled_tape = compile(Qtape)
        # ∇f!(results, inputs) = gradient!(results, compiled_tape, inputs)
        nodes[m+1] = EM_Node(
            x_node, y_node, cluster,
        )
        # if (m+1) % 1_000 == 0
        #     flush(stdout)
        #     println("Created $(m+1) nodes")
        # end
    end
    return nodes
end


"""
Compute Federated EM/GEM
"""
function compute_β_gem_fl!(storage, nodes::Vector{EM_Node}, p; ν=Inf, debug=false)
    # ν = Inf -> GEM case (always do 1 inner max step)
    βs = storage.βs_em
    σ2s = storage.σ2s_em

    results = (
        zeros(p.d),
        zeros(1)
    )

    idx = 1
    for tt in 1:p.max_em_iters
        vars = (@view(βs[:,idx]), @view(σ2s[idx]))
        ∇β_norm = Inf

        # Inner maximization (ν non-infinite)
        # ν = Inf ⇒ GEM (while loop never repeats)
        inner_count = 0
        while ∇β_norm >= ν && inner_count < 50
            for m in 1:p.M
                node = nodes[m]
                # grad calc
                dQ!(results, vars, (x=node.x, y=node.y), p)

                # send back to server and compute average
                @inbounds for ii in 1:p.d
                    βs[ii,idx+1] += (βs[ii,idx] + p.α*results[1][ii]) / p.M
                end
                σ2s[idx+1] += max(σ2s[idx] + p.α*results[2][1], 1e-4) / p.M
            end

            # Track inner norm & update iteration counters
            ∇β_norm = norm((βs[:,idx+1] - βs[:,idx])/p.α)
            inner_count += 1
            idx += 1
            idx > p.max_em_iters && break

            if debug
                flush(stdout)
                println(∇β_norm)
                idx % 10 == 0 && println("Finished iter $idx")
            end
        end

        if debug
            flush(stdout)
            println("Inner iterations = $inner_count")
        end
        idx > p.max_em_iters && break
    end
end
