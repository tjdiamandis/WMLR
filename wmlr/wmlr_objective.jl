"""
Computes 𝖤[ψ_{β, γᵢ}(x, y)]
"""
function E_psi(β::AbstractVector{T}, γ₁::AbstractVector{T}, γ₂::AbstractVector{T}, data, p) where {T}
    arg1, arg2, ret = zero(T), zero(T), zero(T)
    @inbounds for ii in 1:length(data.y)
        x_i = @view(data.x[ii,:])
        y_i = data.y[ii]
        arg1 = y_i * dot(γ₁, x_i)
        arg2 = y_i * dot(γ₂, x_i)
        # function ψ, as defined in paper
        ret += (logaddexp(-arg1, arg1) - logaddexp(-arg2, arg2))
    end
    return ret / length(data.y)
end


"""
Non-allocating f(β, γᵢ, data)
"""
function f(β::AbstractVector{T}, γ₁::AbstractVector{T}, γ₂::AbstractVector{T}, data, p, cache) where {T}
    x_gen = data.x

    # Compute regularization (unallocating)
    reg = zero(T)
    @inbounds for ii in 1:p.d
        reg += (γ₁[ii] - p.γ̃[ii]).^2
        reg += (γ₂[ii] - p.γ̃[ii]).^2
    end
    reg *= p.λ/2

    # Calculate y_generator (unallocating)
    rand!(cache.rand_k, (-1,1))
    randn!(cache.rand_noise)

    mul!(cache.y_gen, x_gen, β)
    cache.y_gen .= cache.rand_k .* cache.y_gen .+ cache.rand_noise
    return E_psi(β, γ₁, γ₂, data, p) - E_psi(β, γ₁, γ₂, (x=x_gen, y=cache.y_gen), p) - reg
end


"""
Allocating version of f(β, γᵢ, data) for reverse mode AD
"""
function f(β::AbstractVector{T}, γ₁::AbstractVector{T}, γ₂::AbstractVector{T}, data, p) where {T}
    x_gen = data.x

    # Compute regularization
    reg = zero(T)
    @inbounds for ii in 1:p.d
        reg += (γ₁[ii] - p.γ̃[ii]).^2
        reg += (γ₂[ii] - p.γ̃[ii]).^2
    end
    reg *= p.λ/2

    y_gen = rand((-1,1), p.N) .* x_gen*β .+ randn(p.N)
    return E_psi(β, γ₁, γ₂, data, p) - E_psi(β, γ₁, γ₂, (x=x_gen, y=y_gen), p) - reg
    # return E_psi(β, γ₁, γ₂, data, p) - E_psi(β, γ₁, γ₂, (x=x_gen, y=y_gen), p) - p.λ/2 * (sum(abs2, γ₁ - p.γ̃) + sum(abs2, γ₁ - p.γ̃))
end


"""
Get compiled gradient of f wrt vars (β, γ₁, γ₂)
"""
function get_grad_function_rd(data, p)
    results = (
        Vector{Float64}(undef, p.d),
        Vector{Float64}(undef, p.d),
        Vector{Float64}(undef, p.d),
    )
    f_tape = GradientTape((β, γ₁, γ₂) -> f(β, γ₁, γ₂, data, p), results)
    compiled_f_tape = compile(f_tape)

    ∇f!(results, inputs) = gradient!(results, compiled_f_tape, inputs)
    return ∇f!
end


"""
Take GDA step for variables vars using ∇f! and stores in results
"""
function _gda_step(∇f!, vars, results, p)
    # Compute gradient
    ∇f!(results, vars)

    # Update variables
    @. vars[1] = vars[1] - p.α_min*results[1]
    @. vars[2] = vars[2] + p.α_max*results[2]
    @. vars[3] = vars[3] + p.α_max*results[3]
end
