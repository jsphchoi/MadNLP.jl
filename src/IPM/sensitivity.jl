# Parametric sensitivity ds/dp = -M^-1*N_p, M is KKT system and N_p = [d^2L/dxdp; dc/dp; 0; 0]
"""
    backsolve_kkt!(solver::AbstractMadNLPSolver, px::AbstractMatrix, py::AbstractMatrix)

Solves `M d = p` where `p = (px, py, 0, 0)` and `M` is last KKT matrix factorized by [`solve!`](@ref).
Returns `d = (; dx, dy, dzl, dzu)`.

- `px`: variable block, `nvar × k`
- `py`: constraint block, `ncon × k`
"""
function backsolve_kkt!(solver::AbstractMadNLPSolver{T}, px::AbstractMatrix, py::AbstractMatrix) where T
    get_status(solver) in (SOLVE_SUCCEEDED, SOLVED_TO_ACCEPTABLE_LEVEL) ||
        error("backsolve_kkt! requires a solver on which solve! has converged")
    nlp, cb, kkt = get_nlp(solver), get_cb(solver), get_kkt(solver)
    nvar, ncon, k = get_nvar(nlp), get_ncon(nlp), size(px, 2)
    size(px, 1) == nvar || throw(ArgumentError("px must be nvar × k"))
    size(py) == (ncon, k) || throw(ArgumentError("py must be ncon × k"))

    p, d, w, zbuf = get_p(solver), get_d(solver), get__w4(solver), primal(get__w1(solver))
    nx = length(variable(get_x(solver)))
    ifree, ufree = _free_indices(cb)
    x0 = get_x0(nlp)
    dx, dy, dzl, dzu = (fill!(similar(x0, n, k), zero(T)) for n in (nvar, ncon, nvar, nvar))
    for j in 1:k
        fill!(full(p), zero(T))
        view(primal(p), ifree) .= view(px, ufree, j) .* (cb.obj_sign * cb.obj_scale[])
        dual(p) .= view(py, :, j) .* cb.con_scale
        solve_refine_wrapper!(d, solver, p, w) || error("KKT back-solve failed")
        view(dx, ufree, j) .= view(primal(d), ifree)
        unpack_y!(view(dy, :, j), cb, dual(d))
        for (dz, ind, dzd) in ((dzl, kkt.ind_lb, dual_lb(d)), (dzu, kkt.ind_ub, dual_ub(d)))
            fill!(zbuf, zero(T))
            zbuf[ind] .= dzd
            unpack_z!(view(dz, :, j), cb, view(zbuf, 1:nx))
        end
    end
    return (; dx, dy, dzl, dzu)
end

"""
    sensitivity(solver::AbstractMadNLPSolver[, θ])
    sensitivity(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix)

Returns the first-order parametric sensitivity `(; dx, dy, dzl, dzu)/dθ` at the primal-dual solution.

- `Hxθ`: `∂²L/∂x∂θ` at the solution, `nvar × nθ`
- `Jθ`: `∂c/∂θ` at the solution, `ncon × nθ`
"""
sensitivity(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix) =
    backsolve_kkt!(solver, .-Hxθ, .-Jθ)
function sensitivity(solver::AbstractMadNLPSolver, θ...)
    nlp, stats = get_nlp(solver), update!(MadNLPExecutionStats(solver), solver)
    applicable(nlp, θ..., stats.solution, stats.multipliers) ||
        throw(ArgumentError("the model does not provide nlp(x, y) or nlp(θ, x, y), Hxθ and Jθ required"))
    return sensitivity(solver, nlp(θ..., stats.solution, stats.multipliers)...)
end

"""
    sensitivity_result(solver::AbstractMadNLPSolver, θ, θnew; restore_parameter = true, recompute_residuals = false)
    sensitivity_result(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix, dθ::AbstractVector; recompute_residuals = false)

Returns the [`MadNLPExecutionStats`](@ref) at `θnew` from the first-order parametric sensitivity at `θ`.

- `θ`: parameter block, accessed as `nlp[θ]`
- `θnew`: new values of `θ`
- `dθ`: `θnew - θ`, with the model already at `θnew`
- `restore_parameter`: restore `nlp[θ]` after the call
- `recompute_residuals`: recompute `primal_feas` and `dual_feas` at the new solution
"""
function sensitivity_result(solver::AbstractMadNLPSolver, θ, θnew; restore_parameter = true, recompute_residuals = false)
    nlp = get_nlp(solver)
    θ0 = copy(nlp[θ])
    dθ = copyto!(similar(θ0), θnew) .- θ0
    s = sensitivity(solver, θ)
    nlp[θ] = θnew
    try
        return _sensitivity_result(solver, s, dθ, recompute_residuals)
    finally
        restore_parameter && (nlp[θ] = θ0)
    end
end
sensitivity_result(solver::AbstractMadNLPSolver, Hxθ::AbstractMatrix, Jθ::AbstractMatrix, dθ::AbstractVector; recompute_residuals = false) =
    _sensitivity_result(solver, sensitivity(solver, Hxθ, Jθ), dθ, recompute_residuals)

function _sensitivity_result(solver, s, dθ, recompute_residuals)
    nlp, stats = get_nlp(solver), update!(MadNLPExecutionStats(solver), solver)
    stats.solution .+= s.dx * dθ
    stats.multipliers .+= s.dy * dθ
    stats.multipliers_L .+= s.dzl * dθ
    stats.multipliers_U .+= s.dzu * dθ
    stats.objective = NLPModels.obj(nlp, stats.solution)
    get_ncon(nlp) > 0 && NLPModels.cons!(nlp, stats.solution, stats.constraints)
    recompute_residuals && _residuals!(stats, nlp)
    return stats
end

_violation(v, l, u) = max(maximum(l .- v; init = zero(eltype(v))), maximum(v .- u; init = zero(eltype(v))))
function _residuals!(stats::MadNLPExecutionStats, nlp)
    x, lvar, uvar = stats.solution, NLPModels.get_lvar(nlp), NLPModels.get_uvar(nlp)
    stats.primal_feas = max(
        _violation(x, lvar, uvar),
        _violation(stats.constraints, NLPModels.get_lcon(nlp), NLPModels.get_ucon(nlp)),
    )
    r = similar(x)
    NLPModels.grad!(nlp, x, r)
    r .-= stats.multipliers_L
    r .+= stats.multipliers_U
    if get_ncon(nlp) > 0
        jtv = similar(x)
        NLPModels.jtprod!(nlp, x, stats.multipliers, jtv)
        r .+= jtv
    end
    stats.dual_feas = maximum(abs, r .* (lvar .!= uvar); init = zero(eltype(r)))
    return stats
end

# Positions of the free variables in `cb` and in `nlp`.
_free_indices(cb::AbstractCallback) = (1:get_nvar(cb.nlp), 1:get_nvar(cb.nlp))
_free_indices(cb::SparseCallback{T, VT, VI, NLP, FH}) where {T, VT, VI, NLP, FH<:MakeParameter} =
    (1:length(cb.fixed_handler.free), cb.fixed_handler.free)
_free_indices(cb::DenseCallback{T, VT, VI, NLP, FH}) where {T, VT, VI, NLP, FH<:MakeParameter} =
    (cb.fixed_handler.free, cb.fixed_handler.free)
