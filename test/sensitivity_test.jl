using Test
using NLPModels
using LinearAlgebra
using MadNLPTests
using Random

# Convex QP with constant Hessian P and Jacobian A, so the last KKT matrix has W = P and J = A.
function _solve_qp(; n=10, m=5, fixed_variables=Int[], equality_cons=[1, 3], kwargs...)
    nlp = MadNLPTests.DenseDummyQP(zeros(n); m=m, fixed_variables=fixed_variables, equality_cons=equality_cons)
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR, tol=1e-8, kwargs...)
    stats = MadNLP.solve!(solver)
    return nlp, solver, stats
end

# Residuals of M d = p with the P and A of the QP: P dx + Aᵀ dy - dzl + dzu - px and A dx - py.
function _kkt_residuals(nlp, d, px, py)
    return nlp.P * d.dx .+ nlp.A' * d.dy .- d.dzl .+ d.dzu .- px, nlp.A * d.dx .- py
end

# Model implementing the parameter protocol of `sensitivity` around another model.
struct ParametricModel{T, M} <: NLPModels.AbstractNLPModel{T, Vector{T}}
    meta::NLPModels.NLPModelMeta{T, Vector{T}}
    counters::NLPModels.Counters
    inner::M
    θ::Vector{T}
    Hxθ::Matrix{T}
    Jθ::Matrix{T}
end
ParametricModel(inner, θ, Hxθ, Jθ) = ParametricModel(inner.meta, inner.counters, inner, θ, Hxθ, Jθ)
NLPModels.obj(nlp::ParametricModel, x::AbstractVector) = NLPModels.obj(nlp.inner, x)
NLPModels.grad!(nlp::ParametricModel, x::AbstractVector, g::AbstractVector) = NLPModels.grad!(nlp.inner, x, g)
NLPModels.cons!(nlp::ParametricModel, x::AbstractVector, c::AbstractVector) = NLPModels.cons!(nlp.inner, x, c)
NLPModels.jtprod!(nlp::ParametricModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector) = NLPModels.jtprod!(nlp.inner, x, v, Jtv)
NLPModels.jac_structure!(nlp::ParametricModel, rows::AbstractVector, cols::AbstractVector) = NLPModels.jac_structure!(nlp.inner, rows, cols)
NLPModels.jac_coord!(nlp::ParametricModel, x::AbstractVector, vals::AbstractVector) = NLPModels.jac_coord!(nlp.inner, x, vals)
NLPModels.hess_structure!(nlp::ParametricModel, rows::AbstractVector, cols::AbstractVector) = NLPModels.hess_structure!(nlp.inner, rows, cols)
NLPModels.hess_coord!(nlp::ParametricModel, x::AbstractVector, y::AbstractVector, vals::AbstractVector; obj_weight=1.0) =
    NLPModels.hess_coord!(nlp.inner, x, y, vals; obj_weight=obj_weight)
struct Θ end
(nlp::ParametricModel)(x, y) = (nlp.Hxθ, nlp.Jθ)
(nlp::ParametricModel)(::Θ, x, y) = nlp(x, y)
Base.getindex(nlp::ParametricModel, ::Θ) = nlp.θ
Base.setindex!(nlp::ParametricModel, v, ::Θ) = (nlp.θ .= v; nlp)

sparse_options = Dict{Symbol, Any}(
    :callback=>MadNLP.SparseCallback,
    :kkt_system=>MadNLP.SparseKKTSystem,
)
dense_options = Dict{Symbol, Any}(
    :callback=>MadNLP.DenseCallback,
    :kkt_system=>MadNLP.DenseKKTSystem,
    :linear_solver=>MadNLP.LapackCPUSolver,
)

@testset "Sensitivity: backsolve_kkt!" begin
    n, m, eq = 10, 5, [1, 3]
    Random.seed!(1)
    px, py = randn(n, 2), randn(m, 2)

    @testset "$name" for (name, options, fixed) in [
        ("sparse", sparse_options, Int[]),
        ("sparse + scaling", merge(sparse_options, Dict{Symbol, Any}(:nlp_scaling_max_gradient=>0.1)), Int[]),
        ("sparse + fixed variables", sparse_options, [9, 10]),
        ("dense + fixed variables", dense_options, [9, 10]),
    ]
        nlp, solver, stats = _solve_qp(; n=n, m=m, fixed_variables=fixed, equality_cons=eq, options...)
        d = MadNLP.backsolve_kkt!(solver, px, py)
        free = setdiff(1:n, fixed)
        resx, resy = _kkt_residuals(nlp, d, px, py)
        @test norm(resx[free, :], Inf) <= 1e-6
        @test norm(resy[eq, :], Inf) <= 1e-6
        @test iszero(d.dx[fixed, :]) && iszero(d.dzl[fixed, :]) && iszero(d.dzu[fixed, :])
    end
end

@testset "Sensitivity: sensitivity" begin
    n, m, k = 10, 5, 3
    Random.seed!(1)
    Hxθ, Jθ = randn(n, k), randn(m, k)
    nlp = ParametricModel(MadNLPTests.DenseDummyQP(zeros(n); m=m, equality_cons=[1, 3]), [0.5, -1.0, 2.0], Hxθ, Jθ)
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR, tol=1e-8, sparse_options...)
    MadNLP.solve!(solver)
    s = MadNLP.sensitivity(solver, Hxθ, Jθ)
    @test s == MadNLP.backsolve_kkt!(solver, -Hxθ, -Jθ)
    @test MadNLP.sensitivity(solver) == s
    @test MadNLP.sensitivity(solver, Θ()) == s
end

@testset "Sensitivity: sensitivity_result" begin
    n, m, k = 10, 5, 3
    Random.seed!(1)
    Hxθ, Jθ = randn(n, k), randn(m, k)
    θ0, θnew = [0.5, -1.0, 2.0], [0.6, -1.2, 2.3]
    dθ = θnew .- θ0
    nlp = ParametricModel(MadNLPTests.DenseDummyQP(zeros(n); m=m, equality_cons=[1, 3]), copy(θ0), Hxθ, Jθ)
    solver = MadNLPSolver(nlp; print_level=MadNLP.ERROR, tol=1e-8, sparse_options...)
    stats = MadNLP.solve!(solver)
    s = MadNLP.sensitivity(solver, Hxθ, Jθ)

    updated = MadNLP.sensitivity_result(solver, Hxθ, Jθ, dθ)
    @test updated.solution == stats.solution .+ s.dx * dθ
    @test updated.multipliers == stats.multipliers .+ s.dy * dθ
    @test updated.multipliers_L == stats.multipliers_L .+ s.dzl * dθ
    @test updated.multipliers_U == stats.multipliers_U .+ s.dzu * dθ
    @test updated.objective == NLPModels.obj(nlp, updated.solution)
    @test updated.constraints == NLPModels.cons(nlp, updated.solution)

    @test MadNLP.sensitivity_result(solver, Θ(), θnew).solution == updated.solution
    @test nlp[Θ()] == θ0
    MadNLP.sensitivity_result(solver, Θ(), θnew; restore_parameter=false)
    @test nlp[Θ()] == θnew

    exact = MadNLP.sensitivity_result(solver, Hxθ, Jθ, zeros(k); recompute_residuals=true)
    @test exact.primal_feas <= 1e-6
    @test exact.dual_feas <= 1e-6
end
