module ODEGaussianProcesses

export MaternCovariance, *
export logmarginallikelihood, maximize_logmarginallikelihood, maximize_logmarginallikelihood!
export posterior_variance, posterior, posteriorsquare_integral, maximize_variance, posterior_gradient, posterior_variancegradientnorm, maximize_variancegradientnorm
export downsample, S_tdo, S_vdpo, S_brus
export normalize!, unnormalize!, standardize!, unstandardize!, Lnorm, C2HermiteSpline, QuinticMonotonicSpline
export reparameterize, hausdorff_distance
export conventional_doe!, reparameterized_doe!

# all MIT license
using Infiltrator

import Base: *
import BracketingNonlinearSolve as BNS
import DataInterpolations as DI
import DelimitedFiles as DF
import ForwardDiff as FD
import Integrals
import LinearAlgebra as LA
import Optimization
# (we do not use the GNU LGPL code)
# import OptimizationNLopt.NLopt as ONL
import OptimizationOptimJL as OOJL
import OrdinaryDiffEq as ODE
import Printf
import SciMLBase
import SpecialFunctions as SF
import Statistics

VectorOrMatrix    = Union{AbstractVector{<:Real}, AbstractMatrix{<:Real}}
CholeskyOrPivoted = Union{LA.Cholesky, LA.CholeskyPivoted}
VectorOrNothing   = Union{AbstractVector{<:Real}, Nothing}
MatrixOrNothing   = Union{AbstractMatrix{<:Real}, Nothing}
Array3OrNothing   = Union{AbstractArray{<:Real, 3}, Nothing}
CholeskyOrNothing = Union{LA.Cholesky, LA.CholeskyPivoted, Nothing}

include("Covariances/MaternCovariance.jl")
include("Covariances/Covariance.jl")
include("LogMarginalLikelihood.jl")
include("Posterior.jl")
include("ExampleODEs.jl")
include("Reparameterization.jl")
include("DOE.jl")

function normalize!(P::AbstractMatrix{<:Real})
    m = zeros(size(P, 1))
    n = zeros(size(P, 1))

    # normalize each row individually
    for i in range(1, size(P, 1))
        m[i]   = minimum(P[i,:])
        P[i,:] = P[i,:] .- m[i]
        n[i]   = LA.norm(P[i,:], Inf)
        P[i,:] = P[i,:] / n[i]
    end

    return (m=m, n=n)
end

function normalize!(Ph::VectorOrMatrix, m::AbstractVector{<:Real}, n::AbstractVector{<:Real})
    # normalize each row individually
    for i in range(1, size(Ph, 1))
        Ph[i,:] = (Ph[i,:] .- m[i]) / n[i]
    end

    return nothing
end

function unnormalize!(P::VectorOrMatrix, m::AbstractVector{<:Real}, n::AbstractVector{<:Real})
    # unnormalize each row individually
    for i in range(1, size(P, 1))
        P[i,:] = n[i] * P[i,:] .+ m[i]
    end

    return nothing
end

function standardize!(x::AbstractVector{<:Real})
    mu    = Statistics.mean(x)
    sigma = Statistics.std(x)
    x[:]  = (x .- mu) / sigma

    return (mu=mu, sigma=sigma)
end

function unstandardize!(x::AbstractVector{<:Real}, mu::Real, sigma::Real)
    x[:] = sigma * x .+ mu

    return nothing
end

function Lnorm(t::AbstractVector{<:Real}, x::AbstractVector{<:Real}; p::Int=2)
    if length(x) != length(t)
        throw(DomainError([length(x) length(t)], "only supports length(x) = length(t)."))
    end

    if p < 1
        throw(DomainError(p, "only supports p >= 1."))
    end

    # trapezoidal rule
    I = 0

    for i in range(2, length(t))
        I = I + (t[i] - t[i-1]) * (abs(x[i-1])^p + abs(x[i])^p) / 2
    end

    return I^(1/p)
end

function C2HermiteSpline(t::AbstractVector{<:Real}, x::AbstractVector{<:Real}, f::AbstractVector{<:Real})
    if length(x) != length(t)
        throw(DomainError([length(x) length(t)], "only supports length(x) = length(t)."))
    end

    if length(f) != length(t)
        throw(DomainError([length(f) length(t)], "only supports length(f) = length(t)."))
    end

    # first order finite difference estimate
    fp = [diff(f) ./ diff(t); (f[end] - f[end-1]) / (t[end] - t[end-1])]

    return DI.QuinticHermiteSpline(fp, f, x, t)
end

function QuinticMonotonicSpline(t::AbstractVector{<:Real}, x::AbstractVector{<:Real}, f::AbstractVector{<:Real})
    if length(x) != length(t)
        throw(DomainError([length(x) length(t)], "only supports length(x) = length(t)."))
    end

    if length(f) != length(t)
        throw(DomainError([length(f) length(t)], "only supports length(f) = length(t)."))
    end

    # first order finite difference estimate
    dt = diff(t)
    fp = [diff(f) ./ dt; (f[end] - f[end-1]) / (t[end] - t[end-1])]

    # check monotonicity conditions from https://www.sciencedirect.com/science/article/pii/0377042794901848
    decrease = true

    for i in range(1, length(t))
        if i < length(t)
            fc = -4*f[i] / dt[i]

            if fp[i] < fc
                # @warn "First condition violated."
                fp[i] = fc

                # check second condition
                if i > 1
                    sc = 4*f[i] / dt[i-1]

                    if fp[i] > sc
                        @warn "Second condition violated."
                        throw(ErrorException("fp[i] violates either first or second condition."))
                    end
                end

                # check third condition
                tc = 60 * (x[i+1] - x[i]) - 24*dt[i] * (f[i+1] + f[i]) + 3*dt[i]^2 * (fp[i+1] - fp[i])

                if 0 > tc
                    # @warn "Third condition violated."

                    # set fp[i+1] to third condition
                    fp[i+1]  = (-60 * (x[i+1] - x[i]) + 24*dt[i] * (f[i+1] + f[i]) + 3*dt[i]^2 * fp[i]) / (3*dt[i]^2)
                    decrease = false
                end

                continue
            end
        end

        if i > 1
            sc = 4*f[i] / dt[i-1]

            if fp[i] > sc
                # @warn "Second condition violated."

                if decrease
                    fp[i] = sc

                    if i < length(t)
                        # check first condition
                        fc = -4*f[i] / dt[i]

                        if fp[i] < fc
                            @warn "First condition violated."
                            throw(ErrorException("fp[i] violates either first or second condition."))
                        end

                        # check third condition
                        tc = 60 * (x[i+1] - x[i]) - 24*dt[i] * (f[i+1] + f[i]) + 3*dt[i]^2 * (fp[i+1] - fp[i])

                        if 0 > tc
                            # @warn "Third condition violated."

                            # set fp[i] to minimum of first and third condition
                            tc    = (60 * (x[i+1] - x[i]) - 24*dt[i] * (f[i+1] + f[i]) + 3*dt[i]^2 * fp[i+1]) / (3*dt[i]^2)
                            fp[i] = max(tc, fc)
                        end

                        continue
                    end
                else
                    throw(ErrorException("fp[i] violates second condition and decrease = false."))
                end
            end
        end

        if i < length(t)
            fc = -4*f[i] / dt[i]
            tc = 60 * (x[i+1] - x[i]) - 24*dt[i] * (f[i+1] + f[i]) + 3*dt[i]^2 * (fp[i+1] - fp[i])

            if 0 > tc
                # @warn "Third condition violated."

                if decrease
                    # set fp[i] to minimum of first and third condition
                    tc    = (60 * (x[i+1] - x[i]) - 24*dt[i] * (f[i+1] + f[i]) + 3*dt[i]^2 * fp[i+1]) / (3*dt[i]^2)
                    fp[i] = max(tc, fc)
                else
                    # set fp[i+1] to third condition
                    fp[i+1]  = (-60 * (x[i+1] - x[i]) + 24*dt[i] * (f[i+1] + f[i]) + 3*dt[i]^2 * fp[i]) / (3*dt[i]^2)
                    decrease = false
                end
            end
        end
    end

    return DI.QuinticHermiteSpline(fp, f, x, t)
end
end
