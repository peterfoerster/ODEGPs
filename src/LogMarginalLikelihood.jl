function trprod(K::AbstractMatrix{<:Real}, Kp::AbstractMatrix{<:Real})
    k = 0

    for j in range(1, size(Kp, 2))
        # only compute diagonal
        k = k + K[j,:]' * Kp[:,j]
    end

    return k
end

function logmarginallikelihood(k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # arguments are checked downstream
    if length(x) != size(P, 2)
        throw(DomainError([length(x) size(P, 2)], "only supports length(x) = size(P, 2)."))
    end

    K = covariance_matrix(k, P, P, sigma=sigma, theta=theta, autodiff=autodiff, symmetric=symmetric, parallel=parallel)

    if autodiff
        C = LA.cholesky(K, check=false)
    else
        C = LA.cholesky(K, LA.RowMaximum(), tol=1e-16, check=false)
    end

    if ~LA.issuccess(C)
        # @warn "Cholesky factorization failed. Returning NaN."
        return NaN
    end

    # assume zero mean
    w = C \ x
    L = -1/2 * ( (x' * w) + LA.logdet(C) + length(x) * log(2*pi) )

    return L
end

function logmarginallikelihood_gradient!(J_L::AbstractVector{<:Real}, k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # arguments are checked downstream
    if length(J_L) != 1+sum(k.n)
        throw(DomainError([length(J_L) 1+sum(k.n)], "only supports length(J_L) = 1+sum(k.n)."))
    end

    if length(x) != size(P, 2)
        throw(DomainError([length(x) size(P, 2)], "only supports length(x) = size(P, 2)."))
    end

    if autodiff
        J_L[:] = FD.gradient(btheta -> logmarginallikelihood(k, P, x, sigma=[btheta[1]; sigma[2]], theta=btheta[2:end], autodiff=autodiff, symmetric=symmetric, parallel=parallel), [sigma[1]; theta])
    else
        K = covariance_matrix(k, P, P, sigma=sigma, theta=theta, symmetric=symmetric, parallel=parallel)
        C = LA.cholesky(K, LA.RowMaximum(), tol=1e-16, check=false)

        if ~LA.issuccess(C)
            # @warn "Cholesky factorization failed. Returning NaN."
            return NaN
        end

        # assume zero mean
        Ki = inv(C)
        w  = Ki * x

        (K_sigma, K_theta) = covariance_matrix_derivatives(k, P, P, sigma=sigma, theta=theta, parallel=parallel)

        J_L[1] = 1/2 * ( (w' * (K_sigma * w)) - trprod(Ki, K_sigma) )

        for i in range(1, sum(k.n))
            J_L[i+1] = 1/2 * ( (w' * (K_theta[i,:,:] * w)) - trprod(Ki, K_theta[i,:,:]) )
        end
    end

    return nothing
end

function maximize_logmarginallikelihood(k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, optimizer::String="OOJL-LBFGS", autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # arguments are checked downstream
    if true in (lb .>= ub)
        throw(DomainError([lb ub], "only supports lb < ub."))
    end

    if length(lb) != 1+sum(k.n)
        throw(DomainError([length(lb) 1+sum(k.n)], "only supports length(lb) = 1+sum(k.n)."))
    end

    # derivative-free: ONL-BOBYQA
    # gradient-based: ONL-CCSAQ, ONL-SLSQP
    if ~(optimizer in ["OOJL-LBFGS", "ONL-CCSAQ", "ONL-SLSQP", "ONL-BOBYQA"])
        throw(DomainError(optimizer, """only supports optimizer in ["OOJL-LBFGS", "ONL-CCSAQ", "ONL-SLSQP", "ONL-BOBYQA"]."""))
    end

    if autodiff
        L = (u, p) -> logmarginallikelihood(k, P, x, sigma=[u[1]; sigma[2]], theta=u[2:end], autodiff=autodiff, symmetric=symmetric, parallel=parallel)

        # initial guess
        u = [sigma[1]; theta]
        F = Optimization.OptimizationFunction(L, Optimization.AutoForwardDiff())
    else
        L    = (u, p) -> logmarginallikelihood(k, P, x, sigma=[u[1]; sigma[2]], theta=u[2:end], autodiff=autodiff, symmetric=symmetric, parallel=parallel)
        J_L! = (J_L, u, p) -> logmarginallikelihood_gradient!(J_L, k, P, x, sigma=[u[1]; sigma[2]], theta=u[2:end], autodiff=autodiff, symmetric=symmetric, parallel=parallel)
        u    = [sigma[1]; theta]

        F = Optimization.OptimizationFunction(L, grad=J_L!)
    end

    O = Optimization.OptimizationProblem(F, u, lb=lb, ub=ub, sense=Optimization.MaxSense)

    # Optimization:
    # maxiters=1000
    # maxtime=?
    # abstol=1e-16
    # reltol=1e-4 -> f_reltol=1e-4

    # OptimJL:
    # x_tol=1e-16 -> x_abstol=1e-16
    # g_tol=1e-8
    # allow_f_increases=false

    # OOJL-LBFGS:
    # m=10
    # alphaguess=Optim.InitialStatic()
    # linesearch=Optim.HagerZhang()
    # linesearch=Optim.BackTracking()

    # NLopt:
    # xtol_rel=?
    # xtol_abs=?

    if optimizer == "OOJL-LBFGS"
        S = Optimization.solve(O, OOJL.LBFGS(linesearch=OOJL.Optim.BackTracking()), maxiters=1000, maxtime=200, f_reltol=1e-4, x_abstol=1e-16, g_tol=1e-8)
    # elseif optimizer == "ONL-CCSAQ"
    #     S = Optimization.solve(O, ONL.LD_CCSAQ(), maxiters=1000, maxtime=200, reltol=1e-4, xtol_abs=1e-16)
    # elseif optimizer == "ONL-SLSQP"
    #     S = Optimization.solve(O, ONL.LD_SLSQP(), maxiters=1000, maxtime=200, reltol=1e-4, xtol_abs=1e-16)
    # elseif optimizer == "ONL-BOBYQA"
    #     S = Optimization.solve(O, ONL.LN_BOBYQA(), maxiters=1000, maxtime=200, reltol=1e-4, xtol_abs=1e-16)
    end

    if SciMLBase.successful_retcode(S.retcode)
        return S.u
    else
        throw(ErrorException("Log marginal likelihood maximization failed."))
    end
end

function maximize_logmarginallikelihood!(k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, optimizer::String="OOJL-LBFGS", autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    btheta = maximize_logmarginallikelihood(k, P, x, lb, ub, sigma=sigma, theta=theta, optimizer=optimizer, autodiff=autodiff, symmetric=symmetric, parallel=parallel)

    k.sigma[1] = btheta[1]
    k.theta[:] = btheta[2:end]

    return nothing
end
