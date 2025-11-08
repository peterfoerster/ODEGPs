function posterior_variance(k::MaternCovariance, P::AbstractMatrix{<:Real}, Ph::VectorOrMatrix; K_xxh::MatrixOrNothing=nothing, C::CholeskyOrNothing=nothing, autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # arguments are checked downstream
    isconsistent(k)

    if isnothing(K_xxh)
        K_xxh = covariance_matrix(k, P, Ph, autodiff=autodiff, symmetric=false, posterior=true, parallel=parallel)
    end

    if isnothing(C)
        K = covariance_matrix(k, P, P, symmetric=symmetric, parallel=parallel)
        C = LA.cholesky(K, LA.RowMaximum(), tol=1e-16)
    end

    # need every column once
    KiK_xxh = C \ K_xxh

    # allocate output
    kh = zeros(eltype(Ph), size(Ph, 2))

    if parallel
        Threads.@threads for i in range(1, length(kh))
            # additive Gaussian noise, k.k(p, p) = k.sigma[1]^2
            k_xh = k.sigma[1]^2 * (1 + k.sigma[2]^2)

            # use columns due to transpose
            kh[i] = k_xh - K_xxh[:,i]' * KiK_xxh[:,i]
        end
    else
        for i in range(1, length(kh))
            k_xh  = k.sigma[1]^2 * (1 + k.sigma[2]^2)
            kh[i] = k_xh - K_xxh[:,i]' * KiK_xxh[:,i]
        end
    end

    # check if posterior variance less than zero
    i_kh = kh .< 0

    if true in i_kh
        # @warn "Computed negative variances. Setting these to zero."
        kh[i_kh] .= 0
    end

    return kh
end

function posterior(k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, Ph::VectorOrMatrix; K_xxh::MatrixOrNothing=nothing, C::CholeskyOrNothing=nothing, autodiff::Bool=false, symmetric::Bool=true, variance::Bool=false, parallel::Bool=false)
    # arguments are checked downstream
    if length(x) != size(P, 2)
        throw(DomainError([length(x) size(P, 2)], "only supports length(x) = size(P, 2)."))
    end

    if isnothing(K_xxh)
        K_xxh = covariance_matrix(k, P, Ph, autodiff=autodiff, symmetric=false, posterior=true, parallel=parallel)
    end

    if isnothing(C)
        K = covariance_matrix(k, P, P, symmetric=symmetric, parallel=parallel)
        C = LA.cholesky(K, LA.RowMaximum(), tol=1e-16)
    end

    if variance
        kh = posterior_variance(k, P, Ph, K_xxh=K_xxh, C=C, parallel=parallel)
    end

    # assume zero mean
    w  = C \ x
    xh = K_xxh' * w

    if variance
        return (xh=xh, kh=kh)
    end

    return xh
end

function posteriorsquare_integral(k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, Ph::VectorOrMatrix, lb::Real; Ksi::Array3OrNothing=nothing, C::CholeskyOrNothing=nothing, symmetric::Bool=false, parallel::Bool=false)
    # arguments are checked downstream
    if length(x) != size(P, 2)
        throw(DomainError([length(x) size(P, 2)], "only supports length(x) = size(P, 2)."))
    end

    if isnothing(Ksi)
        Ksi = covariance_matrix_integrals(k, P, Ph, lb, symmetric=symmetric, parallel=parallel)
    end

    if isnothing(C)
        K = covariance_matrix(k, P, P, symmetric=symmetric, parallel=parallel)

        if LA.cond(K, Inf) > 1e10
            @warn "Condition number larger than 1e10. Halving length scales."
            K = covariance_matrix(k, P, P, theta=k.theta/2, symmetric=symmetric, parallel=parallel)
        end

        C = LA.cholesky(K, LA.RowMaximum(), tol=1e-16)
    end

    w    = C \ x
    xhsi = zeros(size(Ph, 2))

    for m in range(1, size(Ph, 2))
        for j in range(1, size(P, 2))
            for i in range(1, size(P,2))
                xhsi[m] = xhsi[m] + w[i] * w[j] * Ksi[i,j,m]
            end
        end
    end

    return xhsi
end

function variance_gradient!(J_kh::AbstractVector{<:Real}, k::MaternCovariance, P::AbstractMatrix{<:Real}, ph::AbstractVector{<:Real}; C::CholeskyOrNothing=nothing, autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # arguments are checked downstream
    if length(J_kh) != sum(k.n)
        throw(DomainError([length(J_kh) sum(k.n)], "only supports length(J_kh) = sum(k.n)."))
    end

    if autodiff
        # C does not depend on ph, implicitly ensure everything is scalar-valued via []
        J_kh[:] = FD.gradient(ph -> posterior_variance(k, P, ph, C=C, autodiff=autodiff, symmetric=symmetric, parallel=parallel)[], ph)
    else
        k_xxh = covariance_matrix(k, P, ph, symmetric=false, posterior=true, parallel=parallel)
        K_pp  = covariance_matrix_derivatives(k, P, ph, posterior=true, parallel=parallel)

        if isnothing(C)
            K = covariance_matrix(k, P, P, symmetric=symmetric, parallel=parallel)
            C = LA.cholesky(K, LA.RowMaximum(), tol=1e-16)
        end

        KiK_pp  = C \ K_pp[:,:,1]'
        J_kh[:] = -2 * (KiK_pp' * k_xxh)
    end

    return nothing
end

function maximize_variance(k::MaternCovariance, P::AbstractMatrix{<:Real}, ph::AbstractVector{<:Real}, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; C::CholeskyOrNothing=nothing, optimizer::String="LBFGS", autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # arguments are checked downstream
    if true in (lb .>= ub)
        throw(DomainError([lb ub], "only supports lb < ub."))
    end

    if length(lb) != sum(k.n)
        throw(DomainError([length(lb) sum(k.n)], "only supports length(lb) = sum(k.n)."))
    end

    if ~(optimizer in ["LBFGS"])
        throw(DomainError(optimizer, """only supports optimizer in ["LBFGS"]."""))
    end

    if isnothing(C)
        K = covariance_matrix(k, P, P, symmetric=symmetric, parallel=parallel)
        C = LA.cholesky(K, LA.RowMaximum(), tol=1e-16)
    end

    if autodiff
        # C does not depend on u, implicitly ensure everything is scalar-valued via []
        kh = (u, p) -> posterior_variance(k, P, u, C=C, autodiff=autodiff, parallel=parallel)[]

        # initial guess
        u = ph
        F = Optimization.OptimizationFunction(kh, Optimization.AutoForwardDiff())
    else
        kh    = (u, p) -> posterior_variance(k, P, u, C=C, parallel=parallel)
        J_kh! = (J_kh, u, p) -> variance_gradient!(J_kh, k, P, u, C=C, parallel=parallel)

        u = ph
        F = Optimization.OptimizationFunction(kh, grad=J_kh!)
    end

    O = Optimization.OptimizationProblem(F, u, lb=lb, ub=ub, sense=Optimization.MaxSense)

    if optimizer == "LBFGS"
        S = Optimization.solve(O, OOJL.LBFGS(linesearch=OOJL.Optim.BackTracking()), maxiters=1000, f_reltol=1e-4, x_abstol=1e-16, g_tol=1e-8)
    end

    if SciMLBase.successful_retcode(S.retcode)
        return S.u
    else
        throw(ErrorException("Variance maximization failed."))
    end
end

function posterior_gradient(k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, ph::AbstractVector{<:Real}; C::CholeskyOrNothing=nothing, autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # arguments are checked downstream
    if length(x) != size(P, 2)
        throw(DomainError([length(x) size(P, 2)], "only supports length(x) = size(P, 2)."))
    end

    # allocate output
    xh_ph = zeros(eltype(ph), sum(k.n))

    if autodiff
        # C does not depend on ph, implicitly ensure everything is scalar-valued via []
        J_xh     = ph -> FD.gradient(ph -> posterior(k, P, x, ph, C=C, autodiff=autodiff, symmetric=symmetric, parallel=parallel)[], ph)
        xh_ph[:] = J_xh(ph)
    else
        K_pp = covariance_matrix_derivatives(k, P, ph, posterior=true, parallel=parallel)

        if isnothing(C)
            K = covariance_matrix(k, P, P, symmetric=symmetric, parallel=parallel)
            C = LA.cholesky(K, LA.RowMaximum(), tol=1e-16)
        end

        w        = C \ x
        xh_ph[:] = K_pp[:,:,1] * w
    end

    return xh_ph
end

function posterior_variancegradientnorm(k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, ph::AbstractVector{<:Real}; C::CholeskyOrNothing=nothing, autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # implicitly ensure everything is scalar-valued via []
    kh    = posterior_variance(k, P, ph, C=C, autodiff=autodiff, symmetric=symmetric, parallel=parallel)[]
    xh_ph = posterior_gradient(k, P, x, ph, C=C, autodiff=autodiff, symmetric=symmetric, parallel=parallel)
    vgn   = kh * sum(xh_ph.^2)

    return vgn
end

function variancegradientnorm_gradient!(J_vgn::AbstractVector{<:Real}, k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, ph::AbstractVector{<:Real}; C::CholeskyOrNothing=nothing, autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # arguments are checked downstream
    if length(J_vgn) != sum(k.n)
        throw(DomainError([length(J_vgn) sum(k.n)], "only supports length(J_vgn) = sum(k.n)."))
    end

    if autodiff
        # C does not depend on ph
        J_vgn[:] = FD.gradient(ph -> posterior_variancegradientnorm(k, P, x, ph, C=C, autodiff=autodiff, symmetric=symmetric, parallel=parallel), ph)
    else
        if isnothing(C)
            K = covariance_matrix(k, P, P, symmetric=symmetric, parallel=parallel)
            C = LA.cholesky(K, LA.RowMaximum(), tol=1e-16)
        end

        # implicitly ensure everything is scalar-valued via [] (K_xxh)
        kh = posterior_variance(k, P, ph, C=C, parallel=parallel)[]

        # (k_xxh, K_pp)
        J_kh = zeros(sum(k.n))
        variance_gradient!(J_kh, k, P, ph, C=C, parallel=parallel)

        # (K_pp)
        xh_ph = posterior_gradient(k, P, x, ph, C=C, parallel=parallel)
        gn    = sum(xh_ph.^2)

        K_pps   = covariance_matrix_derivatives(k, P, ph, hessian=true, parallel=parallel)
        w       = C \ x
        J_xh_ph = zeros(sum(k.n), sum(k.n))

        for j in range(1, sum(k.n))
            for i in range(1, sum(k.n))
                J_xh_ph[i,j] = K_pps[i,j,:,1]' * w
            end
        end

        J_gn = 2 * J_xh_ph * xh_ph

        # product rule
        J_vgn[:] = J_kh * gn + J_gn * kh
    end

    return nothing
end

function maximize_variancegradientnorm(k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, ph::AbstractVector{<:Real}, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; C::CholeskyOrNothing=nothing, optimizer::String="LBFGS", autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # arguments are checked downstream
    if true in (lb .>= ub)
        throw(DomainError([lb ub], "only supports lb < ub."))
    end

    if length(lb) != sum(k.n)
        throw(DomainError([length(lb) sum(k.n)], "only supports length(lb) = sum(k.n)."))
    end

    if ~(optimizer in ["LBFGS"])
        throw(DomainError(optimizer, """only supports optimizer in ["LBFGS"]."""))
    end

    if isnothing(C)
        K = covariance_matrix(k, P, P, symmetric=symmetric, parallel=parallel)
        C = LA.cholesky(K, LA.RowMaximum(), tol=1e-16)
    end

    if autodiff
        # C does not depend on u
        vgn = (u, p) -> posterior_variancegradientnorm(k, P, x, u, C=C, autodiff=autodiff, symmetric=symmetric, parallel=parallel)

        # initial guess
        u = ph
        F = Optimization.OptimizationFunction(vgn, Optimization.AutoForwardDiff())
    else
        vgn    = (u, p) -> posterior_variancegradientnorm(k, P, x, u, C=C, symmetric=symmetric, parallel=parallel)
        J_vgn! = (J_vgn, u, p) -> variancegradientnorm_gradient!(J_vgn, k, P, x, u, C=C, parallel=parallel)

        u = ph
        F = Optimization.OptimizationFunction(vgn, grad=J_vgn!)
    end

    O = Optimization.OptimizationProblem(F, u, lb=lb, ub=ub, sense=Optimization.MaxSense)

    if optimizer == "LBFGS"
        S = Optimization.solve(O, OOJL.LBFGS(linesearch=OOJL.Optim.BackTracking()), maxiters=1000, f_reltol=1e-4, x_abstol=1e-16, g_tol=1e-8)
    end

    if SciMLBase.successful_retcode(S.retcode)
        return S.u
    else
        throw(ErrorException("Variance gradient norm maximization failed."))
    end
end

function update_cholesky(C::CholeskyOrPivoted, kp::AbstractVector{<:Real})
    n  = length(kp)
    Up = zeros(n, n)
    U  = C.U

    if C isa LA.CholeskyPivoted
        p               = C.p
        Up[1:n-1,1:n-1] = U
        Up[1:n-1,n]     = U' \ kp[1:n-1][p]

        try
            Up[n,n] = sqrt(kp[n] - Up[1:n-1,n]' * Up[1:n-1,n])
        catch exception
            throw(exception)
        end

        append!(p, n)
        Cp = LA.CholeskyPivoted(LA.UpperTriangular(Up), C.uplo, p, n, C.tol, C.info)
    else
        Up[1:n-1,1:n-1] = U
        Up[1:n-1,n]     = U' \ kp[1:n-1]

        try
            Up[n,n] = sqrt(kp[n] - Up[1:n-1,n]' * Up[1:n-1,n])
        catch exception
            throw(exception)
        end

        Cp = LA.Cholesky(LA.UpperTriangular(Up))
    end

    return Cp
end
