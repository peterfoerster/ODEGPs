function initial_design(lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; sparse::Bool=true)
    if sparse
        n  = length(lb)
        P  = zeros(n, 1+2*n)
        mu = Statistics.mean([lb ub], dims=2)
        P .= mu

        for i in range(1, n)
            P[i,2*i]   = lb[i]
            P[i,2*i+1] = ub[i]
        end
    else
        n   = length(lb)
        N   = 2^n - 1
        P   = zeros(n, N+1)
        i_P = [digits(m, base=2, pad=ndigits(N, base=2)) .+ 1 for m in range(0, N)]
        Pb  = [lb ub]

        for j in range(1, N+1)
            for i in range(1, n)
                # i_P[j] contains vector of indices
                P[i,j] = Pb[i,i_P[j][i]]
            end
        end
    end

    return P
end

function initial_data_conventional(S::Function, P_i::AbstractMatrix{<:Real})
    (t, x) = S(P_i[:,1])

    # boundary points plus extra point
    i = [1, round(Int, length(t)/2), length(t)]
    P = [t[i]'; repeat(P_i[:,1], 1, length(i))]
    x = x[i]

    for n in range(2, size(P_i, 2))
        (t, x_n) = S(P_i[:,n])

        i   = [1, round(Int, length(t)/2), length(t)]
        P_n = [t[i]'; repeat(P_i[:,n], 1, length(i))]
        P   = [P P_n]
        append!(x, x_n[i])
    end

    return (P=P, x=x)
end

function initial_data_reparameterized(S::Function, P_i::AbstractMatrix{<:Real}; derivative::Bool=false, nu::Int=2, mu::Real=Inf, integrator::String="Rosenbrock23", abstol::Real=1e-6, reltol::Real=1e-3, reltol_ext::Real=1e-3, arclength::Bool=true)
    if derivative
        (t, x, f)                     = S(P_i[:,1])
        (tauith, xth, tauitph, s_ext) = reparameterize(t, x, f, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
    else
        (t, x)                        = S(P_i[:,1])
        (tauith, xth, tauitph, s_ext) = reparameterize(t, x, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
    end

    # boundary points plus central point
    s = [0, 0.5, 1]

    P      = [s'; repeat(P_i[:,1], 1, length(s))]
    tauit  = tauith(s)
    xt     = xth(s)
    tauitp = tauitph.(s)
    N_ext  = length(s_ext)

    for n in range(2, size(P_i, 2))
        if derivative
            (t, x, f)                     = S(P_i[:,n])
            (tauith, xth, tauitph, s_ext) = reparameterize(t, x, f, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
        else
            (t, x)                        = S(P_i[:,n])
            (tauith, xth, tauitph, s_ext) = reparameterize(t, x, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
        end

        if length(s_ext) != N_ext
            @warn "Number of extremas changed."
        end

        P_n = [s'; repeat(P_i[:,n], 1, length(s))]
        P   = [P P_n]
        append!(tauit, tauith(s))
        append!(xt, xth(s))
        append!(tauitp, tauitph.(s))
    end

    return (P=P, tauit=tauit, xt=xt, tauitp=tauitp, N_ext=N_ext)
end

function test_data_conventional(S::Function, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; n_test_t::Int=100, n_test_p::Int=1+length(lb))
    P_e = zeros(1+length(lb), 0)
    x_e = zeros(0)

    for n in range(1, n_test_p)
        p_e    = lb + rand(length(lb)) .* (ub - lb)
        (t, x) = S(p_e)
        i_t    = unique(rand(1:length(t), n_test_t))
        P      = [t[i_t]'; repeat(p_e, 1, length(i_t))]
        P_e    = [P_e P]
        append!(x_e, x[i_t])
    end

    return (P_e=P_e, x_e=x_e)
end

function test_data_reparameterized(S::Function, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; n_test_s::Int=100, n_test_p::Int=1+length(lb), derivative::Bool=false, nu::Int=2, mu::Real=Inf, integrator::String="Rosenbrock23", abstol::Real=1e-6, reltol::Real=1e-3, reltol_ext::Real=1e-3, arclength::Bool=true)
    P_e      = zeros(1+length(lb), 0)
    tauit_e  = zeros(0)
    xt_e     = zeros(0)
    tauitp_e = zeros(0)

    for n in range(1, n_test_p)
        p_e = lb + rand(length(lb)) .* (ub - lb)

        if derivative
            (t, x, f)                     = S(p_e)
            (tauith, xth, tauitph, s_ext) = reparameterize(t, x, f, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
        else
            (t, x)                        = S(p_e)
            (tauith, xth, tauitph, s_ext) = reparameterize(t, x, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
        end

        s   = unique(rand(n_test_s))
        P   = [s'; repeat(p_e, 1, length(s))]
        P_e = [P_e P]
        append!(tauit_e, tauith(s))
        append!(xt_e, xth(s))
        append!(tauitp_e, tauitph.(s))
    end

    return (P_e=P_e, tauit_e=tauit_e, xt_e=xt_e, tauitp_e=tauitp_e)
end

function update_data_conventional!(x::AbstractVector{<:Real}, P::AbstractMatrix{<:Real}, S::Function, ph::AbstractVector{<:Real}; abstol_p::AbstractVector{<:Real}=1e-16*ones(length(ph)), derivative::Bool=false, nu::Int=2)
    if size(P, 2) != length(x)
        throw(DomainError([size(P, 2) length(x)], "only supports size(P, 2) = length(x)."))
    end

    if length(ph) != size(P, 1)
        throw(DomainError([length(ph) size(P, 1)], "only supports length(ph) = size(P, 1)."))
    end

    if ~(nu in [0, 1, 2])
        throw(DomainError(nu, "only supports nu in [0, 1, 2]."))
    end

    # reuse existing solution if there is one close enough to ph
    if length(ph) > 1
        i_P = argmin(LA.norm.(eachcol(P[2:end,:] .- ph[2:end]), 2))

        if all(abs.(P[2:end,i_P] - ph[2:end]) .<= abstol_p[2:end])
            # @warn "Reusing existing solution."
            ph[2:end] .= P[2:end,i_P]
        end
    end

    # load/compute solution
    if derivative
        (t, x_p, f) = S(ph[2:end])

        if nu == 0
            throw(DomainError(nu, "only supports nu = 0 if derivative = false."))
        elseif nu == 1
            xh = DI.CubicHermiteSpline(f, x_p, t)
        elseif nu == 2
            xh = C2HermiteSpline(t, x_p, f)
        end
    else
        (t, x_p) = S(ph[2:end])

        if nu == 0
            xh = DI.LinearInterpolation(x_p, t)
        elseif nu == 1
            # xh = DI.QuadraticSpline(x_p, t)

            # avoid unphysical oscillations
            xh = DI.PCHIPInterpolation(x_p, t)
        elseif nu == 2
            xh = DI.CubicSpline(x_p, t)
        end
    end

    P = [P ph]
    append!(x, xh(ph[1]))

    return P
end

function update_data_reparameterized!(tauit::AbstractVector{<:Real}, xt::AbstractVector{<:Real}, tauitp::AbstractVector{<:Real}, P::AbstractMatrix{<:Real}, S::Function, ph::AbstractVector{<:Real}, N_ext::Int; abstol_p::AbstractVector{<:Real}=1e-16*ones(length(ph)), derivative::Bool=false, nu::Int=2, mu::Real=Inf, integrator::String="Rosenbrock23", abstol::Real=1e-6, reltol::Real=1e-3, reltol_ext::Real=1e-3, arclength::Bool=true)
    if size(P, 2) != length(tauit)
        throw(DomainError([size(P, 2) length(tauit)], "only supports size(P, 2) = length(tauit)."))
    end

    if size(P, 2) != length(xt)
        throw(DomainError([size(P, 2) length(xt)], "only supports size(P, 2) = length(xt)."))
    end

    if size(P, 2) != length(tauitp)
        throw(DomainError([size(P, 2) length(tauitp)], "only supports size(P, 2) = length(tauitp)."))
    end

    if length(ph) != size(P, 1)
        throw(DomainError([length(ph) size(P, 1)], "only supports length(ph) = size(P, 1)."))
    end

    # reuse existing solution if there is one close enough to ph
    if length(ph) > 1
        i_P = argmin(LA.norm.(eachcol(P[2:end,:] .- ph[2:end]), 2))

        if all(abs.(P[2:end,i_P] - ph[2:end]) .<= abstol_p[2:end])
            # @warn "Reusing existing solution."
            ph[2:end] .= P[2:end,i_P]
        end
    end

    # load/compute solution
    if derivative
        (t, x, f)                     = S(ph[2:end])
        (tauith, xth, tauitph, s_ext) = reparameterize(t, x, f, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
    else
        (t, x)                        = S(ph[2:end])
        (tauith, xth, tauitph, s_ext) = reparameterize(t, x, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
    end

    if length(s_ext) != N_ext
        # @warn "Number of extremas changed."
    end

    P = [P ph]
    append!(tauit, tauith(ph[1]))
    append!(xt, xth(ph[1]))
    append!(tauitp, tauitph(ph[1]))

    return P
end

function try_maximize_logmarginallikelihood!(k::MaternCovariance, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, p_min::AbstractVector{<:Real}, p_max::AbstractVector{<:Real}; abstol_p::AbstractVector{<:Real}=1e-16*ones(length(p_min)), autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    k.sigma[1] = Statistics.std(x)
    lb         = [1/5 * k.sigma[1]; abstol_p .* (p_max - p_min)]
    ub         = [5 * k.sigma[1]; 10 * (p_max - p_min)]

    # avoids issues due to rounding errors
    if true in (k.theta .> ub[2:end])
        @infiltrate
        i_ub          = k.theta .> ub[2:end]
        k.theta[i_ub] = ub[2:end][i_ub]
    end

    # try
        maximize_logmarginallikelihood!(k, P, x, lb, ub, optimizer="OOJL-LBFGS", autodiff=autodiff, symmetric=symmetric, parallel=parallel)
    # catch exception
    #     # println(exception)

    #     @warn "Log marginal likelihood maximization failed. Retrying with different optimizer."
    #     maximize_logmarginallikelihood!(k, P, x, lb, ub, optimizer="ONL-CCSAQ", autodiff=autodiff, symmetric=symmetric, parallel=parallel)
    # end

    return nothing
end

function try_cholesky!(k::MaternCovariance, P::AbstractMatrix{<:Real}; symmetric::Bool=true, parallel::Bool=false)
    try
        Kp = covariance_matrix(k, P, P, symmetric=symmetric, parallel=parallel)
        C  = LA.cholesky(Kp, LA.RowMaximum(), tol=1e-16)

        return (Kp=Kp, C=C)
    catch exception
        println(exception)
        rethrow()
    end

    return nothing
end

function conventional_doe_step!(k::MaternCovariance, K::MatrixOrNothing, P::AbstractMatrix{<:Real}, x::AbstractVector{<:Real}, p_min::AbstractVector{<:Real}, p_max::AbstractVector{<:Real}, P_e::AbstractMatrix{<:Real}, maximize_lml::Bool; abstol_p::AbstractVector{<:Real}=1e-16*ones(length(p_min)), design::String="variance", n_design::Int=1, update::Bool=true, C::CholeskyOrNothing=nothing, autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false)
    # maximize log marginal likelihood
    lml_success = true

    if maximize_lml
        @info "Maximizing log marginal likelihood."
        try
            try_maximize_logmarginallikelihood!(k, P, x, p_min, p_max, abstol_p=abstol_p, autodiff=autodiff, symmetric=symmetric, parallel=parallel)
        catch exception
            println(exception)

            lml_success = false
        end
    end

    # update covariance matrix
    if (~maximize_lml || ~lml_success) && ~isnothing(K)
        N  = length(x)
        kp = zeros(N)

        for i in range(1, N-1)
            kp[i] = k.k(P[:,i], P[:,N], sigma=k.sigma[1], theta=k.theta)
        end

        kp[N] = k.k(P[:,N], P[:,N], sigma=k.sigma[1], theta=k.theta) + k.sigma[1]^2 * k.sigma[2]^2

        Kp              = zeros(N, N)
        Kp[1:N-1,1:N-1] = K
        Kp[N,1:N]       = kp
        Kp[1:N-1,N]     = kp[1:N-1]
    else
        Kp = covariance_matrix(k, P, P, symmetric=symmetric, parallel=parallel)
        kp = Kp[end,1:end]
    end

    # update Cholesky factorization
    if (~maximize_lml || ~lml_success) && update && ~isnothing(C)
        # @info "Updating Cholesky factorization."

        try
            C = update_cholesky(C, kp)
        catch exception
            # println(exception)

            @warn "Cholesky update failed. Maximizing log marginal likelihood."
            try_maximize_logmarginallikelihood!(k, P, x, p_min, p_max, abstol_p=abstol_p, autodiff=autodiff, symmetric=symmetric, parallel=parallel)
            (Kp, C) = try_cholesky!(k, P, symmetric=symmetric, parallel=parallel)
        end
    else
        (Kp, C) = try_cholesky!(k, P, symmetric=symmetric, parallel=parallel)
    end

    # maximize design criterion (continuously)
    ph = zeros(sum(k.n))
    dh = -Inf

    if design == "variance"
        for n in range(1, n_design)
            if dh == -Inf
                ph[:] = p_min + rand(sum(k.n)) .* (p_max - p_min)

                try
                    ph[:] = maximize_variance(k, P, ph, p_min, p_max, C=C, autodiff=autodiff, parallel=parallel)
                    dh    = posterior_variance(k, P, ph, C=C, parallel=parallel)[]
                catch exception
                    # println(exception)

                    @warn "Design optimization failed."
                    dh = -Inf
                end
            else
                break
            end
        end
    elseif design == "variancegradientnorm"
        for n in range(1, n_design)
            if dh == -Inf
                ph[:] = p_min + rand(sum(k.n)) .* (p_max - p_min)

                try
                    ph[:] = maximize_variancegradientnorm(k, P, x, ph, p_min, p_max, C=C, autodiff=autodiff, symmetric=symmetric, parallel=parallel)
                    dh    = posterior_variancegradientnorm(k, P, x, ph, C=C, symmetric=symmetric, parallel=parallel)
                catch exception
                    println(exception)

                    @warn "Design optimization failed."
                    dh = -Inf
                end
            else
                break
            end
        end
    end

    # evaluate posterior for error estimate
    (xh, kh) = posterior(k, P, x, P_e, C=C, variance=true, parallel=parallel)

    return (ph=ph, dh=dh, xh=xh, kh=kh, K=Kp, C=C, lml_success=lml_success)
end

# S(p) returns (t, x) or (t, x, f) with x, f single solution components
# lb and ub only contain bounds on the parameters (assume fixed time domain)
function conventional_doe!(k::MaternCovariance, S::Function, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; P::MatrixOrNothing=nothing, x::VectorOrNothing=nothing, P_e::MatrixOrNothing=nothing, x_e::VectorOrNothing=nothing, maxiters::Int=100, reltol_x::Real=1e-2, reltol_k::Real=1e-2, abstol_p::AbstractVector{<:Real}=1e-16*ones(sum(k.n)), derivative::Bool=false, nu::Int=2, sparse::Bool=true, r_lml::Real=0.5, design::String="variance", n_design::Int=10, n_test_t::Int=100, n_test_p::Int=sum(k.n), n_save::Int=10, n_print::Int=10, update::Bool=true, autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false, directory::String="", filename::String="conventional_doe_normalize=$(k.normalize)_standardize=$(k.standardize)_reltol_x=$(reltol_x)_reltol_k=$(reltol_k)_abstol_p=$(abstol_p)_derivative=$(derivative)_nu=$(nu)")
    # arguments are checked downstream
    if length(lb) != sum(k.n)-1
        throw(DomainError([length(lb) sum(k.n)-1], "only supports length(lb) = sum(k.n)-1."))
    end

    if true in (lb .>= ub)
        throw(DomainError([lb ub], "only supports lb < ub."))
    end

    if maxiters <= 0
        throw(DomainError(maxiters, "only supports maxiters > 0."))
    end

    if reltol_x <= 0
        throw(DomainError(reltol_x, "only supports reltol_x > 0."))
    end

    if reltol_k <= 0
        throw(DomainError(reltol_k, "only supports reltol_k > 0."))
    end

    if true in (abstol_p .<= 0)
        throw(DomainError(abstol_p, "only supports abstol_p > 0."))
    end

    if r_lml <= 0 || r_lml > 1
        throw(DomainError(r_lml, "only supports 0 < r_lml <= 1."))
    end

    if ~(design in ["variance", "variancegradientnorm"])
        throw(DomainError(design, """only supports design in ["variance", "variancegradientnorm"]."""))
    end

    if n_design <= 0
        throw(DomainError(n_design, "only supports n_design > 0."))
    end

    if n_test_t <= 0
        throw(DomainError(n_test_t, "only supports n_test_t > 0."))
    end

    if n_test_p <= 0
        throw(DomainError(n_test_p, "only supports n_test_p > 0."))
    end

    if n_save <= 0
        throw(DomainError(n_save, "only supports n_save > 0."))
    end

    if n_print <= 0
        throw(DomainError(n_print, "only supports n_print > 0."))
    end

    if isfile(directory*filename*"_btheta.csv")
        @warn "DOE with filename = $(directory*filename) already exists. Loading this model."

        if ~isnothing(P) || ~isnothing(x) || ~isnothing(P_e) || ~isnothing(x_e)
            @warn "Ignoring optional P, x, P_e and x_e arguments."
        end

        # load existing model
        btheta     = DF.readdlm(directory*filename*"_btheta.csv")[:]
        P          = DF.readdlm(directory*filename*"_P.csv", ',')
        x          = DF.readdlm(directory*filename*"_x.csv")[:]
        U          = DF.readdlm(directory*filename*"_U.csv", ',')
        p          = DF.readdlm(directory*filename*"_p.csv", Int)[:]
        P_e        = DF.readdlm(directory*filename*"_P_e.csv", ',')
        x_e        = DF.readdlm(directory*filename*"_x_e.csv")[:]
        e_x        = DF.readdlm(directory*filename*"_e_x.csv")[:]
        k_x        = DF.readdlm(directory*filename*"_k_x.csv")[:]
        reltol_k_m = DF.readdlm(directory*filename*"_reltol_k_m.csv")[]
        m_lml      = DF.readdlm(directory*filename*"_m_lml.csv", Int)[]

        k.sigma = btheta[1:2]
        k.theta = btheta[3:end]
        C       = LA.CholeskyPivoted(LA.UpperTriangular(U), 'U', p, length(p), 1e-16, 0)

        if e_x[end] <= reltol_x && k_x[end] <= reltol_k
            return (k=k, P=P, x=x, P_e=P_e, x_e=x_e, e_x=e_x, k_x=k_x)
        end

        if length(x) == maxiters
            @warn "DOE did not reach requested accuracy. Try increasing maxiters."

            return (k=k, P=P, x=x, P_e=P_e, x_e=x_e, e_x=e_x, k_x=k_x)
        end

        if k_x[end] <= reltol_k_m * e_x[end]
            maximize_lml = true
            reltol_k_m   = reltol_k_m / 10
            m_lml        = 0
        elseif m_lml / length(x) >= r_lml
            maximize_lml = true
            m_lml        = 0
        else
            maximize_lml = false
            m_lml        = m_lml + 1
        end

        m_0 = length(x) + 1
    else
        if isnothing(P)
            P_i    = initial_design(lb, ub, sparse=sparse)
            (P, x) = initial_data_conventional(S, P_i)
        end

        if isnothing(P_e)
            (P_e, x_e) = test_data_conventional(S, lb, ub, n_test_t=n_test_t, n_test_p=n_test_p)
        end

        e_x          = zeros(0)
        k_x          = zeros(0)
        reltol_k_m   = 1e-2
        C            = nothing
        maximize_lml = false
        m_lml        = 0
        m_0          = length(x) + 1
    end

    ph          = 0
    dh          = 0
    xh          = zeros(size(P_e, 2))
    kh          = zeros(size(P_e, 2))
    K           = nothing
    lml_success = true

    for m in range(m_0, maxiters)
        if mod(m, n_print) == 0
            @info "Performing step no. $m/$maxiters."
        end

        if k.normalize
            t_min = minimum(P[1,:])
            t_max = maximum(P[1,:])
            m_P   = [t_min; lb]
            n_P   = [t_max - t_min; ub - lb]

            normalize!(P, m_P, n_P)
            normalize!(P_e, m_P, n_P)
            normalize!(lb, m_P[2:end], n_P[2:end])
            normalize!(ub, m_P[2:end], n_P[2:end])
        end

        if k.standardize
            (mu_x, sigma_x) = standardize!(x)
        end

        t_min = minimum(P[1,:])
        t_max = maximum(P[1,:])
        p_min = [t_min; lb]
        p_max = [t_max; ub]

        if length(e_x) == 0
            # initial guess for hyperparameters
            k.theta[:] = [1; 5*ones(sum(k.n) - 1)] .* (p_max - p_min)
        end

        try
            (ph, dh, xh[:], kh[:], K, C, lml_success) = conventional_doe_step!(k, K, P, x, p_min, p_max, P_e, maximize_lml, abstol_p=abstol_p, design=design, n_design=n_design, update=update, C=C, autodiff=autodiff, symmetric=symmetric, parallel=parallel)
        catch exception
            println(exception)

            @warn "Returning early."

            return (k=k, P=P, x=x, P_e=P_e, x_e=x_e, e_x=e_x, k_x=k_x)
        end

        if k.normalize
            unnormalize!(P, m_P, n_P)
            unnormalize!(P_e, m_P, n_P)
            unnormalize!(lb, m_P[2:end], n_P[2:end])
            unnormalize!(ub, m_P[2:end], n_P[2:end])
            unnormalize!(ph, m_P, n_P)
        end

        if k.standardize
            unstandardize!(x, mu_x, sigma_x)
            unstandardize!(xh, mu_x, sigma_x)

            kh = sigma_x^2 * kh
        end

        append!(e_x, LA.norm(xh - x_e, 2) / LA.norm(x_e, 2))
        append!(k_x, maximum(kh))

        if mod(m, n_print) == 0
            @info "e_x = $(Printf.@sprintf("%g", e_x[end])) and k_x = $(Printf.@sprintf("%g", k_x[end]))"
        end

        if e_x[end] <= reltol_x && k_x[end] <= reltol_k
            break
        else
            # update data
            if dh == -Inf
                error("Design optimization failed. Try restarting the DOE (and increasing n_design).")
            end

            P = update_data_conventional!(x, P, S, ph, abstol_p=abstol_p, derivative=derivative, nu=nu)
        end

        if mod(m, n_save) == 0
            # save model
            DF.writedlm(directory*filename*"_btheta.csv", [k.sigma; k.theta])
            DF.writedlm(directory*filename*"_P.csv", P, ',')
            DF.writedlm(directory*filename*"_x.csv", x)
            DF.writedlm(directory*filename*"_U.csv", C.U, ',')
            DF.writedlm(directory*filename*"_p.csv", C.p)
            DF.writedlm(directory*filename*"_P_e.csv", P_e, ',')
            DF.writedlm(directory*filename*"_x_e.csv", x_e)
            DF.writedlm(directory*filename*"_e_x.csv", e_x)
            DF.writedlm(directory*filename*"_k_x.csv", k_x)
            DF.writedlm(directory*filename*"_reltol_k_m.csv", reltol_k_m)
            DF.writedlm(directory*filename*"_m_lml.csv", m_lml)
        end

        if k_x[end] <= reltol_k_m * e_x[end]
            maximize_lml = true
            reltol_k_m   = reltol_k_m / 10
            m_lml        = 0
        elseif m_lml / length(x) >= r_lml
            maximize_lml = true
            m_lml        = 0
        elseif lml_success
            maximize_lml = false
            m_lml        = m_lml + 1
        end

        if m == maxiters
            @warn "DOE did not reach requested accuracy. Try increasing maxiters."
        end
    end

    # save model
    DF.writedlm(directory*filename*"_btheta.csv", [k.sigma; k.theta])
    DF.writedlm(directory*filename*"_P.csv", P, ',')
    DF.writedlm(directory*filename*"_x.csv", x)
    DF.writedlm(directory*filename*"_U.csv", C.U, ',')
    DF.writedlm(directory*filename*"_p.csv", C.p)
    DF.writedlm(directory*filename*"_P_e.csv", P_e, ',')
    DF.writedlm(directory*filename*"_x_e.csv", x_e)
    DF.writedlm(directory*filename*"_e_x.csv", e_x)
    DF.writedlm(directory*filename*"_k_x.csv", k_x)
    DF.writedlm(directory*filename*"_reltol_k_m.csv", reltol_k_m)
    DF.writedlm(directory*filename*"_m_lml.csv", m_lml)

    return (k=k, P=P, x=x, P_e=P_e, x_e=x_e, e_x=e_x, k_x=k_x)
end

# automatically learns sqrt for variable="tauitp"
function reparameterized_doe!(k::MaternCovariance, S::Function, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}; variable::String="tauit", P::MatrixOrNothing=nothing, tauit::VectorOrNothing=nothing, xt::VectorOrNothing=nothing, tauitp::VectorOrNothing=nothing, P_e::MatrixOrNothing=nothing, tauit_e::VectorOrNothing=nothing, xt_e::VectorOrNothing=nothing, tauitp_e::VectorOrNothing=nothing, maxiters::Int=100, reltol_v::Real=1e-2, reltol_k::Real=1e-2, abstol_p::AbstractVector{<:Real}=1e-16*ones(sum(k.n)), derivative::Bool=false, nu::Int=2, mu::Real=Inf, integrator::String="Rosenbrock23", abstol::Real=1e-6, reltol::Real=1e-3, reltol_ext::Real=1e-3, arclength::Bool=true, sparse::Bool=true, r_lml::Real=0.2, design::String="variance", n_design::Int=10, n_test_s::Int=100, n_test_p::Int=sum(k.n), n_save::Int=10, n_print::Int=10, update::Bool=true, autodiff::Bool=false, symmetric::Bool=true, parallel::Bool=false, directory::String="", filename::String="reparameterized_doe_variable=$(variable)_normalize=$(k.normalize)_standardize=$(k.standardize)_reltol_v=$(reltol_v)_reltol_k=$(reltol_k)_abstol_p=$(abstol_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_arclength=$(arclength)")
    # arguments are checked downstream
    if length(lb) != sum(k.n)-1
        throw(DomainError([length(lb) sum(k.n)-1], "only supports length(lb) = sum(k.n)-1."))
    end

    if true in (lb .>= ub)
        throw(DomainError([lb ub], "only supports lb < ub."))
    end

    if ~(variable in ["tauit", "xt", "tauitp"])
        throw(DomainError(variable, """only supports variable in ["tauit", "xt", "tauitp"]."""))
    end

    if maxiters <= 0
        throw(DomainError(maxiters, "only supports maxiters > 0."))
    end

    if reltol_v <= 0
        throw(DomainError(reltol_v, "only supports reltol_v > 0."))
    end

    if reltol_k <= 0
        throw(DomainError(reltol_k, "only supports reltol_k > 0."))
    end

    if true in (abstol_p .<= 0)
        throw(DomainError(abstol_p, "only supports abstol_p > 0."))
    end

    if r_lml <= 0 || r_lml > 1
        throw(DomainError(r_lml, "only supports 0 < r_lml <= 1."))
    end

    if ~(design in ["variance", "variancegradientnorm"])
        throw(DomainError(design, """only supports design in ["variance", "variancegradientnorm"]."""))
    end

    if n_design <= 0
        throw(DomainError(n_design, "only supports n_design > 0."))
    end

    if n_test_s <= 0
        throw(DomainError(n_test_s, "only supports n_test_s > 0."))
    end

    if n_test_p <= 0
        throw(DomainError(n_test_p, "only supports n_test_p > 0."))
    end

    if n_save <= 0
        throw(DomainError(n_save, "only supports n_save > 0."))
    end

    if n_print <= 0
        throw(DomainError(n_print, "only supports n_print > 0."))
    end

    if isfile(directory*filename*"_btheta.csv")
        @warn "DOE with filename = $(directory*filename) already exists. Loading this model."

        if ~isnothing(P) || ~isnothing(tauit) || ~isnothing(xt) || ~isnothing(tauitp) || ~isnothing(P_e) || ~isnothing(tauit_e) || ~isnothing(xt_e) || ~isnothing(tauitp_e)
            @warn "Ignoring optional P, tauit, xt, tauitp, P_e, taui_e, xt_e and tauitp_e arguments."
        end

        # load existing model
        btheta     = DF.readdlm(directory*filename*"_btheta.csv")[:]
        P          = DF.readdlm(directory*filename*"_P.csv", ',')
        tauit      = DF.readdlm(directory*filename*"_tauit.csv")[:]
        xt         = DF.readdlm(directory*filename*"_xt.csv")[:]
        tauitp     = DF.readdlm(directory*filename*"_tauitp.csv")[:]
        N_ext      = DF.readdlm(directory*filename*"_N_ext.csv", Int)[]
        U          = DF.readdlm(directory*filename*"_U.csv", ',')
        p          = DF.readdlm(directory*filename*"_p.csv", Int)[:]
        P_e        = DF.readdlm(directory*filename*"_P_e.csv", ',')
        tauit_e    = DF.readdlm(directory*filename*"_tauit_e.csv")[:]
        xt_e       = DF.readdlm(directory*filename*"_xt_e.csv")[:]
        tauitp_e   = DF.readdlm(directory*filename*"_tauitp_e.csv")[:]
        reltol_k_m = DF.readdlm(directory*filename*"_reltol_k_m.csv")[]
        m_lml      = DF.readdlm(directory*filename*"_m_lml.csv", Int)[]

        if variable == "tauit"
            v   = tauit
            v_e = tauit_e
            e_v = DF.readdlm(directory*filename*"_e_tauit.csv")[:]
            k_v = DF.readdlm(directory*filename*"_k_tauit.csv")[:]
        elseif variable == "xt"
            v   = xt
            v_e = xt_e
            e_v = DF.readdlm(directory*filename*"_e_xt.csv")[:]
            k_v = DF.readdlm(directory*filename*"_k_xt.csv")[:]
        elseif variable == "tauitp"
            v   = tauitp
            v_e = tauitp_e
            e_v = DF.readdlm(directory*filename*"_e_tauitp.csv")[:]
            k_v = DF.readdlm(directory*filename*"_k_tauitp.csv")[:]
        end

        k.sigma = btheta[1:2]
        k.theta = btheta[3:end]
        C       = LA.CholeskyPivoted(LA.UpperTriangular(U), 'U', p, length(p), 1e-16, 0)

        if e_v[end] <= reltol_v && k_v[end] <= reltol_k
            return (k=k, P=P, tauit=tauit, xt=xt, tauitp=tauitp, P_e=P_e, tauit_e=tauit_e, xt_e=xt_e, tauitp_e=tauitp_e, e_v=e_v, k_v=k_v)
        end

        if length(v) == maxiters
            @warn "DOE did not reach requested accuracy. Try increasing maxiters."

            return (k=k, P=P, tauit=tauit, xt=xt, tauitp=tauitp, P_e=P_e, tauit_e=tauit_e, xt_e=xt_e, tauitp_e=tauitp_e, e_v=e_v, k_v=k_v)
        end

        if k_v[end] <= reltol_k_m * e_v[end]
            maximize_lml = true
            reltol_k_m   = reltol_k_m / 10
            m_lml        = 0
        elseif m_lml / length(v) >= r_lml
            maximize_lml = true
            m_lml        = 0
        else
            maximize_lml = false
            m_lml        = m_lml + 1
        end

        m_0 = length(v) + 1
    else
        if isnothing(P)
            P_i                           = initial_design(lb, ub, sparse=sparse)
            (P, tauit, xt, tauitp, N_ext) = initial_data_reparameterized(S, P_i, derivative=derivative, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
        end

        DF.writedlm(directory*filename*"_N_ext.csv", N_ext)

        if isnothing(P_e)
            (P_e, tauit_e, xt_e, tauitp_e) = test_data_reparameterized(S, lb, ub, n_test_s=n_test_s, n_test_p=n_test_p, derivative=derivative, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
        end

        if variable == "tauit"
            v   = tauit
            v_e = tauit_e
        elseif variable == "xt"
            v   = xt
            v_e = xt_e
        elseif variable == "tauitp"
            v   = tauitp
            v_e = tauitp_e
        end

        e_v          = zeros(0)
        k_v          = zeros(0)
        reltol_k_m   = 1e-2
        C            = nothing
        maximize_lml = false
        m_lml        = 0
        m_0          = length(v) + 1
    end

    ph          = 0
    dh          = 0
    vh          = zeros(size(P_e, 2))
    kh          = zeros(size(P_e, 2))
    K           = nothing
    lml_success = true

    for m in range(m_0, maxiters)
        if mod(m, n_print) == 0
            @info "Performing step no. $m/$maxiters."
        end

        if k.normalize
            s_min = minimum(P[1,:])
            s_max = maximum(P[1,:])
            m_P   = [s_min; lb]
            n_P   = [s_max - s_min; ub - lb]

            normalize!(P, m_P, n_P)
            normalize!(P_e, m_P, n_P)
            normalize!(lb, m_P[2:end], n_P[2:end])
            normalize!(ub, m_P[2:end], n_P[2:end])
        end

        if k.standardize
            (mu_v, sigma_v) = standardize!(v)
        end

        s_min = minimum(P[1,:])
        s_max = maximum(P[1,:])
        p_min = [s_min; lb]
        p_max = [s_max; ub]

        if length(e_v) == 0
            # initial guess for hyperparameters
            k.theta[:] = [1; 5*ones(sum(k.n) - 1)] .* (p_max - p_min)
        end

        try
            if variable == "tauit" || variable == "xt"
                (ph, dh, vh[:], kh[:], K, C, lml_success) = conventional_doe_step!(k, K, P, v, p_min, p_max, P_e, maximize_lml, abstol_p=abstol_p, design=design, n_design=n_design, update=update, C=C, autodiff=autodiff, symmetric=symmetric, parallel=parallel)
            elseif variable == "tauitp"
                (ph, dh, vh[:], kh[:], K, C, lml_success) = conventional_doe_step!(k, K, P, sqrt.(v), p_min, p_max, P_e, maximize_lml, abstol_p=abstol_p, design=design, n_design=n_design, update=update, C=C, autodiff=autodiff, symmetric=symmetric, parallel=parallel)

                vh[:] = vh.^2
            end
        catch exception
            println(exception)

            @warn "Returning early."

            return (k=k, P=P, tauit=tauit, xt=xt, tauitp=tauitp, P_e=P_e, tauit_e=tauit_e, xt_e=xt_e, tauitp_e=tauitp_e, e_v=e_v, k_v=k_v)
        end

        if k.normalize
            unnormalize!(P, m_P, n_P)
            unnormalize!(P_e, m_P, n_P)
            unnormalize!(lb, m_P[2:end], n_P[2:end])
            unnormalize!(ub, m_P[2:end], n_P[2:end])
            unnormalize!(ph, m_P, n_P)
        end

        if k.standardize
            unstandardize!(v, mu_v, sigma_v)
            unstandardize!(vh, mu_v, sigma_v)

            kh = sigma_v^2 * kh
        end

        append!(e_v, LA.norm(vh - v_e, 2) / LA.norm(v_e, 2))
        append!(k_v, maximum(kh))

        if mod(m, n_print) == 0
            @info "e_v = $(Printf.@sprintf("%g", e_v[end])) and k_v = $(Printf.@sprintf("%g", k_v[end]))"
        end

        if e_v[end] <= reltol_v && k_v[end] <= reltol_k
            break
        else
            # update data
            if dh == -Inf
                error("Design optimization failed. Try restarting the DOE (and increasing n_design).")
            end

            P = update_data_reparameterized!(tauit, xt, tauitp, P, S, ph, N_ext, abstol_p=abstol_p, derivative=derivative, nu=nu, mu=mu, integrator=integrator, abstol=abstol, reltol=reltol, reltol_ext=reltol_ext, arclength=arclength)
        end

        if mod(m, n_save) == 0
            # save model
            DF.writedlm(directory*filename*"_btheta.csv", [k.sigma; k.theta])
            DF.writedlm(directory*filename*"_P.csv", P, ',')
            DF.writedlm(directory*filename*"_tauit.csv", tauit)
            DF.writedlm(directory*filename*"_xt.csv", xt)
            DF.writedlm(directory*filename*"_tauitp.csv", tauitp)
            DF.writedlm(directory*filename*"_U.csv", C.U, ',')
            DF.writedlm(directory*filename*"_p.csv", C.p)
            DF.writedlm(directory*filename*"_P_e.csv", P_e, ',')
            DF.writedlm(directory*filename*"_tauit_e.csv", tauit_e)
            DF.writedlm(directory*filename*"_xt_e.csv", xt_e)
            DF.writedlm(directory*filename*"_tauitp_e.csv", tauitp_e)
            DF.writedlm(directory*filename*"_reltol_k_m.csv", reltol_k_m)
            DF.writedlm(directory*filename*"_m_lml.csv", m_lml)

            if variable == "tauit"
                DF.writedlm(directory*filename*"_e_tauit.csv", e_v)
                DF.writedlm(directory*filename*"_k_tauit.csv", k_v)
            elseif variable == "xt"
                DF.writedlm(directory*filename*"_e_xt.csv", e_v)
                DF.writedlm(directory*filename*"_k_xt.csv", k_v)
            elseif variable == "tauitp"
                DF.writedlm(directory*filename*"_e_tauitp.csv", e_v)
                DF.writedlm(directory*filename*"_k_tauitp.csv", k_v)
            end
        end

        if k_v[end] <= reltol_k_m * e_v[end]
            maximize_lml = true
            reltol_k_m   = reltol_k_m / 10
            m_lml        = 0
        elseif m_lml / length(v) >= r_lml
            maximize_lml = true
            m_lml        = 0
        elseif lml_success
            maximize_lml = false
            m_lml        = m_lml + 1
        end

        if m == maxiters
            @warn "DOE did not reach requested accuracy. Try increasing maxiters."
        end
    end

    # save model
    DF.writedlm(directory*filename*"_btheta.csv", [k.sigma; k.theta])
    DF.writedlm(directory*filename*"_P.csv", P, ',')
    DF.writedlm(directory*filename*"_tauit.csv", tauit)
    DF.writedlm(directory*filename*"_xt.csv", xt)
    DF.writedlm(directory*filename*"_tauitp.csv", tauitp)
    DF.writedlm(directory*filename*"_U.csv", C.U, ',')
    DF.writedlm(directory*filename*"_p.csv", C.p)
    DF.writedlm(directory*filename*"_P_e.csv", P_e, ',')
    DF.writedlm(directory*filename*"_tauit_e.csv", tauit_e)
    DF.writedlm(directory*filename*"_xt_e.csv", xt_e)
    DF.writedlm(directory*filename*"_tauitp_e.csv", tauitp_e)
    DF.writedlm(directory*filename*"_reltol_k_m.csv", reltol_k_m)
    DF.writedlm(directory*filename*"_m_lml.csv", m_lml)

    return (k=k, P=P, tauit=tauit, xt=xt, tauitp=tauitp, P_e=P_e, tauit_e=tauit_e, xt_e=xt_e, tauitp_e=tauitp_e, e_v=e_v, k_v=k_v)
end
