mutable struct MaternCovariance
    const nu::AbstractVector{Float64}
    const n::AbstractVector{Int}
    sigma::AbstractVector{Float64}
    theta::AbstractVector{Float64}
    const autodiff::Bool
    const normalize::Bool
    const standardize::Bool
    const reparameterize::Bool
    const k::Function
    const k_sigma::Function
    const k_theta::Function
    const k_pp::Function
    const k_pps::Function
    const ksi::Function
end

function MaternCovariance(nu::Real, n::Int=1; sigma::AbstractVector{<:Real}=[1, 1e-8], theta::AbstractVector{<:Real}=5*ones(n), autodiff::Bool=true, normalize::Bool=false, standardize::Bool=false, reparameterize::Bool=false)
    if ~(nu in [Inf, 0, 1, 2])
        throw(DomainError(nu, "only supports nu in [Inf, 0, 1, 2]."))
    end

    if nu == Inf
        if reparameterize
            return ReparameterizedCInfMaternCovariance(n, sigma=sigma, theta=theta, autodiff=autodiff, normalize=normalize, standardize=standardize)
        else
            return CInfMaternCovariance(n, sigma=sigma, theta=theta, autodiff=autodiff, normalize=normalize, standardize=standardize)
        end
    elseif nu == 0
        return C0MaternCovariance(n, sigma=sigma, theta=theta, autodiff=autodiff, normalize=normalize, standardize=standardize)
    elseif nu == 1
        return C1MaternCovariance(n, sigma=sigma, theta=theta, autodiff=autodiff, normalize=normalize, standardize=standardize)
    elseif nu == 2
        return C2MaternCovariance(n, sigma=sigma, theta=theta, autodiff=autodiff, normalize=normalize, standardize=standardize)
    end
end

function CInfMaternCovariance(n::Int=1; sigma::AbstractVector{<:Real}=[1, 1e-8], theta::AbstractVector{<:Real}=5*ones(n), autodiff::Bool=true, normalize::Bool=false, standardize::Bool=false)
    if n < 1
        throw(DomainError(n, "only supports n >= 1."))
    end

    if length(sigma) != 2
        throw(DomainError(length(sigma), "only supports length(sigma) = 2."))
    end

    if true in (sigma .<= 0)
        throw(DomainError(sigma, "only supports sigma > 0."))
    end

    if length(theta) != n
        throw(DomainError([length(theta) n], "only supports length(theta) = n."))
    end

    if true in (theta .<= 0)
        throw(DomainError(theta, "only supports theta > 0."))
    end

    k       = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * exp( -sum( (p - pp).^2 ./ (2*theta.^2) ) )
    k_sigma = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> 2*sigma * exp( -sum( (p - pp).^2 ./ (2*theta.^2) ) )

    if autodiff
        k_theta_ad = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(theta -> k(p, pp, sigma=sigma, theta=theta), theta)
        k_pp_ad    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(pp -> k(p, pp, sigma=sigma, theta=theta), pp)
        k_pps_ad   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.jacobian(pp -> k_pp_ad(p, pp, sigma=sigma, theta=theta), pp)

        function ksi_ad(p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta)
            # only integrate over first component
            ksi = u -> k(p_i, [u; pp[2:end]], sigma=sigma, theta=theta) * k(p_j, [u; pp[2:end]], sigma=sigma, theta=theta)
            I   = Integrals.IntegralProblem((u, p) -> ksi(u), (lb, pp[1]))
            S   = Integrals.solve(I, Integrals.QuadGKJL(), maxiters=100, abstol=1e-16, reltol=1e-8)

            if SciMLBase.successful_retcode(S.retcode)
                return S.u
            else
                throw(ErrorException("Integration failed."))
            end
        end
    else
        k_theta = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * (p - pp).^2 ./ theta.^3 * exp( -sum( (p - pp).^2 ./ (2*theta.^2) ) )
        k_pp    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * (p - pp) ./ theta.^2 * exp( -sum( (p - pp).^2 ./ (2*theta.^2) ) )

        function k_pps(p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta)
            K_pps = zeros(n, n)

            for j in range(1, n)
                # K_pps symmetric (Schwarz's theorem)
                for i in range(1, j)
                    if i == j
                        K_pps[i,j] = sigma^2 * ( (p[i] - pp[i])^2 / theta[i]^4 - 1 / theta[i]^2 ) * exp( -sum( (p - pp).^2 ./ (2*theta.^2) ) )
                    else
                        K_pps[i,j] = sigma^2 * (p[i] - pp[i]) / theta[i]^2 * (p[j] - pp[j]) / theta[j]^2 * exp( -sum( (p - pp).^2 ./ (2*theta.^2) ) )
                        K_pps[j,i] = K_pps[i,j]
                    end
                end
            end

            return K_pps
        end

        function ksi(p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta)
            # only integrate over first component
            lb = (lb - (p_i[1] + p_j[1]) / 2 ) / theta[1]
            ub = (pp[1] - (p_i[1] + p_j[1]) / 2 ) / theta[1]

            # set first components of p_i, p_j, pp to zero for additional terms from product construction and set sigma = 1 in those cases
            ksi = sqrt(pi) / 2 * sigma^4 * theta[1] * exp( -(p_i[1] - p_j[1])^2 / (4*theta[1]^2) ) * SF.erf(lb, ub) * k([0; p_i[2:end]], [0; pp[2:end]], sigma=1, theta=theta) * k([0; p_j[2:end]], [0; pp[2:end]], sigma=1, theta=theta)

            return ksi
        end
    end

    if autodiff
        return MaternCovariance([Inf], [n], sigma, theta, autodiff, normalize, standardize, false, k, k_sigma, k_theta_ad, k_pp_ad, k_pps_ad, ksi_ad)
    else
        return MaternCovariance([Inf], [n], sigma, theta, autodiff, normalize, standardize, false, k, k_sigma, k_theta, k_pp, k_pps, ksi)
    end
end

function ReparameterizedCInfMaternCovariance(n::Int=1; sigma::AbstractVector{<:Real}=[1, 1e-8], theta::AbstractVector{<:Real}=log.(5*ones(n)), autodiff::Bool=true, normalize::Bool=false, standardize::Bool=false)
    if n < 1
        throw(DomainError(n, "only supports n >= 1."))
    end

    if length(sigma) != 2
        throw(DomainError(length(sigma), "only supports length(sigma) = 2."))
    end

    if true in (sigma .<= 0)
        throw(DomainError(sigma, "only supports sigma > 0."))
    end

    if length(theta) != n
        throw(DomainError([length(theta) n], "only supports length(theta) = n."))
    end

    k       = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * exp( -sum( (p - pp).^2 ./ (2*exp.(2*theta)) ) )
    k_sigma = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> 2*sigma * exp( -sum( (p - pp).^2 ./ (2*exp.(2*theta)) ) )

    if autodiff
        k_theta_ad = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(theta -> k(p, pp, sigma=sigma, theta=theta), theta)
        k_pp_ad    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(pp -> k(p, pp, sigma=sigma, theta=theta), pp)
        k_pps_ad   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.jacobian(pp -> k_pp_ad(p, pp, sigma=sigma, theta=theta), pp)

        function ksi_ad(p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta)
            # only integrate over first component
            ksi = u -> k(p_i, [u; pp[2:end]], sigma=sigma, theta=theta) * k(p_j, [u; pp[2:end]], sigma=sigma, theta=theta)
            I   = Integrals.IntegralProblem((u, p) -> ksi(u), (lb, pp[1]))
            S   = Integrals.solve(I, Integrals.QuadGKJL(), maxiters=100, abstol=1e-16, reltol=1e-8)

            if SciMLBase.successful_retcode(S.retcode)
                return S.u
            else
                throw(ErrorException("Integration failed."))
            end
        end
    else
        k_theta = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * (p - pp).^2 ./ exp.(2*theta) * exp( -sum( (p - pp).^2 ./ (2*exp.(2*theta)) ) )
        k_pp    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * (p - pp) ./ exp.(2*theta) * exp( -sum( (p - pp).^2 ./ (2*exp.(2*theta)) ) )
        k_pps   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> throw(ErrorException("k_pps not implemented for nu = Inf and autodiff = false."))
        ksi     = (p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> throw(ErrorException("ksi not implemented for nu = Inf and autodiff = false."))
    end

    if autodiff
        return MaternCovariance([Inf], [n], sigma, theta, autodiff, normalize, standardize, true, k, k_sigma, k_theta_ad, k_pp_ad, k_pps_ad, ksi_ad)
    else
        return MaternCovariance([Inf], [n], sigma, theta, autodiff, normalize, standardize, true, k, k_sigma, k_theta, k_pp, k_pps, ksi)
    end
end

function C0MaternCovariance(n::Int=1; sigma::AbstractVector{<:Real}=[1, 1e-8], theta::AbstractVector{<:Real}=5*ones(n), autodiff::Bool=true, normalize::Bool=false, standardize::Bool=false)
    if n < 1
        throw(DomainError(n, "only supports n >= 1."))
    end

    if length(sigma) != 2
        throw(DomainError(length(sigma), "only supports length(sigma) = 2."))
    end

    if true in (sigma .<= 0)
        throw(DomainError(sigma, "only supports sigma > 0."))
    end

    if length(theta) != n
        throw(DomainError([length(theta) n], "only supports length(theta) = n."))
    end

    if true in (theta .<= 0)
        throw(DomainError(theta, "only supports theta > 0."))
    end

    k       = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * exp( -sum( abs.(p - pp) ./ theta ) )
    k_sigma = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> 2*sigma * exp( -sum( abs.(p - pp) ./ theta ) )

    if autodiff
        k_theta_ad = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(theta -> k(p, pp, sigma=sigma, theta=theta), theta)
        k_pp_ad    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(pp -> k(p, pp, sigma=sigma, theta=theta), pp)
        k_pps_ad   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> throw(ErrorException("k_pps not defined for nu = 0."))

        function ksi_ad(p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta)
            ksi = u -> k(p_i, [u; pp[2:end]], sigma=sigma, theta=theta) * k(p_j, [u; pp[2:end]], sigma=sigma, theta=theta)
            I   = Integrals.IntegralProblem((u, p) -> ksi(u), (lb, pp[1]))
            S   = Integrals.solve(I, Integrals.QuadGKJL(order=1), maxiters=100, abstol=1e-16, reltol=1e-8)

            if SciMLBase.successful_retcode(S.retcode)
                return S.u
            else
                throw(ErrorException("Integration failed."))
            end
        end
    else
        k_theta = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * abs.(p - pp) ./ theta.^2 * exp( -sum( abs.(p - pp) ./ theta ) )
        k_pp    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * ( -(p .< pp) + (p .>= pp) ) ./ theta * exp( -sum( abs.(p - pp) ./ theta ) )
        k_pps   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> throw(ErrorException("k_pps not defined for nu = 0."))

        function ksi(p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta)
            if lb > pp[1]
                lbp = pp[1]
                ubp = lb
            else
                lbp = lb
                ubp = pp[1]
            end

            # simplify cases (only integrate over first component)
            p_min = min(p_i[1], p_j[1])
            p_max = max(p_i[1], p_j[1])

            if lbp < p_min
                if p_min < ubp
                    ksi = sigma^4 / 2 * theta[1] * ( exp((p_min - p_max) / theta[1]) - exp((2*lbp - p_min - p_max) / theta[1]) )

                    if p_max < ubp
                        ksi = ksi + sigma^4 * exp((p_min - p_max) / theta[1]) * (p_max - p_min)
                        ksi = ksi - sigma^4 / 2 * theta[1] * ( exp((p_min + p_max - 2*ubp) / theta[1]) - exp((p_min - p_max) / theta[1]) )
                    else
                        ksi = ksi + sigma^4 * exp((p_min - p_max) / theta[1]) * (ubp - p_min)
                    end
                else
                    ksi = sigma^4 / 2 * theta[1] * ( exp((2*ubp - p_min - p_max) / theta[1]) - exp((2*lbp - p_min - p_max) / theta[1]) )
                end
            elseif lbp < p_max
                if p_max < ubp
                    ksi = sigma^4 * exp((p_min - p_max) / theta[1]) * (p_max - lbp)
                    ksi = ksi - sigma^4 / 2 * theta[1] * ( exp((p_min + p_max - 2*ubp) / theta[1]) - exp((p_min - p_max) / theta[1]) )
                else
                    ksi = sigma^4 * exp((p_min - p_max) / theta[1]) * (ubp - lbp)
                end
            else
                ksi = -sigma^4 / 2 * theta[1] * ( exp((p_min + p_max - 2*ubp) / theta[1]) - exp((p_min + p_max - 2*lbp) / theta[1]) )
            end

            # set first components of p_i, p_j, pp to zero for additional terms from product construction and set sigma = 1 in those cases
            ksi = ksi * k([0; p_i[2:end]], [0; pp[2:end]], sigma=1, theta=theta) * k([0; p_j[2:end]], [0; pp[2:end]], sigma=1, theta=theta)

            if lb > pp[1]
                return -ksi
            else
                return ksi
            end
        end
    end

    if autodiff
        return MaternCovariance([0], [n], sigma, theta, autodiff, normalize, standardize, false, k, k_sigma, k_theta_ad, k_pp_ad, k_pps_ad, ksi_ad)
    else
        return MaternCovariance([0], [n], sigma, theta, autodiff, normalize, standardize, false, k, k_sigma, k_theta, k_pp, k_pps, ksi)
    end
end

function C1MaternCovariance(n::Int=1; sigma::AbstractVector{<:Real}=[1, 1e-8], theta::AbstractVector{<:Real}=5*ones(n), autodiff::Bool=true, normalize::Bool=false, standardize::Bool=false)
    if n < 1
        throw(DomainError(n, "only supports n >= 1."))
    end

    if length(sigma) != 2
        throw(DomainError(length(sigma), "only supports length(sigma) = 2."))
    end

    if true in (sigma .<= 0)
        throw(DomainError(sigma, "only supports sigma > 0."))
    end

    if length(theta) != n
        throw(DomainError([length(theta) n], "only supports length(theta) = n."))
    end

    if true in (theta .<= 0)
        throw(DomainError(theta, "only supports theta > 0."))
    end

    # polynomial terms in product construction for multivariate case
    c       = (p, pp, theta) -> prod(1 .+ sqrt(3)*abs.(p - pp) ./ theta)
    k       = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * c(p, pp, theta) * exp( -sqrt(3)*sum( abs.(p - pp) ./ theta ) )
    k_sigma = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> 2*sigma * c(p, pp, theta) * exp( -sqrt(3)*sum( abs.(p - pp) ./ theta ) )

    if autodiff
        k_theta_ad = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(theta -> k(p, pp, sigma=sigma, theta=theta), theta)
        k_pp_ad    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(pp -> k(p, pp, sigma=sigma, theta=theta), pp)
        k_pps_ad   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.jacobian(pp -> k_pp_ad(p, pp, sigma=sigma, theta=theta), pp)

        function ksi_ad(p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta)
            # only integrate over first component
            ksi = u -> k(p_i, [u; pp[2:end]], sigma=sigma, theta=theta) * k(p_j, [u; pp[2:end]], sigma=sigma, theta=theta)
            I   = Integrals.IntegralProblem((u, p) -> ksi(u), (lb, pp[1]))
            S   = Integrals.solve(I, Integrals.QuadGKJL(order=2), maxiters=100, abstol=1e-16, reltol=1e-8)

            if SciMLBase.successful_retcode(S.retcode)
                return S.u
            else
                throw(ErrorException("Integration failed."))
            end
        end
    else
        # indices of non-differentiated components
        i = [setdiff(1:n, j) for j in range(1, n)]

        # coefficients of all components
        c_theta = (p, pp, theta) -> 3*(p - pp).^2 ./ theta.^3 .* [c(p[i[j]], pp[i[j]], theta[i[j]]) for j in range(1, n)]
        c_pp    = (p, pp, theta) -> 3*(p - pp) ./ theta.^2 .* [c(p[i[j]], pp[i[j]], theta[i[j]]) for j in range(1, n)]
        k_theta = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * c_theta(p, pp, theta) * exp( -sqrt(3)*sum( abs.(p - pp) ./ theta ) )
        k_pp    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * c_pp(p, pp, theta) * exp( -sqrt(3)*sum( abs.(p - pp) ./ theta ) )
        k_pps   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> throw(ErrorException("k_pps not implemented for nu = 1 and autodiff = false."))
        ksi     = (p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> throw(ErrorException("ksi not implemented for nu = 1 and autodiff = false."))
    end

    if autodiff
        return MaternCovariance([1], [n], sigma, theta, autodiff, normalize, standardize, false, k, k_sigma, k_theta_ad, k_pp_ad, k_pps_ad, ksi_ad)
    else
        return MaternCovariance([1], [n], sigma, theta, autodiff, normalize, standardize, false, k, k_sigma, k_theta, k_pp, k_pps, ksi)
    end
end

function C2MaternCovariance(n::Int=1; sigma::AbstractVector{<:Real}=[1, 1e-8], theta::AbstractVector{<:Real}=5*ones(n), autodiff::Bool=true, normalize::Bool=false, standardize::Bool=false)
    if n < 1
        throw(DomainError(n, "only supports n >= 1."))
    end

    if length(sigma) != 2
        throw(DomainError(length(sigma), "only supports length(sigma) = 2."))
    end

    if true in (sigma .<= 0)
        throw(DomainError(sigma, "only supports sigma > 0."))
    end

    if length(theta) != n
        throw(DomainError([length(theta) n], "only supports length(theta) = n."))
    end

    if true in (theta .<= 0)
        throw(DomainError(theta, "only supports theta > 0."))
    end

    # polynomial terms in product construction for multivariate case
    c       = (p, pp, theta) -> prod(1 .+ sqrt(5)*abs.(p - pp) ./ theta + 5*(p - pp).^2 ./ (3*theta.^2))
    k       = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * c(p, pp, theta) * exp( -sqrt(5)*sum( abs.(p - pp) ./ theta ) )
    k_sigma = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> 2*sigma * c(p, pp, theta) * exp( -sqrt(5)*sum( abs.(p - pp) ./ theta ) )

    if autodiff
        k_theta_ad = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(theta -> k(p, pp, sigma=sigma, theta=theta), theta)
        k_pp_ad    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(pp -> k(p, pp, sigma=sigma, theta=theta), pp)
        k_pps_ad   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.jacobian(pp -> k_pp_ad(p, pp, sigma=sigma, theta=theta), pp)

        function ksi_ad(p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta)
            # only integrate over first component
            ksi = u -> k(p_i, [u; pp[2:end]], sigma=sigma, theta=theta) * k(p_j, [u; pp[2:end]], sigma=sigma, theta=theta)
            I   = Integrals.IntegralProblem((u, p) -> ksi(u), (lb, pp[1]))
            S   = Integrals.solve(I, Integrals.QuadGKJL(order=3), maxiters=100, abstol=1e-16, reltol=1e-8)

            if SciMLBase.successful_retcode(S.retcode)
                return S.u
            else
                throw(ErrorException("Integration failed."))
            end
        end
    else
        # indices of non-differentiated components
        i = [setdiff(1:n, j) for j in range(1, n)]

        # coefficients of all components
        c_theta = (p, pp, theta) -> ( 5*(p - pp).^2 ./ (3*theta.^3) + 5*sqrt(5)*abs.(p - pp).^3 ./ (3*theta.^4) ) .* [c(p[i[j]], pp[i[j]], theta[i[j]]) for j in range(1, n)]
        c_pp    = (p, pp, theta) -> ( 5*(p - pp) ./ (3*theta.^2) + ( -(p .< pp) + (p .>= pp) ) .* ( 5*sqrt(5)*(p - pp).^2 ./ (3*theta.^3) ) ) .* [c(p[i[j]], pp[i[j]], theta[i[j]]) for j in range(1, n)]
        k_theta = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * c_theta(p, pp, theta) * exp( -sqrt(5)*sum( abs.(p - pp) ./ theta ) )
        k_pp    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> sigma^2 * c_pp(p, pp, theta) * exp( -sqrt(5)*sum( abs.(p - pp) ./ theta ) )
        k_pps   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> throw(ErrorException("k_pps not implemented for nu = 2 and autodiff = false."))
        ksi     = (p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> throw(ErrorException("ksi not implemented for nu = 2 and autodiff = false."))
    end

    if autodiff
        return MaternCovariance([2], [n], sigma, theta, autodiff, normalize, standardize, false, k, k_sigma, k_theta_ad, k_pp_ad, k_pps_ad, ksi_ad)
    else
        return MaternCovariance([2], [n], sigma, theta, autodiff, normalize, standardize, false, k, k_sigma, k_theta, k_pp, k_pps, ksi)
    end
end

function isconsistent(k::MaternCovariance)
    for nu in k.nu
        if ~(nu in [Inf, 0, 1, 2])
            throw(DomainError(k.nu, "MaternCovariance only supports k.nu in [Inf, 0, 1, 2]."))
        end
    end

    if length(k.nu) != length(k.n)
        throw(DomainError([length(k.nu) length(k.n)], "MaternCovariance only supports length(k.nu) = length(k.n)."))
    end

    for n in k.n
        if n < 1
            throw(DomainError(k.n, "MaternCovariance only supports k.n >= 1."))
        end
    end

    if length(k.sigma) != 2
        throw(DomainError(length(k.sigma), "MaternCovariance only supports length(k.sigma) = 2."))
    end

    if true in (k.sigma .<= 0)
        throw(DomainError(k.sigma, "MaternCovariance only supports k.sigma > 0."))
    end

    if length(k.theta) != sum(k.n)
        throw(DomainError([length(k.theta) sum(k.n)], "MaternCovariance only supports length(k.theta) = sum(k.n)."))
    end

    if ~k.reparameterize && (true in (k.theta .<= 0))
        throw(DomainError(k.theta, "MaternCovariance only supports k.theta > 0 if k.reparameterize = false."))
    end

    return nothing
end

function *(k::MaternCovariance, kp::MaternCovariance)
    isconsistent(k)
    isconsistent(kp)

    nu    = [k.nu; kp.nu]
    n     = [k.n; kp.n]
    sigma = k.sigma
    theta = [k.theta; kp.theta]

    if k.autodiff != kp.autodiff
        throw(DomainError([k.autodiff kp.autodiff], "only supports k.autodiff = kp.autodiff."))
    end

    if k.normalize != kp.normalize
        throw(DomainError([k.normalize kp.normalize], "only supports k.normalize = kp.normalize."))
    end

    if k.standardize != kp.standardize
        throw(DomainError([k.standardize kp.standardize], "only supports k.standardize = kp.standardize."))
    end

    if k.reparameterize != kp.reparameterize
        throw(DomainError([k.reparameterize kp.reparameterize], "only supports k.reparameterize = kp.reparameterize."))
    end

    # indices of respective components
    i_k  = 1:sum(k.n)
    i_kp = sum(k.n)+1:sum(k.n)+sum(kp.n)

    # set kp.sigma[1] = 1
    kpp       = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> k.k(p[i_k], pp[i_k], sigma=sigma, theta=theta[i_k]) * kp.k(p[i_kp], pp[i_kp], sigma=1, theta=theta[i_kp])
    kpp_sigma = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> k.k_sigma(p[i_k], pp[i_k], sigma=sigma, theta=theta[i_k]) * kp.k(p[i_kp], pp[i_kp], sigma=1, theta=theta[i_kp])

    if k.autodiff
        kpp_theta = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(theta -> kpp(p, pp, sigma=sigma, theta=theta), theta)
        kpp_pp    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.gradient(pp -> kpp(p, pp, sigma=sigma, theta=theta), pp)
        kpp_pps   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> FD.jacobian(pp -> kpp_pp(p, pp, sigma=sigma, theta=theta), pp)
    else
        kpp_theta = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> [k.k_theta(p[i_k], pp[i_k], sigma=sigma, theta=theta[i_k]) * kp.k(p[i_kp], pp[i_kp], sigma=1, theta=theta[i_kp]); k.k(p[i_k], pp[i_k], sigma=sigma, theta=theta[i_k]) * kp.k_theta(p[i_kp], pp[i_kp], sigma=1, theta=theta[i_kp])]
        kpp_pp    = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> [k.k_pp(p[i_k], pp[i_k], sigma=sigma, theta=theta[i_k]) * kp.k(p[i_kp], pp[i_kp], sigma=1, theta=theta[i_kp]); k.k(p[i_k], pp[i_k], sigma=sigma, theta=theta[i_k]) * kp.k_pp(p[i_kp], pp[i_kp], sigma=1, theta=theta[i_kp])]
        kpp_pps   = (p::AbstractVector{<:Real}, pp::AbstractVector{<:Real}; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> throw(ErrorException("kpp_pps not implemented for autodiff = false."))
    end

    kppsi = (p_i::AbstractVector{<:Real}, p_j::AbstractVector{<:Real}, pp::AbstractVector{<:Real}, lb::Real; sigma::Real=sigma[1], theta::AbstractVector{<:Real}=theta) -> k.ksi(p_i[i_k], p_j[i_k], pp[i_k], lb, sigma=sigma, theta=theta[i_k]) * kp.k(p_i[i_kp], pp[i_kp], sigma=1, theta=theta[i_kp]) * kp.k(p_j[i_kp], pp[i_kp], sigma=1, theta=theta[i_kp])

    return MaternCovariance(nu, n, sigma, theta, k.autodiff, k.normalize, k.standardize, k.reparameterize, kpp, kpp_sigma, kpp_theta, kpp_pp, kpp_pps, kppsi)
end

function covariance_matrix!(K::AbstractMatrix{<:Real}, k::MaternCovariance, P::AbstractMatrix{<:Real}, Pp::VectorOrMatrix; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, symmetric::Bool=false, parallel::Bool=false)
    isconsistent(k)

    if size(K, 1) != size(P, 2)
        throw(DomainError([size(K, 1) size(P, 2)], "only supports size(K, 1) = size(P, 2)."))
    end

    if size(K, 2) != size(Pp, 2)
        throw(DomainError([size(K, 2) size(Pp, 2)], "only supports size(K, 2) = size(Pp, 2)."))
    end

    if size(P, 1) != sum(k.n)
        throw(DomainError([size(P, 1) sum(k.n)], "only supports size(P, 1) = sum(k.n)."))
    end

    if size(Pp, 1) != sum(k.n)
        throw(DomainError([size(Pp, 1) sum(k.n)], "only supports size(Pp, 1) = sum(k.n)."))
    end

    if length(sigma) != 2
        throw(DomainError(length(sigma), "only supports length(sigma) = 2."))
    end

    if true in (sigma .<= 0)
        throw(DomainError(sigma, "only supports sigma > 0."))
    end

    if length(theta) != sum(k.n)
        throw(DomainError([length(theta) sum(k.n)], "only supports length(theta) = sum(k.n)."))
    end

    if ~k.reparameterize && (true in (theta .<= 0))
        throw(DomainError(theta, "only supports theta > 0 if k.reparameterize = false."))
    end

    if symmetric
        if size(P, 2) != size(Pp, 2)
            throw(DomainError([size(P, 2) size(Pp, 2)], "only supports size(P, 2) = size(Pp, 2) if symmetric = true."))
        end
    end

    if symmetric && parallel
        Threads.@threads for j in range(1, size(Pp, 2))
            # inner loop over rows (column-major ordering), only build upper triangle (K symmetric)
            for i in range(1, j)
                if P[:,i] == Pp[:,j]
                    # additive Gaussian noise
                    K[i,j] = k.k(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta) + sigma[1]^2 * sigma[2]^2
                else
                    K[i,j] = k.k(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
                end
            end
        end
    end

    if symmetric && ~parallel
        for j in range(1, size(Pp, 2))
            for i in range(1, j)
                if P[:,i] == Pp[:,j]
                    K[i,j] = k.k(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta) + sigma[1]^2 * sigma[2]^2
                else
                    K[i,j] = k.k(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
                end
            end
        end
    end

    if ~symmetric && parallel
        Threads.@threads for j in range(1, size(Pp, 2))
            for i in range(1, size(P, 2))
                if P[:,i] == Pp[:,j]
                    K[i,j] = k.k(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta) + sigma[1]^2 * sigma[2]^2
                else
                    K[i,j] = k.k(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
                end
            end
        end
    end

    if ~symmetric && ~parallel
        for j in range(1, size(Pp, 2))
            for i in range(1, size(P, 2))
                if P[:,i] == Pp[:,j]
                    K[i,j] = k.k(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta) + sigma[1]^2 * sigma[2]^2
                else
                    K[i,j] = k.k(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
                end
            end
        end
    end

    return nothing
end

function covariance_matrix_derivatives!(K_sigma::AbstractMatrix{<:Real}, K_theta::AbstractArray{<:Real, 3}, k::MaternCovariance, P::AbstractMatrix{<:Real}, Pp::AbstractMatrix{<:Real}; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, parallel::Bool=false)
    isconsistent(k)

    if size(K_sigma, 1) != size(P, 2)
        throw(DomainError([size(K_sigma, 1) size(P, 2)], "only supports size(K_sigma, 1) = size(P, 2)."))
    end

    if size(K_sigma, 2) != size(Pp, 2)
        throw(DomainError([size(K_sigma, 2) size(Pp, 2)], "only supports size(K_sigma, 2) = size(Pp, 2)."))
    end

    if size(K_theta, 1) != sum(k.n)
        throw(DomainError([size(K_theta, 1) sum(k.n)], "only supports size(K_theta, 1) = sum(k.n)."))
    end

    if size(K_theta, 2) != size(P, 2)
        throw(DomainError([size(K_theta, 2) size(P, 2)], "only supports size(K_theta, 2) = size(P, 2)."))
    end

    if size(K_theta, 3) != size(Pp, 2)
        throw(DomainError([size(K_theta, 3) size(Pp, 2)], "only supports size(K_theta, 3) = size(Pp, 2)."))
    end

    if size(P, 1) != sum(k.n)
        throw(DomainError([size(P, 1) sum(k.n)], "only supports size(P, 1) = sum(k.n)."))
    end

    if size(Pp, 1) != sum(k.n)
        throw(DomainError([size(Pp, 1) k.n], "only supports size(Pp, 1) = sum(k.n)."))
    end

    if length(sigma) != 2
        throw(DomainError(length(sigma), "only supports length(sigma) = 2."))
    end

    if true in (sigma .<= 0)
        throw(DomainError(sigma, "only supports sigma > 0."))
    end

    if length(theta) != sum(k.n)
        throw(DomainError([length(theta) sum(k.n)], "only supports length(theta) = sum(k.n)."))
    end

    if ~k.reparameterize && (true in (theta .<= 0))
        throw(DomainError(theta, "only supports theta > 0 if k.reparameterize = false."))
    end

    if parallel
        Threads.@threads for j in range(1, size(Pp, 2))
            # inner loop over rows (column-major ordering)
            for i in range(1, size(P, 2))
                if P[:,i] == Pp[:,j]
                    # derivative of additive Gaussian noise
                    K_sigma[i,j] = k.k_sigma(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta) + 2*sigma[1] * sigma[2]^2
                else
                    K_sigma[i,j] = k.k_sigma(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
                end

                K_theta[:,i,j] = k.k_theta(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
            end
        end
    else
        for j in range(1, size(Pp, 2))
            for i in range(1, size(P, 2))
                if P[:,i] == Pp[:,j]
                    K_sigma[i,j] = k.k_sigma(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta) + 2*sigma[1] * sigma[2]^2
                else
                    K_sigma[i,j] = k.k_sigma(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
                end

                K_theta[:,i,j] = k.k_theta(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
            end
        end
    end

    return nothing
end

function covariance_matrix_derivatives!(K_pp::AbstractArray{<:Real, 3}, k::MaternCovariance, P::AbstractMatrix{<:Real}, Pp::VectorOrMatrix; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, parallel::Bool=false)
    isconsistent(k)

    if size(K_pp, 1) != sum(k.n)
        throw(DomainError([size(K_pp, 1) sum(k.n)], "only supports size(K_pp, 1) = sum(k.n)."))
    end

    if size(K_pp, 2) != size(P, 2)
        throw(DomainError([size(K_pp, 2) size(P, 2)], "only supports size(K_pp, 2) = size(P, 2)."))
    end

    if size(K_pp, 3) != size(Pp, 2)
        throw(DomainError([size(K_pp, 3) size(Pp, 2)], "only supports size(K_pp, 3) = size(Pp, 2)."))
    end

    if size(P, 1) != sum(k.n)
        throw(DomainError([size(P, 1) sum(k.n)], "only supports size(P, 1) = sum(k.n)."))
    end

    if size(Pp, 1) != sum(k.n)
        throw(DomainError([size(Pp, 1) sum(k.n)], "only supports size(Pp, 1) = sum(k.n)."))
    end

    if length(sigma) != 2
        throw(DomainError(length(sigma), "only supports length(sigma) = 2."))
    end

    if true in (sigma .<= 0)
        throw(DomainError(sigma, "only supports sigma > 0."))
    end

    if length(theta) != sum(k.n)
        throw(DomainError([length(theta) sum(k.n)], "only supports length(theta) = sum(k.n)."))
    end

    if ~k.reparameterize && (true in (theta .<= 0))
        throw(DomainError(theta, "only supports theta > 0 if k.reparameterize = false."))
    end

    if parallel
        Threads.@threads for j in range(1, size(Pp, 2))
            # inner loop over rows (column-major ordering)
            for i in range(1, size(P, 2))
                K_pp[:,i,j] = k.k_pp(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
            end
        end
    else
        for j in range(1, size(Pp, 2))
            for i in range(1, size(P, 2))
                K_pp[:,i,j] = k.k_pp(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
            end
        end
    end

    return nothing
end

function covariance_matrix_derivatives!(K_pps::AbstractArray{<:Real, 4}, k::MaternCovariance, P::AbstractMatrix{<:Real}, Pp::VectorOrMatrix; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, parallel::Bool=false)
    isconsistent(k)

    if size(K_pps, 1) != sum(k.n)
        throw(DomainError([size(K_pps, 1) sum(k.n)], "only supports size(K_pps, 1) = sum(k.n)."))
    end

    if size(K_pps, 2) != sum(k.n)
        throw(DomainError([size(K_pps, 2) sum(k.n)], "only supports size(K_pps, 2) = sum(k.n)."))
    end

    if size(K_pps, 3) != size(P, 2)
        throw(DomainError([size(K_pps, 3) size(P, 2)], "only supports size(K_pps, 3) = size(P, 2)."))
    end

    if size(K_pps, 4) != size(Pp, 2)
        throw(DomainError([size(K_pps, 4) size(Pp, 2)], "only supports size(K_pps, 4) = size(Pp, 2)."))
    end

    if size(P, 1) != sum(k.n)
        throw(DomainError([size(P, 1) sum(k.n)], "only supports size(P, 1) = sum(k.n)."))
    end

    if size(Pp, 1) != sum(k.n)
        throw(DomainError([size(Pp, 1) sum(k.n)], "only supports size(Pp, 1) = sum(k.n)."))
    end

    if length(sigma) != 2
        throw(DomainError(length(sigma), "only supports length(sigma) = 2."))
    end

    if true in (sigma .<= 0)
        throw(DomainError(sigma, "only supports sigma > 0."))
    end

    if length(theta) != sum(k.n)
        throw(DomainError([length(theta) sum(k.n)], "only supports length(theta) = sum(k.n)."))
    end

    if ~k.reparameterize && (true in (theta .<= 0))
        throw(DomainError(theta, "only supports theta > 0 if k.reparameterize = false."))
    end

    if parallel
        Threads.@threads for j in range(1, size(Pp, 2))
            # inner loop over rows (column-major ordering)
            for i in range(1, size(P, 2))
                K_pps[:,:,i,j] = k.k_pps(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
            end
        end
    else
        for j in range(1, size(Pp, 2))
            for i in range(1, size(P, 2))
                K_pps[:,:,i,j] = k.k_pps(P[:,i], Pp[:,j], sigma=sigma[1], theta=theta)
            end
        end
    end

    return nothing
end

function covariance_matrix_integrals!(Ksi::AbstractArray{<:Real, 3}, k::MaternCovariance, P::AbstractMatrix{<:Real}, Pp::VectorOrMatrix, lb::Real; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, symmetric::Bool=false, parallel::Bool=false)
    isconsistent(k)

    if size(Ksi, 1) != size(P, 2)
        throw(DomainError([size(Ksi, 1) size(P, 2)], "only supports size(Ksi, 1) = size(P, 2)."))
    end

    if size(Ksi, 2) != size(P, 2)
        throw(DomainError([size(Ksi, 2) size(P, 2)], "only supports size(Ksi, 2) = size(P, 2)."))
    end

    if size(Ksi, 3) != size(Pp, 2)
        throw(DomainError([size(Ksi, 3) size(Pp, 2)], "only supports size(Ksi, 3) = size(Pp, 2)."))
    end

    if size(P, 1) != sum(k.n)
        throw(DomainError([size(P, 1) sum(k.n)], "only supports size(P, 1) = sum(k.n)."))
    end

    if size(Pp, 1) != sum(k.n)
        throw(DomainError([size(Pp, 1) sum(k.n)], "only supports size(Pp, 1) = sum(k.n)."))
    end

    if length(sigma) != 2
        throw(DomainError(length(sigma), "only supports length(sigma) = 2."))
    end

    if true in (sigma .<= 0)
        throw(DomainError(sigma, "only supports sigma > 0."))
    end

    if length(theta) != sum(k.n)
        throw(DomainError([length(theta) sum(k.n)], "only supports length(theta) = sum(k.n)."))
    end

    if ~k.reparameterize && (true in (theta .<= 0))
        throw(DomainError(theta, "only supports theta > 0 if k.reparameterize = false."))
    end

    if symmetric && parallel
        Threads.@threads for m in range(1, size(Pp, 2))
            for j in range(1, size(P, 2))
                # inner loop over rows (column-major ordering)
                for i in range(1, j)
                    Ksi[i,j,m] = k.ksi(P[:,i], P[:,j], Pp[:,m], lb, sigma=sigma[1], theta=theta)
                    Ksi[j,i,m] = Ksi[i,j,m]
                end
            end
        end
    end

    if symmetric && ~parallel
        for m in range(1, size(Pp, 2))
            for j in range(1, size(P, 2))
                for i in range(1, j)
                    Ksi[i,j,m] = k.ksi(P[:,i], P[:,j], Pp[:,m], lb, sigma=sigma[1], theta=theta)
                    Ksi[j,i,m] = Ksi[i,j,m]
                end
            end
        end
    end

    if ~symmetric && parallel
        Threads.@threads for m in range(1, size(Pp, 2))
            for j in range(1, size(P, 2))
                for i in range(1, size(P, 2))
                    Ksi[i,j,m] = k.ksi(P[:,i], P[:,j], Pp[:,m], lb, sigma=sigma[1], theta=theta)
                end
            end
        end
    end

    if ~symmetric && ~parallel
        for m in range(1, size(Pp, 2))
            for j in range(1, size(P, 2))
                for i in range(1, size(P, 2))
                    Ksi[i,j,m] = k.ksi(P[:,i], P[:,j], Pp[:,m], lb, sigma=sigma[1], theta=theta)
                end
            end
        end
    end

    return nothing
end
