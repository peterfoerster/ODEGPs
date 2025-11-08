"""
    covariance_matrix(k::MaternCovariance, P::AbstractMatrix{<:Real}, Pp::VectorOrMatrix; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, autodiff::Bool=false, symmetric::Bool=false, posterior::Bool=false, parallel::Bool=false)

Assemble the covariance matrix `K[i,j] = k.k(P[:,i], Pp[:,j])`.
"""
function covariance_matrix(k::MaternCovariance, P::AbstractMatrix{<:Real}, Pp::VectorOrMatrix; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, autodiff::Bool=false, symmetric::Bool=false, posterior::Bool=false, parallel::Bool=false)
    # arguments are checked downstream
    if symmetric && posterior
        throw(DomainError([symmetric posterior], "only supports posterior = true if symmetric = false."))
    end

    if autodiff && ~posterior
        K = zeros(eltype(theta), size(P, 2), size(Pp, 2))
    elseif autodiff && posterior
        K = zeros(eltype(Pp), size(P, 2), size(Pp, 2))
    else
        K = zeros(size(P, 2), size(Pp, 2))
    end

    covariance_matrix!(K, k, P, Pp, sigma=sigma, theta=theta, symmetric=symmetric, parallel=parallel)

    if symmetric
        K = LA.Symmetric(K)
    end

    return K
end

function covariance_matrix_derivatives(k::MaternCovariance, P::AbstractMatrix{<:Real}, Pp::VectorOrMatrix; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, posterior::Bool=false, hessian::Bool=false, parallel::Bool=false)
    # arguments are checked downstream
    if posterior && hessian
        throw(DomainError([posterior hessian], "only supports hessian = true if posterior = false."))
    end

    if ~posterior && ~hessian
        K_sigma = zeros(size(P, 2), size(Pp, 2))
        K_theta = zeros(sum(k.n), size(P, 2), size(Pp, 2))
        covariance_matrix_derivatives!(K_sigma, K_theta, k, P, Pp, sigma=sigma, theta=theta, parallel=parallel)

        return (K_sigma=K_sigma, K_theta=K_theta)
    end

    if posterior
        K_pp = zeros(sum(k.n), size(P, 2), size(Pp, 2))
        covariance_matrix_derivatives!(K_pp, k, P, Pp, sigma=sigma, theta=theta, parallel=parallel)

        return K_pp
    end

    if hessian
        K_pps = zeros(sum(k.n), sum(k.n), size(P,2), size(Pp, 2))
        covariance_matrix_derivatives!(K_pps, k, P, Pp, sigma=sigma, theta=theta, parallel=parallel)

        return K_pps
    end
end

function covariance_matrix_integrals(k::MaternCovariance, P::AbstractMatrix{<:Real}, Pp::VectorOrMatrix, lb::Real; sigma::AbstractVector{<:Real}=k.sigma, theta::AbstractVector{<:Real}=k.theta, symmetric::Bool=false, parallel::Bool=false)
    Ksi = zeros(size(P, 2), size(P, 2), size(Pp, 2))
    covariance_matrix_integrals!(Ksi, k, P, Pp, lb, sigma=sigma, theta=theta, symmetric=symmetric, parallel=parallel)

    return Ksi
end
