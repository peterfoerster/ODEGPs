import Pkg
Pkg.activate(".")

using ODEGaussianProcesses
using LaTeXStrings
using Plots

import DataInterpolations as DI
import DelimitedFiles as DF
import Integrals
import Statistics

unicodeplots()
# plotly()

directory = "./csv/brus/"

if ~isdir(directory)
    mkdir(directory)
end

directory_ode = directory*"ode/"

if ~isdir(directory_ode)
    mkdir(directory_ode)
end

directory_rep = directory*"reparameterization/"

if ~isdir(directory_rep)
    mkdir(directory_rep)
end

directory_doe = directory*"doe/"

if ~isdir(directory_doe)
    mkdir(directory_doe)
end

# first component
# max(|f|) ~ 8.8, mean(|f|) ~ 1.9, I(|f|)/t_f ~ 0.8
# max(|ft|) ~ 0.99, mean(|ft|) ~ 0.6, I(|ft|)/t_f ~ 0.37
# second component
# max(|f|) ~ 10.7, mean(|f|) ~ 2.1, I(|f|)/t_f ~ 1
# max(|ft|) ~ 1, mean(|ft|) ~ 0.7, I(|ft|)/t_f ~ 0.6
# t_f       = 25
# k         = [1, 3, 1, 1]
# i_x       = 2
# (t, x, f) = S_brus(0, t_f, k=k, i_x=i_x, abstol=1e-10, reltol=1e-5, directory=directory_ode)
# @info "max(|f|) = $(maximum(abs.(f)))"
# @info "mean(|f|) = $(Statistics.mean(abs.(f)))"
# I = Integrals.SampledIntegralProblem(abs.(f), t)
# S = Integrals.solve(I, Integrals.TrapezoidalRule())
# @info "I(|f|) / t_f = $(S.u/t_f)"
# ft = f ./ sqrt.(1 .+ f.^2)
# @info "max(|ft|) = $(maximum(abs.(ft)))"
# @info "mean(|ft|) = $(Statistics.mean(abs.(ft)))"
# I = Integrals.SampledIntegralProblem(abs.(ft), t)
# S = Integrals.solve(I, Integrals.TrapezoidalRule())
# @info "I(|ft|) / t_f = $(S.u/t_f)"

function brus_reparameterization(t_f::Real, k_min::AbstractVector{<:Real}, k_max::AbstractVector{<:Real}, i_x::Int, n_k::Int, n_s::Int, derivative::Bool, nu::Int, mu::Real, filename::String)
    for j in range(1, length(k_min))
        kh      = collect(range(k_min[j], k_max[j], length=n_k))
        sh      = collect(range(0, 1, length=n_s))
        Ph      = zeros(2, 0)
        tauith  = zeros(0)
        xth     = zeros(0)
        tauitph = zeros(0)

        for i in range(1, n_k)
            Ph = [Ph [sh'; repeat(kh[i:i], 1, n_s)]]

            if j == 1
                k = [kh[i:i]; (k_min[2:4] + k_max[2:4]) / 2]
            elseif j == 2
                k = [(k_min[1] + k_max[1]) / 2; kh[i:i]; (k_min[3:4] + k_max[3:4]) / 2]
            elseif j == 3
                k = [(k_min[1:2] + k_max[1:2]) / 2; kh[i:i]; (k_min[4] + k_max[4]) / 2]
            elseif j == 4
                k = [(k_min[1:3] + k_max[1:3]) / 2; kh[i:i]]
            end

            (t, x, f) = S_brus(0, t_f, k=k, i_x=i_x, abstol=1e-10, reltol=1e-5, directory=directory_ode)

            if derivative
                (tauit, xt, tauitp, t_ext) = reparameterize(t, x, f, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7)
            else
                (tauit, xt, tauitp, t_ext) = reparameterize(t, x, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7)
            end

            append!(tauith, tauit(sh))
            append!(xth, xt(sh))
            append!(tauitph, tauitp.(sh))
        end

        display(heatmap(reshape(tauith, n_s, n_k)', xlabel=L"s", ylabel=L"k", title=L"\tilde{\tau}^{-1}", colorbar=:none))
        display(heatmap(reshape(xth, n_s, n_k)', xlabel=L"s", ylabel=L"k", title=L"\tilde{x}", colorbar=:none))
        display(heatmap(reshape(tauitph, n_s, n_k)', xlabel=L"s", ylabel=L"k", title=L"\frac{d}{ds} \tilde{\tau}^{-1}", colorbar=:none))

        open(filename*"_i_k=$j.csv", "w") do io
            write(io, "s,k,tauit,xt,tauitp\n")
            DF.writedlm(io, [Ph[1,:] Ph[2,:] tauith xth tauitph], ',')
        end
    end

    return nothing
end

t_f        = 25
k_min      = [1-1e-2, 3-3e-2, 1-1e-2, 1-1e-2]
k_max      = [1+1e-2, 3+3e-2, 1+1e-2, 1+1e-2]
i_x        = 1
n_k        = 50
n_s        = 200
derivative = false
nu         = 2
mu         = Inf
filename   = directory_rep*"t_f=$(t_f)_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_n_k=$(n_k)_n_s=$(n_s)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# brus_reparameterization(t_f, k_min, k_max, i_x, n_k, n_s, derivative, nu, mu, filename)

i_x      = 2
filename = directory_rep*"t_f=$(t_f)_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_n_k=$(n_k)_n_s=$(n_s)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)"
# brus_reparameterization(t_f, k_min, k_max, i_x, n_k, n_s, derivative, nu, mu, filename)

function brus_reference(t_f::Real, k_min::AbstractVector{<:Real}, k_max::AbstractVector{<:Real}, i_x::Int, n_k::Int, n_t::Int, filename::String)
    for j in range(1, length(k_min))
        kh = collect(range(k_min[j], k_max[j], length=n_k))
        th = collect(range(0, t_f, length=n_t))
        Ph = zeros(2, 0)
        xh = zeros(0)

        for i in range(1, n_k)
            Ph = [Ph [th'; repeat(kh[i:i], 1, n_t)]]

            if j == 1
                k = [kh[i:i]; (k_min[2:4] + k_max[2:4]) / 2]
            elseif j == 2
                k = [(k_min[1] + k_max[1]) / 2; kh[i:i]; (k_min[3:4] + k_max[3:4]) / 2]
            elseif j == 3
                k = [(k_min[1:2] + k_max[1:2]) / 2; kh[i:i]; (k_min[4] + k_max[4]) / 2]
            elseif j == 4
                k = [(k_min[1:3] + k_max[1:3]) / 2; kh[i:i]]
            end

            (t, x, f) = S_brus(0, t_f, k=k, i_x=i_x, abstol=1e-10, reltol=1e-5, directory=directory_ode)

            xh_i = DI.CubicHermiteSpline(f, x, t)(th)
            append!(xh, xh_i)
        end

        @info "min(xh) = $(minimum(xh))"
        @info "max(xh) = $(maximum(xh))"
        display(heatmap(reshape(xh, n_t, n_k)', xlabel=L"t", ylabel=L"k", colorbar=:none))

        open(filename*"_i_k=$j.csv", "w") do io
            write(io, "t,k,x\n")
            DF.writedlm(io, [Ph[1,:] Ph[2,:] xh], ',')
        end
    end

    return nothing
end

# min(xh) ~ 0.29
# max(xh) ~ 3.85
t_f      = 25
k_min    = [1-1e-2, 3-3e-2, 1-1e-2, 1-1e-2]
k_max    = [1+1e-2, 3+3e-2, 1+1e-2, 1+1e-2]
i_x      = 1
n_k      = 50
n_t      = 200
filename = directory_rep*"t_f=$(t_f)_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_n_k=$(n_k)_n_t=$(n_t)_reference"
# brus_reference(t_f, k_min, k_max, i_x, n_k, n_t, filename)

# min(xh) ~ 0.83
# max(xh) ~ 4.85
i_x      = 2
filename = directory_rep*"t_f=$(t_f)_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_n_k=$(n_k)_n_t=$(n_t)_reference"
# brus_reference(t_f, k_min, k_max, i_x, n_k, n_t, filename)

function brus_conventional_doe(t_f::Real, k_min::AbstractVector{<:Real}, k_max::AbstractVector{<:Real}, i_x::Int, nu_t::Real, nu_p::Real, sigma::AbstractVector{<:Real}, normalize::Bool, standardize::Bool, reltol_x::Real, reltol_k::Real, abstol_p::Real, derivative::Bool, nu::Int, r_lml::Real, filename::String)
    S  = p -> S_brus(0, t_f, k=p, i_x=i_x, abstol=1e-10, reltol=1e-5, directory=directory_ode)
    k  = MaternCovariance(nu_t, sigma=sigma, normalize=normalize, standardize=standardize) * MaternCovariance(nu_p, 4, normalize=normalize, standardize=standardize)
    lb = k_min
    ub = k_max

    @time D = conventional_doe!(k, S, lb, ub, maxiters=400, reltol_x=reltol_x, reltol_k=reltol_k, abstol_p=[1e-6*t_f; abstol_p*(ub - lb)], derivative=derivative, nu=nu, r_lml=r_lml, n_test_p=10, directory=directory_doe, filename=filename)

    P = unique(D.P[2:end,:], dims=2)
    @info "length(P) = $(size(P, 2))"

    display(plot(range(length(D.x)-length(D.e_x), length(D.x)-1), D.e_x, label="", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    return nothing
end

# 382 s
m           = 1
t_f         = 25
k_min       = [1-1e-2, 3-3e-2, 1-1e-2, 1-1e-2]
k_max       = [1+1e-2, 3+3e-2, 1+1e-2, 1+1e-2]
i_x         = 1
nu_t        = Inf
nu_p        = Inf
sigma       = [1, 1e-6]
normalize   = false
standardize = true
reltol_x    = 1e-3
reltol_k    = 1e-3
abstol_p    = 1e-1
derivative  = false
nu          = 2
r_lml       = 0.5
filename    = "conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# brus_conventional_doe(t_f, k_min, k_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 512 s
m        = 1
i_x      = 2
r_lml    = 0.5
filename = "conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# brus_conventional_doe(t_f, k_min, k_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 601 s
m        = 1
i_x      = 1
nu_t     = 2
r_lml    = 0.5
filename = "conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# brus_conventional_doe(t_f, k_min, k_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 165 s
m          = 1
nu_t       = 1
derivative = true
nu         = 1
r_lml      = 0.5
filename   = "conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# brus_conventional_doe(t_f, k_min, k_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 157 s
m          = 1
nu_t       = 0
derivative = false
nu         = 0
r_lml      = 0.5
filename   = "conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# brus_conventional_doe(t_f, k_min, k_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# ? s
m        = 1
i_x      = 2
nu_t     = 2
r_lml    = 0.7
nu       = 2
filename = "conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# brus_conventional_doe(t_f, k_min, k_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 402 s
m          = 1
nu_t       = 1
derivative = true
nu         = 1
r_lml      = 0.5
filename   = "conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# brus_conventional_doe(t_f, k_min, k_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

# 188 s
m          = 1
nu_t       = 0
derivative = false
nu         = 0
r_lml      = 0.7
filename   = "conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
# brus_conventional_doe(t_f, k_min, k_max, i_x, nu_t, nu_p, sigma, normalize, standardize, reltol_x, reltol_k, abstol_p, derivative, nu, r_lml, filename)

function brus_conventional_least_squares(filename::String, I::AbstractVector{Int})
    e_x = DF.readdlm(filename*"_e_x.csv")[:]
    N_x = length(DF.readdlm(filename*"_x.csv")[:])
    N   = range(N_x-length(e_x), N_x-1)

    # least squares estimate
    A = [ones(length(I)) log.(N[I])]
    x = (A' * A) \ (A' * log.(e_x[I]))
    a = exp(x[1])
    b = x[2]
    r = N -> a * N^b

    display(plot(N, [e_x r.(N)], label=[L"e_x" "LS estimate"], xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    open(filename*"_e_l2.csv", "w") do io
        write(io, "N,e_x\n")
        DF.writedlm(io, [N e_x], ',')
    end

    open(filename*"_ls.csv", "w") do io
        write(io, "a,b\n")
        DF.writedlm(io, [a b], ',')
    end

    return nothing
end

m          = 1
k_min      = [1-1e-2, 3-3e-2, 1-1e-2, 1-1e-2]
k_max      = [1+1e-2, 3+3e-2, 1+1e-2, 1+1e-2]
i_x        = 1
nu_t       = Inf
nu_p       = Inf
derivative = false
nu         = 2
filename   = directory_doe*"conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 373)
# brus_conventional_least_squares(filename, I)

m        = 1
nu_t     = 2
filename = directory_doe*"conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 373)
# brus_conventional_least_squares(filename, I)

m          = 1
nu_t       = 1
derivative = true
nu         = 1
filename   = directory_doe*"conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 373)
# brus_conventional_least_squares(filename, I)

m          = 1
nu_t       = 0
derivative = false
nu         = 0
filename   = directory_doe*"conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 373)
# brus_conventional_least_squares(filename, I)

m        = 1
i_x      = 2
nu_t     = Inf
nu       = 2
filename = directory_doe*"conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 373)
# brus_conventional_least_squares(filename, I)

m        = 1
nu_t     = 2
filename = directory_doe*"conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I        = range(10, 373)
# brus_conventional_least_squares(filename, I)

m          = 1
nu_t       = 1
derivative = true
nu         = 1
filename   = directory_doe*"conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 373)
# brus_conventional_least_squares(filename, I)

m          = 1
nu_t       = 0
derivative = false
nu         = 0
filename   = directory_doe*"conventional_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_t=$(nu_t)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_m=$(m)"
I          = range(10, 373)
# brus_conventional_least_squares(filename, I)

function brus_reparameterized_doe(t_f::Real, k_min::AbstractVector{<:Real}, k_max::AbstractVector{<:Real}, i_x::Int, nu_s::Real, nu_p::Real, sigma::AbstractVector{<:Real}, normalize::Bool, standardize::Bool, reltol_v::Real, reltol_k::Real, abstol_p::Real, derivative::Bool, nu::Int, mu::Real, r_lml::Real, filename::String)
    S  = p -> S_brus(0, t_f, k=p, i_x=i_x, abstol=1e-10, reltol=1e-5, directory=directory_ode)
    k  = MaternCovariance(nu_s, sigma=sigma, normalize=normalize, standardize=standardize) * MaternCovariance(nu_p, 4, normalize=normalize, standardize=standardize)
    lb = k_min
    ub = k_max

    @time D_tauit = reparameterized_doe!(k, S, lb, ub, variable="tauit", maxiters=400, reltol_v=reltol_v, reltol_k=reltol_k, abstol_p=[1e-6*t_f; abstol_p*(ub - lb)], derivative=derivative, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7, n_test_p=10, r_lml=r_lml, directory=directory_doe, filename=filename*"_tauit")

    P = unique(D_tauit.P[2:end,:], dims=2)
    @info "length(P) = $(size(P, 2))"

    display(plot(range(length(D_tauit.tauit)-length(D_tauit.e_v), length(D_tauit.tauit)-1), D_tauit.e_v, label=L"e_{\tilde{\tau}^{-1}}", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    @time D_xt = reparameterized_doe!(k, S, lb, ub, variable="xt", maxiters=400, reltol_v=reltol_v, reltol_k=reltol_k, abstol_p=[1e-6*t_f; abstol_p*(ub - lb)], derivative=derivative, nu=nu, mu=mu, abstol=1e-14, reltol=1e-7, n_test_p=10, r_lml=r_lml, directory=directory_doe, filename=filename*"_xt")

    display(plot(range(length(D_xt.xt)-length(D_xt.e_v), length(D_xt.xt)-1), D_xt.e_v, label=L"e_{\tilde{x}}", xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    return nothing
end

# 173/548 s
m           = 1
t_f         = 25
k_min       = [1-1e-2, 3-3e-2, 1-1e-2, 1-1e-2]
k_max       = [1+1e-2, 3+3e-2, 1+1e-2, 1+1e-2]
i_x         = 1
nu_s        = Inf
nu_p        = Inf
sigma       = [1, 1e-6]
normalize   = false
standardize = true
reltol_v    = 1e-3
reltol_k    = 1e-3
abstol_p    = 1e-1
derivative  = false
nu          = 2
mu          = Inf
r_lml       = 0.5
filename    = "reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# brus_reparameterized_doe(t_f, k_min, k_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 131/134 s
m        = 1
nu_s     = 0
r_lml    = 0.5
filename = "reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# brus_reparameterized_doe(t_f, k_min, k_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 166/377 s
m           = 1
nu_s        = 1
derivative  = true
nu          = 1
mu          = 1
r_lml       = 0.5
filename    = "reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# brus_reparameterized_doe(t_f, k_min, k_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 191/576 s
m           = 1
nu_s        = 2
derivative  = false
nu          = 2
mu          = 2
r_lml       = 0.5
filename    = "reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# brus_reparameterized_doe(t_f, k_min, k_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 154/217 s
m           = 1
i_x         = 2
nu_s        = Inf
mu          = Inf
r_lml       = 0.5
filename    = "reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# brus_reparameterized_doe(t_f, k_min, k_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 94/121 s
m        = 1
nu_s     = 0
r_lml    = 0.5
filename = "reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# brus_reparameterized_doe(t_f, k_min, k_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 127/140 s
m           = 1
nu_s        = 1
derivative  = true
nu          = 1
mu          = 1
r_lml       = 0.5
filename    = "reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# brus_reparameterized_doe(t_f, k_min, k_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

# 157/913 s
m           = 1
nu_s        = 2
derivative  = false
nu          = 2
mu          = 2
r_lml       = 0.5
filename    = "reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
# brus_reparameterized_doe(t_f, k_min, k_max, i_x, nu_s, nu_p, sigma, normalize, standardize, reltol_v, reltol_k, abstol_p, derivative, nu, mu, r_lml, filename)

function brus_reparameterized_least_squares(filename::String, I_tauit::AbstractVector{Int}, I_xt::AbstractVector{Int})
    e_tauit = DF.readdlm(filename*"_tauit_e_tauit.csv")[:]
    N_tauit = length(DF.readdlm(filename*"_tauit_tauit.csv")[:])
    N_tauit = range(N_tauit-length(e_tauit), N_tauit-1)
    e_xt    = DF.readdlm(filename*"_xt_e_xt.csv")[:]
    N_xt    = length(DF.readdlm(filename*"_xt_xt.csv")[:])
    N_xt    = range(N_xt-length(e_xt), N_xt-1)

    # least squares estimates
    A = [ones(length(I_tauit)) log.(N_tauit[I_tauit])]
    x = (A' * A) \ (A' * log.(e_tauit[I_tauit]))
    a = exp(x[1])
    b = x[2]
    r = N -> a * N^b

    display(plot(N_tauit, [e_tauit r.(N_tauit)], label=[L"e_{\tilde{\tau}^{-1}}" "LS estimate"], xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    open(filename*"_tauit_e_l2.csv", "w") do io
        write(io, "N,e_tauit\n")
        DF.writedlm(io, [N_tauit e_tauit], ',')
    end

    open(filename*"_tauit_ls.csv", "w") do io
        write(io, "a,b\n")
        DF.writedlm(io, [a b], ',')
    end

    A = [ones(length(I_xt)) log.(N_xt[I_xt])]
    x = (A' * A) \ (A' * log.(e_xt[I_xt]))
    a = exp(x[1])
    b = x[2]
    r = N -> a * N^b

    display(plot(N_xt, [e_xt r.(N_xt)], label=[L"e_{\tilde{x}}" "LS estimate"], xlabel=L"# samples $N$", ylabel=L"relative $l^2$ error", linewidth=1.5, legendfontsize=12, scale=:log10))

    open(filename*"_xt_e_l2.csv", "w") do io
        write(io, "N,e_xt\n")
        DF.writedlm(io, [N_xt e_xt], ',')
    end

    open(filename*"_xt_ls.csv", "w") do io
        write(io, "a,b\n")
        DF.writedlm(io, [a b], ',')
    end

    return nothing
end

m          = 1
k_min      = [1-1e-2, 3-3e-2, 1-1e-2, 1-1e-2]
k_max      = [1+1e-2, 3+3e-2, 1+1e-2, 1+1e-2]
i_x        = 1
nu_s       = Inf
nu_p       = Inf
derivative = false
nu         = 2
mu         = Inf
filename   = directory_doe*"reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 373)
I_xt       = range(10, 373)
# brus_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
nu_s     = 0
filename = directory_doe*"reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 373)
I_xt     = range(10, 373)
# brus_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 1
nu_s       = 1
derivative = true
nu         = 1
mu         = 1
filename   = directory_doe*"reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 373)
I_xt       = range(10, 373)
# brus_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 1
nu_s       = 2
derivative = false
nu         = 2
mu         = 2
filename   = directory_doe*"reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 373)
I_xt       = range(10, 373)
# brus_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 1
i_x        = 2
nu_s       = Inf
mu         = Inf
filename   = directory_doe*"reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 373)
I_xt       = range(10, 373)
# brus_reparameterized_least_squares(filename, I_tauit, I_xt)

m        = 1
nu_s     = 0
filename = directory_doe*"reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit  = range(10, 373)
I_xt     = range(10, 373)
# brus_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 1
nu_s       = 1
derivative = true
nu         = 1
mu         = 1
filename   = directory_doe*"reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 373)
I_xt       = range(10, 373)
# brus_reparameterized_least_squares(filename, I_tauit, I_xt)

m          = 1
nu_s       = 2
derivative = false
nu         = 2
mu         = 2
filename   = directory_doe*"reparameterized_k_min=$(k_min)_k_max=$(k_max)_i_x=$(i_x)_nu_s=$(nu_s)_nu_p=$(nu_p)_derivative=$(derivative)_nu=$(nu)_mu=$(mu)_m=$(m)"
I_tauit    = range(10, 373)
I_xt       = range(10, 373)
# brus_reparameterized_least_squares(filename, I_tauit, I_xt)

# computational cost
# t_conventional          = [382, 601, 165, 157]
# t_reparameterized_tauit = [173, 131, 166, 191]
# t_reparameterized_xt    = [548, 134, 377, 576]
# t_reparameterized       = [t_reparameterized_tauit; t_reparameterized_xt]
# @info "t_x = $(Statistics.mean(t_conventional)) +- $(Statistics.std(t_conventional))"
# @info "t_tauit = $(Statistics.mean(t_reparameterized_tauit)) +- $(Statistics.std(t_reparameterized_tauit))"
# @info "t_xt = $(Statistics.mean(t_reparameterized_xt)) +- $(Statistics.std(t_reparameterized_xt))"
# @info "t_xt(tauit) = $(Statistics.mean(t_reparameterized)) +- $(Statistics.std(t_reparameterized))"
